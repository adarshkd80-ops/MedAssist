"""RAG pipeline for MedAssist.

Medical reference PDFs (guidelines, formularies, patient leaflets) are
ingested into a persistent Chroma vector store using local Ollama
embeddings. The graph's `retrieve_context` node queries this store to
ground symptom assessments and general answers in the uploaded documents.

Requires the Ollama server to be running locally with the embedding
model pulled (`ollama pull nomic-embed-text`).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

load_dotenv()

DOCUMENTS_DIR = Path(__file__).parent / "documents"
PERSIST_DIR = str(Path(__file__).parent / "medassist_kb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Passages scoring below this cosine relevance are treated as noise and
# dropped rather than stuffed into the prompt. Calibrated for
# nomic-embed-text: off-topic queries score ~0.33, on-topic ones 0.7+.
MIN_RELEVANCE = 0.45

_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name="medassist_kb",
    embedding_function=_embeddings,
    persist_directory=PERSIST_DIR,
    collection_metadata={"hnsw:space": "cosine"},
)

_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def ingest_pdf(path: str | Path) -> int:
    """Load, chunk, embed, and index one PDF into the knowledge base.

    Re-ingesting a file with the same name replaces its old chunks, so
    uploading an updated version of a document never duplicates it.
    Returns the number of chunks indexed.
    """
    path = Path(path)
    pages = PyPDFLoader(str(path)).load()
    chunks = _splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["source"] = path.name

    existing = vector_store.get(where={"source": path.name})
    if existing["ids"]:
        vector_store.delete(ids=existing["ids"])
    if chunks:
        vector_store.add_documents(chunks)
    return len(chunks)


def list_documents() -> list[str]:
    """Names of all documents currently in the knowledge base."""
    metadatas = vector_store.get(include=["metadatas"])["metadatas"]
    return sorted({meta.get("source", "unknown") for meta in metadatas})


def delete_document(name: str) -> int:
    """Remove every chunk belonging to one document. Returns chunks removed."""
    existing = vector_store.get(where={"source": name})
    if existing["ids"]:
        vector_store.delete(ids=existing["ids"])
    return len(existing["ids"])


@traceable(run_type="retriever", name="knowledge_base_retrieve")
def retrieve(query: str, k: int = 4) -> list[dict]:
    """Top-k knowledge-base passages relevant to the query.

    Returns [] when the store is empty, nothing clears the relevance
    threshold, or Ollama is unreachable — callers can always fall back
    to answering without document grounding.
    """
    try:
        results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        return []
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "score": round(score, 3),
            "content": doc.page_content,
        }
        for doc, score in results
        if score >= MIN_RELEVANCE
    ]


def format_context(passages: list[dict]) -> str:
    """Render retrieved passages as a prompt block the LLM can cite."""
    if not passages:
        return ""
    blocks = []
    for i, p in enumerate(passages, 1):
        page = f", page {p['page'] + 1}" if p.get("page") is not None else ""
        blocks.append(f"[Doc {i}: {p['source']}{page}]\n{p['content']}")
    return "\n\n".join(blocks)
