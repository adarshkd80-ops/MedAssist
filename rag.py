

"""RAG pipeline for MedAssist.

Medical reference PDFs (guidelines, formularies, patient leaflets) are
ingested into a persistent Chroma vector store. The graph's
`retrieve_context` node queries this store to ground symptom assessments
and general answers in the uploaded documents.

Embedding backends (chosen at startup):
- **ollama** — local Ollama server with `nomic-embed-text`; preferred when
  reachable (typical for local development).
- **fastembed** — ONNX model running on CPU in-process; needs no server or
  API key, so it is the automatic fallback on deployments like Streamlit
  Cloud where Ollama does not exist.

Each backend writes to its own Chroma collection: vectors from different
embedding models have different dimensions and are not comparable, so a
knowledge base indexed with one backend must be re-indexed to be visible
to the other.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

load_dotenv()

DOCUMENTS_DIR = Path(__file__).parent / "documents"
PERSIST_DIR = str(Path(__file__).parent / "medassist_kb")

OLLAMA_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
FASTEMBED_MODEL = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
# "auto" prefers Ollama and falls back to FastEmbed; set to "ollama" or
# "fastembed" to force one backend.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto").lower()


def _ollama_embeddings():
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    # OllamaEmbeddings connects lazily; embed something now so an
    # unreachable server fails here, where "auto" can still fall back.
    embeddings.embed_query("ping")
    return embeddings


def _fastembed_embeddings():
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=FASTEMBED_MODEL)


def _select_backend() -> tuple[str, object]:
    if EMBEDDING_BACKEND == "ollama":
        return "ollama", _ollama_embeddings()
    if EMBEDDING_BACKEND == "fastembed":
        return "fastembed", _fastembed_embeddings()
    try:
        return "ollama", _ollama_embeddings()
    except Exception:
        return "fastembed", _fastembed_embeddings()


BACKEND, _embeddings = _select_backend()

# Passages scoring below this cosine relevance are treated as noise and
# dropped rather than stuffed into the prompt. Calibrated per model:
# nomic-embed-text scores off-topic ~0.33 / on-topic 0.7+; bge-small
# scores off-topic ~0.46 / on-topic 0.6+.
MIN_RELEVANCE = {"ollama": 0.45, "fastembed": 0.55}[BACKEND]

vector_store = Chroma(
    # The original (Ollama-era) collection keeps its name so existing local
    # knowledge bases stay intact.
    collection_name="medassist_kb" if BACKEND == "ollama" else "medassist_kb_fastembed",
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
