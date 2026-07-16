from typing import Annotated, Literal, Optional, TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import asyncio
from fastmcp import Client
from langsmith import traceable
from mcp_server import mcp as med_tools_server
from rag import format_context, retrieve



import os

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")


class PatientProfile(BaseModel):
    """Structured patient context attached to a conversation."""

    name: str = ""
    age: Optional[int] = Field(default=None, ge=0, le=130)
    sex: Optional[str] = None
    weight_kg: Optional[float] = Field(default=None, gt=0)
    height_cm: Optional[float] = Field(default=None, gt=0)
    allergies: list[str] = Field(default_factory=list, description="Known allergies (drugs, food, environmental)")
    conditions: list[str] = Field(default_factory=list, description="Pre-existing diagnosed conditions")



class MedState(TypedDict):
    """State passed between every node in the graph.

    `messages` uses the add_messages reducer so each node can append
    without overwriting conversation history.
    """

    messages: Annotated[list, add_messages]
    patient: dict
    query_type: str          # "symptom" | "general" | "emergency"
    symptoms: list[str]
    symptom_duration: Optional[str]
    tool_results: dict       # raw outputs from the MCP tools
    retrieved_docs: list     # knowledge-base passages from the RAG retriever
    final_response: str


llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)


class QueryClassification(BaseModel):
    """Structured output for the classifier node."""

    query_type: Literal["symptom", "general", "emergency"]
    symptoms: list[str] = Field(default_factory=list, description="Symptoms mentioned by the patient")
    symptom_duration: Optional[str] = Field(default=None, description="How long symptoms have been present")


def _patient_context(state: MedState) -> str:
    patient = state.get("patient") or {}
    if not patient:
        return "No patient profile on file."
    return "Patient profile: " + ", ".join(f"{k}={v}" for k, v in patient.items() if v)


@traceable(run_type="tool", name="mcp_tool")
def call_mcp_tool(name: str, args: dict):
    """Call a tool on the MedAssist FastMCP server (in-process, no transport)."""

    async def _run():
        async with Client(med_tools_server) as client:
            result = await client.call_tool(name, args)
            return result.data

    return asyncio.run(_run())


## Nodes

def classify_query(state: MedState) -> dict:
    """Entry node: classify the user's message and extract symptoms."""
    classifier = llm.with_structured_output(QueryClassification)
    result = classifier.invoke(
        [
            SystemMessage(
                content=(
                    "You are a medical triage classifier. Classify the patient's message as "
                    "'emergency' (life-threatening red flags: chest pain, stroke signs, severe "
                    "bleeding, anaphylaxis, suicidal ideation), 'symptom' (describing symptoms "
                    "and seeking guidance), or 'general' (medication questions, health education). "
                    f"{_patient_context(state)}"
                )
            ),
            *state["messages"],
        ]
    )
    return {
        "query_type": result.query_type,
        "symptoms": result.symptoms,
        "symptom_duration": result.symptom_duration,
    }


def emergency_response(state: MedState) -> dict:
    """Short-circuit path: tell the user to seek emergency care immediately."""
    response = (
        "This may be a medical emergency. Please call your local emergency number "
        "(e.g. 112 in India, 911 in the US) or go to the nearest emergency department now. "
        "I am an AI assistant and cannot handle emergencies."
    )
    return {"final_response": response, "messages": [("assistant", response)]}


def retrieve_context(state: MedState) -> dict:
    """RAG node: pull relevant knowledge-base passages for the user's message.

    Runs before both the symptom and general paths. Returns [] when the
    knowledge base is empty or nothing relevant is found, in which case
    the downstream nodes answer without document grounding.
    """
    query = state["messages"][-1].content
    return {"retrieved_docs": retrieve(query)}


def _kb_context(state: MedState) -> str:
    """Prompt block for retrieved passages, or a note that none were found."""
    context = format_context(state.get("retrieved_docs") or [])
    if not context:
        return "No relevant passages were found in the local knowledge base."
    return (
        "Relevant passages from the local medical knowledge base — prefer these "
        "over general knowledge and cite them as (source, page):\n" + context
    )


def symptom_checker(state: MedState) -> dict:
    """Analyze extracted symptoms using the MCP tools + patient profile."""
    patient = state.get("patient") or {}
    symptoms = state.get("symptoms") or []
    tool_results: dict = {}

    if symptoms:
        tool_results["red_flags"] = call_mcp_tool(
            "check_symptom_red_flags", {"symptoms": symptoms}
        )
    if patient.get("weight_kg") and patient.get("height_cm"):
        tool_results["bmi"] = call_mcp_tool(
            "calculate_bmi",
            {"weight_kg": patient["weight_kg"], "height_cm": patient["height_cm"]},
        )
    assessment = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a cautious medical information assistant, not a doctor. "
                    "Given the symptoms, patient profile, and tool results below, suggest "
                    "possible common causes, self-care guidance, and clear signs that "
                    "warrant seeing a clinician. If the red-flag tool found anything, "
                    "lead with that warning. Never diagnose or prescribe. "
                    f"{_patient_context(state)} "
                    f"Symptoms: {symptoms}; duration: {state.get('symptom_duration')}. "
                    f"Tool results: {tool_results} "
                    f"{_kb_context(state)}"
                )
            ),
            *state["messages"],
        ]
    )
    # Shown in the UI's tool-results expander; added after the LLM call so
    # the passages reach the prompt only once, via _kb_context.
    if state.get("retrieved_docs"):
        tool_results["knowledge_base"] = state["retrieved_docs"]
    return {"tool_results": tool_results, "final_response": assessment.content}


def general_answer(state: MedState) -> dict:
    """Answer general health / medication questions using RAG + web search."""
    question = state["messages"][-1].content
    search = call_mcp_tool("web_search", {"query": question, "max_results": 5})
    tool_results = {"web_search": search}

    answer = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a medical information assistant. Answer general health questions "
                    "accurately and simply, remind the user you are not a substitute for a "
                    "clinician. Ground your answer in the knowledge-base passages and web "
                    "search results below, citing document sources and URLs you relied on. "
                    "If both are empty or irrelevant, answer from your own knowledge and "
                    f"say so. {_patient_context(state)} "
                    f"{_kb_context(state)} "
                    f"Web search results: {search}"
                )
            ),
            *state["messages"],
        ]
    )
    if state.get("retrieved_docs"):
        tool_results["knowledge_base"] = state["retrieved_docs"]
    return {"tool_results": tool_results, "final_response": answer.content}


def generate_response(state: MedState) -> dict:
    """Final node: append the drafted response to the conversation."""
    return {"messages": [("assistant", state["final_response"])]}


def route_query(state: MedState) -> str:
    """Conditional edge: branch on the classifier's query_type."""
    return state["query_type"]


## Adding Nodes to my Graph
Graph = StateGraph(MedState)
Graph.add_node("classify_query", classify_query)
Graph.add_node("emergency_response", emergency_response)
Graph.add_node("retrieve_context", retrieve_context)
Graph.add_node("symptom_checker", symptom_checker)
Graph.add_node("general_answer", general_answer)
Graph.add_node("generate_response", generate_response)

## Wiring Edges
Graph.add_edge(START, "classify_query")
# Emergencies skip retrieval entirely; everything else is grounded in the
# knowledge base first, then routed to its specialist node.
Graph.add_conditional_edges(
    "classify_query",
    route_query,
    {
        "emergency": "emergency_response",
        "symptom": "retrieve_context",
        "general": "retrieve_context",
    },
)
Graph.add_conditional_edges(
    "retrieve_context",
    route_query,
    {
        "symptom": "symptom_checker",
        "general": "general_answer",
    },
)
Graph.add_edge("symptom_checker", "generate_response")
Graph.add_edge("general_answer", "generate_response")
Graph.add_edge("emergency_response", END)
Graph.add_edge("generate_response", END)

# Persistent checkpointer: every conversation thread is saved to SQLite,
# so history survives restarts. check_same_thread=False is required because
# FastAPI/Streamlit may touch the connection from different threads.
conn = sqlite3.connect("medassist_checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
# Create the checkpoint tables immediately: SqliteSaver only sets them up on
# the first write, but list_threads() queries them on startup — on a fresh
# database (e.g. a new deployment) that would raise sqlite3.OperationalError.
checkpointer.setup()
app = Graph.compile(checkpointer=checkpointer)


## Helpers for the frontend to browse saved conversations

def list_threads() -> list[str]:
    """All thread_ids in the checkpoint DB, most recently active first."""
    rows = conn.execute(
        "SELECT thread_id, MAX(rowid) AS latest FROM checkpoints "
        "GROUP BY thread_id ORDER BY latest DESC"
    ).fetchall()
    return [row[0] for row in rows]


def get_thread_history(thread_id: str) -> list[dict]:
    """Rebuild a displayable chat history from a thread's checkpointed state."""
    state = app.get_state({"configurable": {"thread_id": thread_id}})
    messages = (state.values or {}).get("messages", [])
    history = []
    for msg in messages:
        if msg.type == "human":
            history.append({"role": "user", "content": msg.content})
        elif msg.type == "ai":
            history.append({"role": "assistant", "content": msg.content})
    return history



