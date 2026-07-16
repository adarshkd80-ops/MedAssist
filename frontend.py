"""Streamlit frontend for MedAssist.

Run with:
    uv run streamlit run frontend.py
"""

import itertools
import uuid

import streamlit as st
from groq import RateLimitError
from pydantic import ValidationError

from BackEnd import PatientProfile, app, delete_thread, get_thread_history
from rag import BACKEND, DOCUMENTS_DIR, delete_document, ingest_pdf, list_documents

st.set_page_config(page_title="MedAssist", page_icon="🩺", layout="centered")

PROFILE_DEFAULTS = {
    "p_name": "",
    "p_age": None,
    "p_sex": "",
    "p_weight": None,
    "p_height": None,
    "p_allergies": "",
    "p_conditions": "",
}


def _register_thread(thread_id: str) -> None:
    """Mark a thread as owned by this browser session.

    Privacy: the checkpoint DB is shared by every visitor of a deployment,
    so the sidebar must only ever list threads registered here — never the
    global thread list.
    """
    threads = st.session_state.setdefault("my_threads", [])
    if thread_id in threads:
        threads.remove(thread_id)
    threads.insert(0, thread_id)


def _activate_thread(thread_id: str) -> None:
    """Switch to a conversation and restore its full history."""
    st.session_state.thread_id = thread_id
    _register_thread(thread_id)
    st.session_state.chat_history = [
        {
            "role": msg["role"],
            "content": msg["content"],
            "emergency": msg["content"].startswith("This may be a medical emergency"),
        }
        for msg in get_thread_history(thread_id)
    ]


# ---------- Session state ----------
# One conversation = one thread_id. Every app launch starts a fresh
# conversation; a page refresh stays in the current one via the URL
# (?thread=...), and old chats can be reopened from the sidebar.
if "thread_id" not in st.session_state:
    requested = st.query_params.get("thread")
    if requested:
        _activate_thread(requested)
    else:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        _register_thread(st.session_state.thread_id)

# Keep the thread in the URL so a page refresh stays in the same conversation.
st.query_params["thread"] = st.session_state.thread_id


def _parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _new_chat() -> None:
    """Start a fresh thread and reset the patient profile."""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.patient = {}
    _register_thread(st.session_state.thread_id)
    for key, default in PROFILE_DEFAULTS.items():
        st.session_state[key] = default


def _load_thread(thread_id: str) -> None:
    """Switch to a previous conversation and restore its history."""
    _activate_thread(thread_id)


def _delete_thread(thread_id: str) -> None:
    """Permanently delete a conversation (DB + this session's sidebar)."""
    delete_thread(thread_id)
    st.session_state.my_threads = [
        tid for tid in st.session_state.get("my_threads", []) if tid != thread_id
    ]
    if st.session_state.thread_id == thread_id:
        _new_chat()


# ---------- Sidebar ----------
with st.sidebar:
    st.title("🩺 MedAssist")
    st.caption(
        "AI health information assistant. Not a substitute for professional "
        "medical advice — in an emergency, call your local emergency number."
    )

    st.button("➕ New chat", use_container_width=True, on_click=_new_chat)

    st.subheader("Patient profile")
    # A form: the profile only takes effect when "Save profile" is clicked,
    # not on every keystroke.
    with st.form("profile_form"):
        name = st.text_input("Name", key="p_name")
        age = st.number_input(
            "Age", min_value=0, max_value=130, value=None, step=1, key="p_age"
        )
        sex = st.selectbox("Sex", ["", "male", "female", "other"], key="p_sex")
        weight_kg = st.number_input(
            "Weight (kg)", min_value=0.0, value=None, key="p_weight"
        )
        height_cm = st.number_input(
            "Height (cm)", min_value=0.0, value=None, key="p_height"
        )
        allergies = st.text_input("Allergies (comma-separated)", key="p_allergies")
        conditions = st.text_input(
            "Existing conditions (comma-separated)", key="p_conditions"
        )
        save_profile = st.form_submit_button(
            "💾 Save profile", use_container_width=True
        )

    if save_profile:
        # Validate through PatientProfile so out-of-range values (age, weight,
        # height) never reach the graph; on failure, warn and keep the last
        # saved profile rather than crashing.
        try:
            profile = PatientProfile(
                name=name,
                age=age,
                sex=sex or None,
                weight_kg=weight_kg or None,
                height_cm=height_cm or None,
                allergies=_parse_list(allergies),
                conditions=_parse_list(conditions),
            )
            # Only keep fields the user actually filled in.
            st.session_state.patient = {
                key: value for key, value in profile.model_dump().items() if value
            }
            if st.session_state.patient:
                st.success("Profile saved — answers will be personalized.")
            else:
                st.info("Profile cleared — answers will be general.")
        except ValidationError as exc:
            bad_fields = ", ".join(str(err["loc"][0]) for err in exc.errors())
            st.warning(
                f"Profile not saved — invalid value(s) for: {bad_fields}. "
                "Fix the field(s) and save again."
            )

    st.subheader("📚 Knowledge base")
    st.caption(
        "Upload medical reference PDFs (guidelines, leaflets, formularies). "
        "Answers are grounded in these documents when relevant. "
        f"Embeddings: `{BACKEND}`."
    )
    uploads = st.file_uploader(
        "Add PDFs", type="pdf", accept_multiple_files=True, key="kb_uploads"
    )
    if uploads and st.button("Index documents", use_container_width=True):
        DOCUMENTS_DIR.mkdir(exist_ok=True)
        with st.spinner("Indexing…"):
            for file in uploads:
                dest = DOCUMENTS_DIR / file.name
                dest.write_bytes(file.getbuffer())
                try:
                    n_chunks = ingest_pdf(dest)
                except Exception:
                    # Corrupt/encrypted/not-really-a-PDF upload: remove the
                    # saved copy so it doesn't linger in the documents dir.
                    dest.unlink(missing_ok=True)
                    st.error(
                        f"❌ Could not read '{file.name}'. Please provide a "
                        "correct, uncorrupted PDF file."
                    )
                else:
                    if n_chunks == 0:
                        # Loaded fine but produced no text — typically a
                        # scanned/image-only PDF, which can't be indexed.
                        dest.unlink(missing_ok=True)
                        st.warning(
                            f"'{file.name}' contains no readable text (it may "
                            "be a scanned or image-only PDF). Please provide "
                            "a correct text-based PDF."
                        )
                    else:
                        st.toast(f"Indexed {file.name} ({n_chunks} chunks)")
    kb_docs = list_documents()
    if kb_docs:
        with st.expander(f"Indexed documents ({len(kb_docs)})"):
            for doc_name in kb_docs:
                col_name, col_del = st.columns([5, 1])
                col_name.markdown(doc_name)
                if col_del.button(
                    "🗑️",
                    key=f"doc_del_{doc_name}",
                    help=f"Remove '{doc_name}' from the knowledge base",
                ):
                    delete_document(doc_name)
                    (DOCUMENTS_DIR / doc_name).unlink(missing_ok=True)
                    st.rerun()

    st.subheader("Previous chats")
    # Privacy: list only threads started in THIS browser session — the
    # checkpoint DB is shared by every visitor of a deployment, so the
    # global thread list must never be shown.
    my_threads = st.session_state.get("my_threads", [])
    shown_any = False
    for tid in list(my_threads):
        history = get_thread_history(tid)
        first_user_msg = next(
            (m["content"] for m in history if m["role"] == "user"), None
        )
        if first_user_msg is None:
            continue
        shown_any = True
        label = first_user_msg[:40] + ("…" if len(first_user_msg) > 40 else "")
        is_current = tid == st.session_state.thread_id
        col_open, col_del = st.columns([5, 1])
        col_open.button(
            ("🟢 " if is_current else "💬 ") + label,
            key=f"thread_{tid}",
            use_container_width=True,
            disabled=is_current,
            on_click=_load_thread,
            args=(tid,),
        )
        col_del.button(
            "🗑️",
            key=f"del_{tid}",
            on_click=_delete_thread,
            args=(tid,),
            help="Delete this conversation permanently",
        )
    if not shown_any:
        st.caption("No conversations in this session yet.")

# The profile only applies once saved via the sidebar's Save button.
patient = st.session_state.get("patient") or {}


# ---------- Chat area ----------
st.title("MedAssist Chat")

# First-run hint, shown until the conversation starts.
if not st.session_state.chat_history:
    if patient:
        st.info("✅ Patient profile saved — answers will take it into account.")
    else:
        st.info(
            "👋 **Welcome to MedAssist!** If you'd like answers personalized "
            "to you, fill in the patient profile in the sidebar and click "
            "**💾 Save profile** — otherwise, just type your health question "
            "below to get started."
        )

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message.get("emergency"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        if message.get("tool_results"):
            with st.expander("🔧 MCP tool results"):
                st.json(message["tool_results"])

prompt = st.chat_input("Describe your symptoms or ask a health question…")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        config = {
            "configurable": {"thread_id": st.session_state.thread_id},
            # Groups traces into conversations in LangSmith's Threads view.
            "metadata": {"thread_id": st.session_state.thread_id},
            "run_name": "medassist-chat",
        }

        def _token_stream():
            """Yield answer tokens as the graph's LLM calls generate them.

            stream_mode="messages" surfaces every LLM token in the graph;
            filtering by node keeps the triage classifier's structured
            output out of the visible answer.
            """
            for chunk, metadata in app.stream(
                {"messages": [("user", prompt)], "patient": patient},
                config,
                stream_mode="messages",
            ):
                if (
                    metadata.get("langgraph_node")
                    in ("symptom_checker", "general_answer", "greeting_response")
                    and isinstance(chunk.content, str)
                    and chunk.content
                ):
                    yield chunk.content

        # Spinner covers the silent phase (triage + tool calls); once the
        # first token arrives, hand the stream to st.write_stream. The
        # emergency path never streams (no LLM), so the generator may
        # finish without yielding anything.
        tokens = _token_stream()
        try:
            with st.spinner("Thinking…"):
                first_token = next(tokens, None)
            if first_token is not None:
                st.write_stream(itertools.chain([first_token], tokens))
        except RateLimitError:
            # The SDK already retried with backoff (max_retries on ChatGroq),
            # so reaching here means the quota is genuinely exhausted — e.g.
            # the free tier's daily token cap. Recover instead of crashing.
            st.session_state.chat_history.pop()
            st.error(
                "The AI service is temporarily over its usage limit. "
                "Please wait a minute and resend your message — if it keeps "
                "happening, the daily quota may be used up until it resets."
            )
            st.stop()

        result = app.get_state(config).values
        response = result["final_response"]
        is_emergency = result.get("query_type") == "emergency"
        # tool_results persists in the checkpointed thread, so only show it
        # on the turn that actually produced it.
        tool_results = (
            result.get("tool_results")
            if result.get("query_type") in ("symptom", "general")
            else None
        )

        # The normal answer was already rendered token-by-token above; only
        # the emergency banner (or a stream that produced nothing) still
        # needs to be drawn here.
        if is_emergency:
            st.error(response)
        elif first_token is None:
            st.markdown(response)
        if tool_results:
            with st.expander("🔧 MCP tool results"):
                st.json(tool_results)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response,
            "emergency": is_emergency,
            "tool_results": tool_results,
        }
    )
    # A new thread just got its first checkpoint — refresh so it appears
    # in the sidebar immediately.
    st.rerun()
