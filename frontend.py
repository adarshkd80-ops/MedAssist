"""Streamlit frontend for MedAssist.

Run with:
    uv run streamlit run frontend.py
"""

import uuid

import streamlit as st

from BackEnd import app, get_thread_history, list_threads

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


def _activate_thread(thread_id: str) -> None:
    """Switch to a conversation and restore its full history."""
    st.session_state.thread_id = thread_id
    st.session_state.chat_history = [
        {
            "role": msg["role"],
            "content": msg["content"],
            "emergency": msg["content"].startswith("This may be a medical emergency"),
        }
        for msg in get_thread_history(thread_id)
    ]


# ---------- Session state ----------
# One conversation = one thread_id. On startup, resume the thread from the
# URL (?thread=...) if present, otherwise the most recent saved conversation.
# Only the "New chat" button creates a fresh thread_id.
if "thread_id" not in st.session_state:
    requested = st.query_params.get("thread")
    saved_threads = list_threads()
    if requested:
        _activate_thread(requested)
    elif saved_threads:
        _activate_thread(saved_threads[0])
    else:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []

# Keep the thread in the URL so a page refresh stays in the same conversation.
st.query_params["thread"] = st.session_state.thread_id


def _parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _new_chat() -> None:
    """Start a fresh thread and reset the patient profile."""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    for key, default in PROFILE_DEFAULTS.items():
        st.session_state[key] = default


def _load_thread(thread_id: str) -> None:
    """Switch to a previous conversation and restore its history."""
    _activate_thread(thread_id)


# ---------- Sidebar ----------
with st.sidebar:
    st.title("🩺 MedAssist")
    st.caption(
        "AI health information assistant. Not a substitute for professional "
        "medical advice — in an emergency, call your local emergency number."
    )

    st.button("➕ New chat", use_container_width=True, on_click=_new_chat)

    st.subheader("Patient profile")
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

    st.subheader("Previous chats")
    threads = list_threads()
    if not threads:
        st.caption("No conversations yet.")
    for tid in threads:
        history = get_thread_history(tid)
        first_user_msg = next(
            (m["content"] for m in history if m["role"] == "user"), None
        )
        if first_user_msg is None:
            continue
        label = first_user_msg[:40] + ("…" if len(first_user_msg) > 40 else "")
        is_current = tid == st.session_state.thread_id
        st.button(
            ("🟢 " if is_current else "💬 ") + label,
            key=f"thread_{tid}",
            use_container_width=True,
            disabled=is_current,
            on_click=_load_thread,
            args=(tid,),
            help=f"thread_id: {tid}",
        )

# Only include fields the user actually filled in.
patient = {
    key: value
    for key, value in {
        "name": name,
        "age": age,
        "sex": sex or None,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "allergies": _parse_list(allergies),
        "conditions": _parse_list(conditions),
    }.items()
    if value
}


# ---------- Chat area ----------
st.title("MedAssist Chat")
st.caption(f"Conversation: `{st.session_state.thread_id[:8]}…`")

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
        with st.spinner("Thinking…"):
            config = {
                "configurable": {"thread_id": st.session_state.thread_id},
                # Groups traces into conversations in LangSmith's Threads view.
                "metadata": {"thread_id": st.session_state.thread_id},
                "run_name": "medassist-chat",
            }
            result = app.invoke(
                {"messages": [("user", prompt)], "patient": patient},
                config,
            )

        response = result["final_response"]
        is_emergency = result.get("query_type") == "emergency"
        # tool_results persists in the checkpointed thread, so only show it
        # on the turn that actually produced it.
        tool_results = (
            result.get("tool_results")
            if result.get("query_type") in ("symptom", "general")
            else None
        )

        if is_emergency:
            st.error(response)
        else:
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
