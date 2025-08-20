# app.py
import os
from typing import List

# --- Bootstrap & telemetry ----------------------------------------------------
from dotenv import load_dotenv
load_dotenv(override=True)

# LangSmith (optional, safe if absent)
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ollama-streamlit-demo")
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# --- Streamlit UI -------------------------------------------------------------
import streamlit as st

st.set_page_config(
    page_title="LangChain × Ollama — Chat",
    page_icon="🧠",
    layout="wide",
)

# Subtle responsive tweaks
st.markdown(
    """
    <style>
      .block-container {max-width: 1100px;}
      .stChatMessage {font-size: 0.98rem;}
      .stTextInput>div>div>input {font-size: 0.98rem;}
      .small-hint {font-size: 0.85rem; opacity: 0.75;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("LangChain × Ollama — Responsive Chat")

# --- Model wiring (prefer modern package; fall back if unavailable) -----------
try:
    from langchain_ollama import ChatOllama  # pip install langchain-ollama
    USING_CHAT = True
except Exception:
    USING_CHAT = False
    from langchain_community.llms import Ollama  # legacy text-only interface

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# --- Sidebar controls ---------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    # Ollama host (advanced)
    ollama_host = st.text_input(
        "Ollama Host",
        value=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Set if your Ollama server runs remotely or on a non-default host/port.",
    )
    os.environ["OLLAMA_HOST"] = ollama_host  # respected by both integrations

    # Model selection
    default_model = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    model_name = st.text_input("Model", value=default_model)

    # Generation controls
    col_a, col_b = st.columns(2)
    with col_a:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    with col_b:
        top_p = st.slider("Top-p", 0.0, 1.0, 0.95, 0.05)

    col_c, col_d = st.columns(2)
    with col_c:
        num_ctx = st.number_input("Context window (num_ctx)", 1024, 32768, 4096, 256)
    with col_d:
        num_predict = st.number_input("Max tokens (num_predict)", 64, 8192, 512, 64)

    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful assistant. Answer clearly and concisely.",
        height=120,
    )

    enable_langsmith = st.toggle("Enable LangSmith tracing (if key present)", value=bool(LANGCHAIN_API_KEY))
    if not enable_langsmith:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    st.markdown("---")
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.pop("history", None)
        st.rerun()

# --- Session state for chat history ------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []  # list[BaseMessage]

# --- Build LLM & chain --------------------------------------------------------
# Prompt with in-session memory
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

# LLM selection and init
llm_kwargs = {
    "model": model_name,
    "temperature": temperature,
    "top_p": top_p,
}

# Advanced runtime options recognized by Ollama backends
ollama_kwargs = {
    "num_ctx": int(num_ctx),
    "num_predict": int(num_predict),
}

if USING_CHAT:
    # Chat-native interface
    llm = ChatOllama(**llm_kwargs, **ollama_kwargs)
else:
    # Legacy text LLM; we’ll still go through a chat prompt and parse to string
    llm = Ollama(**llm_kwargs, **ollama_kwargs)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# --- Render historical messages ----------------------------------------------
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

def _render_history(history: List[BaseMessage]):
    for msg in history:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(msg.content)

_render_history(st.session_state["history"])

# --- Chat input & streaming response -----------------------------------------
user_input = st.chat_input("Ask me anything…")

def _append_to_history(role: str, content: str):
    if role == "user":
        st.session_state["history"].append(HumanMessage(content=content))
    else:
        st.session_state["history"].append(AIMessage(content=content))

if user_input:
    # echo user
    with st.chat_message("user"):
        st.markdown(user_input)
    _append_to_history("user", user_input)

    # assistant response (stream for responsiveness)
    with st.chat_message("assistant"):
        try:
            # Stream tokens as they arrive
            stream = chain.stream(
                {
                    "system_prompt": system_prompt,
                    "history": st.session_state["history"],
                    "question": user_input,
                }
            )
            # Use Streamlit's write_stream for incremental rendering
            full_text = st.write_stream(stream)

            # Persist assistant message
            _append_to_history("assistant", full_text)

        except Exception as e:
            st.error(
                f"Model invocation failed. Please validate Ollama is running and the model is pulled.\n\nDetails: {e}"
            )
            # Roll back the last user message if failure is critical
            if st.session_state["history"] and isinstance(st.session_state["history"][-1], HumanMessage):
                st.session_state["history"].pop()

# --- Footnotes ----------------------------------------------------------------
st.markdown(
    """
    <div class="small-hint">
    Tip: Ensure your Ollama daemon is active (e.g., <code>ollama serve</code>) and the model is pulled
    (e.g., <code>ollama pull {model}</code>). Update <b>Ollama Host</b> if running remotely.
    </div>
    """.replace("{model}", model_name),
    unsafe_allow_html=True,
)
