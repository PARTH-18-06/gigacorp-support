import html
import os
import re
from uuid import uuid4

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from rag import SESSION_HISTORIES, _build_retrieval_query, build_chain


st.set_page_config(page_title="GigaCorp Support", page_icon="📦", layout="centered")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap');

:root {
    --bg: #0F1729;
    --surface: #1B2740;
    --surface-2: #21304F;
    --accent: #F5A623;
    --accent-soft: rgba(245, 166, 35, 0.16);
    --success: #2DD4BF;
    --success-soft: rgba(45, 212, 191, 0.12);
    --text: #E8ECF4;
    --muted: #8B96AD;
    --border: rgba(139, 150, 173, 0.18);
    --shadow: 0 18px 40px rgba(4, 10, 24, 0.32);
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(45, 212, 191, 0.08), transparent 28%),
        radial-gradient(circle at top left, rgba(245, 166, 35, 0.08), transparent 30%),
        linear-gradient(180deg, #101a2f 0%, var(--bg) 55%);
    color: var(--text);
}

.block-container {
    max-width: 920px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

[data-testid="stHeader"] {
    background: rgba(15, 23, 41, 0.75);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(27, 39, 64, 0.98), rgba(17, 25, 43, 0.98));
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stCaption {
    color: var(--text);
}

[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border);
    color: var(--text);
}

[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px rgba(245, 166, 35, 0.45);
}

[data-testid="stSidebar"] button[kind="secondary"] {
    background: transparent;
    border: 1px solid rgba(139, 150, 173, 0.32);
    color: var(--muted);
}

[data-testid="stSidebar"] button[kind="secondary"]:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(245, 166, 35, 0.08);
}

.shell {
    position: relative;
}

.control-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: linear-gradient(135deg, rgba(27, 39, 64, 0.96), rgba(17, 26, 45, 0.92));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem 1.1rem 1rem 1.2rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}

.control-header__icon {
    width: 3rem;
    height: 3rem;
    border-radius: 14px;
    display: grid;
    place-items: center;
    background: linear-gradient(145deg, rgba(245, 166, 35, 0.2), rgba(245, 166, 35, 0.06));
    border: 1px solid rgba(245, 166, 35, 0.28);
    font-size: 1.35rem;
}

.control-header__eyebrow {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.35rem;
}

.control-header__title {
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    color: var(--text);
    line-height: 1;
}

.control-header__subtitle {
    margin-top: 0.4rem;
    color: var(--muted);
    font-size: 0.95rem;
}

.status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin: 0.25rem 0 1rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.5rem 0.78rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border);
    font-size: 0.85rem;
    color: var(--muted);
}

.status-pill__dot {
    width: 0.62rem;
    height: 0.62rem;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 0 5px rgba(45, 212, 191, 0.12);
    flex: 0 0 auto;
}

.status-pill--inactive .status-pill__dot {
    background: var(--accent);
    box-shadow: 0 0 0 5px rgba(245, 166, 35, 0.12);
}

.deck-label {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
}

[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0;
    margin-bottom: 0.9rem;
}

[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
    margin-top: 0.4rem;
}

.message-bubble {
    width: min(100%, 780px);
    border-radius: 18px;
    border: 1px solid var(--border);
    padding: 0.95rem 1.05rem;
    box-shadow: var(--shadow);
    line-height: 1.65;
}

.message-bubble p {
    margin: 0;
}

.message-bubble p + p {
    margin-top: 0.8rem;
}

.message-bubble--assistant {
    margin-right: auto;
    background: linear-gradient(180deg, rgba(27, 39, 64, 0.98), rgba(24, 35, 56, 0.94));
    border-left: 4px solid rgba(45, 212, 191, 0.72);
}

.message-bubble--user {
    margin-left: auto;
    background: linear-gradient(180deg, rgba(245, 166, 35, 0.18), rgba(245, 166, 35, 0.12));
    border-left: 4px solid rgba(245, 166, 35, 0.72);
}

.message-label {
    display: inline-block;
    margin-bottom: 0.7rem;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.message-label--assistant {
    color: var(--success);
}

.message-label--user {
    color: #ffd899;
}

.message-bubble--assistant strong,
.message-bubble--user strong {
    color: var(--text);
}

.citation-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.16rem 0.48rem;
    margin: 0 0.12rem;
    border-radius: 999px;
    border: 1px solid rgba(45, 212, 191, 0.45);
    background: rgba(45, 212, 191, 0.08);
    color: var(--success);
    font-family: "JetBrains Mono", monospace;
    font-size: 0.77em;
    letter-spacing: 0.05em;
    white-space: nowrap;
}

[data-testid="stExpander"] {
    border: 0;
}

[data-testid="stExpander"] details {
    border-radius: 16px;
    background: rgba(27, 39, 64, 0.72);
    border: 1px solid var(--border);
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    background: rgba(255, 255, 255, 0.02);
    color: var(--text);
}

[data-testid="stExpander"] summary:hover {
    background: rgba(255, 255, 255, 0.04);
}

.source-stack {
    display: grid;
    gap: 0.85rem;
    margin-top: 0.4rem;
}

.source-card {
    background: linear-gradient(180deg, rgba(15, 23, 41, 0.88), rgba(22, 31, 49, 0.92));
    border: 1px solid rgba(139, 150, 173, 0.2);
    border-radius: 14px;
    padding: 0.9rem;
}

.source-card__meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.65rem;
}

.source-card__index {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--muted);
}

.source-card__citation {
    display: inline-flex;
    align-items: center;
    padding: 0.22rem 0.55rem;
    border-radius: 8px;
    border: 1px solid rgba(45, 212, 191, 0.45);
    background: rgba(45, 212, 191, 0.08);
    color: var(--success);
    font-family: "JetBrains Mono", monospace;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
}

.source-card pre {
    white-space: pre-wrap;
    word-break: break-word;
    margin: 0;
    padding: 0.9rem;
    border-radius: 10px;
    background: rgba(232, 236, 244, 0.03);
    color: #c7d0df;
    border: 1px solid rgba(139, 150, 173, 0.14);
    font-family: "JetBrains Mono", monospace;
    font-size: 0.8rem;
    line-height: 1.65;
}

[data-testid="stChatInput"] {
    background: rgba(27, 39, 64, 0.9);
    border: 1px solid var(--border);
    border-radius: 18px;
    box-shadow: var(--shadow);
}

[data-testid="stChatInput"] textarea {
    font-family: "JetBrains Mono", monospace;
    color: var(--text);
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted);
    font-family: "JetBrains Mono", monospace;
    letter-spacing: 0.03em;
}

[data-testid="stChatInput"] textarea:focus {
    box-shadow: none !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(245, 166, 35, 0.65);
    box-shadow: 0 0 0 1px rgba(245, 166, 35, 0.3), var(--shadow);
}

[data-testid="stChatInputSubmitButton"] {
    background: rgba(245, 166, 35, 0.16);
    border-radius: 12px;
}

[data-testid="stChatInputSubmitButton"]:hover {
    background: rgba(245, 166, 35, 0.26);
}

.support-note {
    margin-top: 0.6rem;
    color: var(--muted);
    font-size: 0.92rem;
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem;
    }

    .control-header {
        align-items: flex-start;
    }

    .control-header__title {
        font-size: 1.5rem;
        letter-spacing: 0.12em;
    }

    .message-bubble {
        width: 100%;
        padding: 0.85rem 0.9rem;
    }

    .status-row {
        flex-direction: column;
        align-items: flex-start;
    }
}
</style>
"""


SOURCE_PATTERN = re.compile(r"\(Source:\s*([^)]+)\)")


def get_groq_api_key():
    secret_key = st.secrets.get("GROQ_API_KEY")
    env_key = os.getenv("GROQ_API_KEY")
    manual_key = st.sidebar.text_input("Groq API key", type="password")
    return secret_key or env_key or manual_key


@st.cache_resource
def get_chain(api_key):
    os.environ["GROQ_API_KEY"] = api_key
    return build_chain()


def reset_conversation():
    session_id = st.session_state.get("session_id")
    if session_id:
        SESSION_HISTORIES.pop(session_id, None)

    st.session_state.session_id = str(uuid4())
    st.session_state.messages = []


def to_history_messages(messages):
    history = []

    for message in messages:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            history.append(AIMessage(content=message["content"]))

    return history


def format_message_text(content):
    safe_content = html.escape(content)
    safe_content = SOURCE_PATTERN.sub(
        r'<span class="citation-chip">Source: \1</span>', safe_content
    )
    paragraphs = [
        paragraph.replace("\n", "<br>")
        for paragraph in safe_content.split("\n\n")
        if paragraph.strip()
    ]
    return "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)


def render_message(role, content):
    bubble_class = "message-bubble--user" if role == "user" else "message-bubble--assistant"
    label_class = "message-label--user" if role == "user" else "message-label--assistant"
    label = "Operator query" if role == "user" else "Dispatch response"
    message_html = format_message_text(content)

    st.markdown(
        f"""
        <div class="message-bubble {bubble_class}">
            <div class="message-label {label_class}">{label}</div>
            {message_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources):
    with st.expander("Retrieved sources"):
        if not sources:
            st.write("No sources retrieved.")
            return

        st.markdown('<div class="source-stack">', unsafe_allow_html=True)
        for index, source in enumerate(sources, start=1):
            citation = html.escape(source["citation"])
            content = html.escape(source["content"])
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-card__meta">
                        <span class="source-card__index">Manifest {index:02d}</span>
                        <span class="source-card__citation">{citation}</span>
                    </div>
                    <pre>{content}</pre>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.sidebar.markdown("### Operations")

if st.sidebar.button("Reset conversation", use_container_width=True):
    reset_conversation()
    st.rerun()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

groq_api_key = get_groq_api_key()
status_class = "status-pill" if groq_api_key else "status-pill status-pill--inactive"
status_label = "API key active" if groq_api_key else "API key required"

st.sidebar.markdown(
    f"""
    <div class="{status_class}">
        <span class="status-pill__dot"></span>
        <span>{status_label}</span>
    </div>
    <div class="support-note">
        Reset clears the visible transcript and the in-memory session history.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="shell">
        <div class="deck-label">Control channel // live dispatch board</div>
        <div class="status-row">
            <div class="control-header">
                <div class="control-header__icon">📦</div>
                <div>
                    <div class="control-header__eyebrow">Support Desk</div>
                    <div class="control-header__title">GIGACORP</div>
                    <div class="control-header__subtitle">
                        Customer support routing for shipping, returns, service tiers, and contact operations.
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not groq_api_key:
    st.info(
        "Provide a Groq API key in `st.secrets[\"GROQ_API_KEY\"]`, the "
        "`GROQ_API_KEY` environment variable, or the sidebar input. "
        "Create one at [console.groq.com/keys](https://console.groq.com/keys)."
    )
    st.stop()

chain, retriever = get_chain(groq_api_key)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_message(message["role"], message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

prompt = st.chat_input("Enter your query...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        render_message("user", prompt)

    history_messages = to_history_messages(st.session_state.messages[:-1])
    retrieval_query = _build_retrieval_query(
        {"history": history_messages, "question": prompt}
    )
    retrieved_docs = retriever.invoke(retrieval_query)
    source_entries = [
        {
            "citation": doc.metadata.get("citation", "unknown"),
            "content": doc.page_content,
        }
        for doc in retrieved_docs
    ]

    with st.chat_message("assistant"):
        with st.spinner("Routing through dispatch..."):
            response = chain.invoke(
                {"question": prompt},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
        render_message("assistant", response)
        render_sources(source_entries)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": source_entries}
    )
