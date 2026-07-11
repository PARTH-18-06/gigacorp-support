import os
from uuid import uuid4

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from rag import SESSION_HISTORIES, _build_retrieval_query, build_chain


st.set_page_config(page_title="GigaCorp Support", page_icon=":speech_balloon:")
st.title("GigaCorp Support")
st.caption("Ask questions about shipping, returns, business hours, and memberships.")


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


def render_sources(sources):
    with st.expander("Retrieved sources"):
        if not sources:
            st.write("No sources retrieved.")
            return

        for index, source in enumerate(sources, start=1):
            st.markdown(f"**Source {index}:** `{source['citation']}`")
            st.code(source["content"], language="text")


st.sidebar.header("Settings")

if st.sidebar.button("Reset conversation"):
    reset_conversation()
    st.rerun()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

groq_api_key = get_groq_api_key()
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
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

prompt = st.chat_input("Ask a GigaCorp support question")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

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
        with st.spinner("Thinking..."):
            response = chain.invoke(
                {"question": prompt},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
        st.markdown(response)
        render_sources(source_entries)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": source_entries}
    )
