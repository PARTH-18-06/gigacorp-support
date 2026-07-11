from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


BASE_DIR = Path(__file__).resolve().parent
FAQ_PATH = BASE_DIR / "data" / "gigacorp_faq.txt"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
SESSION_HISTORIES = {}


def load_qa_chunks(path):
    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8").splitlines()
    documents = []
    block_lines = []
    block_start = None
    block_end = None

    def flush_block():
        nonlocal block_lines, block_start, block_end

        if not block_lines:
            return

        if block_lines[0].startswith("Q:"):
            content = "\n".join(block_lines)
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file_path.name,
                        "start_line": block_start,
                        "end_line": block_end,
                        "citation": f"{file_path.name}, lines {block_start}-{block_end}",
                    },
                )
            )

        block_lines = []
        block_start = None
        block_end = None

    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            if block_start is None:
                block_start = line_number
            block_lines.append(line)
            block_end = line_number
        else:
            flush_block()

    flush_block()
    return documents


def build_or_load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    index_files_exist = (
        FAISS_INDEX_DIR.is_dir()
        and (FAISS_INDEX_DIR / "index.faiss").exists()
        and (FAISS_INDEX_DIR / "index.pkl").exists()
    )

    if index_files_exist:
        return FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    documents = load_qa_chunks(FAQ_PATH)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(FAISS_INDEX_DIR))
    return vectorstore


def format_docs_with_citations(docs):
    formatted_chunks = []

    for doc in docs:
        citation = doc.metadata.get("citation", doc.metadata.get("source", "unknown"))
        formatted_chunks.append(f"[Source: {citation}]\n{doc.page_content}")

    return "\n\n".join(formatted_chunks)


def _get_session_history(session_id):
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = InMemoryChatMessageHistory()
    return SESSION_HISTORIES[session_id]


def _build_retrieval_query(inputs):
    history = inputs.get("history", [])
    question = inputs["question"]

    if not history:
        return question

    history_lines = []
    for message in history:
        role = "User" if message.type in {"human", "user"} else "Assistant"
        history_lines.append(f"{role}: {message.content}")

    history_lines.append(f"User: {question}")
    return "\n".join(history_lines)


def build_chain():
    vectorstore = build_or_load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    system_prompt = (
        "You are GigaCorp's customer support assistant. Answer ONLY using the "
        "retrieved context. Every factual claim must be supported by the context "
        "and you must cite sources in the format '(Source: <file>, lines X-Y)'. "
        "If the retrieved context does not contain the answer, say you don't know "
        "based on the available context. Do not use outside knowledge.\n\n"
        "Retrieved context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(_build_retrieval_query)
            | retriever
            | RunnableLambda(format_docs_with_citations)
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=_get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history, retriever


if __name__ == "__main__":
    chunks = load_qa_chunks(FAQ_PATH)

    print(f"Chunk count: {len(chunks)}")
    for index, chunk in enumerate(chunks[:3], start=1):
        print(f"\nChunk {index}:")
        print(chunk.page_content)
        print(chunk.metadata)
