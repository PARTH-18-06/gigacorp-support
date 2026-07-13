# GigaCorp Support

GigaCorp Support is a Streamlit-based RAG chatbot for a fictional customer support team. It answers questions from a local FAQ file, shows line-based citations in the response, exposes the retrieved source chunks, and supports follow-up questions through per-session conversational memory.

## Architecture

The project is split into two main files:

- `app.py`
  - Builds the Streamlit UI.
  - Handles API key loading from `st.secrets`, environment variables, or a sidebar password field.
  - Stores chat messages and a per-browser `session_id` in `st.session_state`.
  - Renders chat bubbles, citation chips, and retrieved source cards.
  - Calls the retriever directly to display raw matched chunks.
  - Invokes the LangChain chain for the final answer.

- `rag.py`
  - Loads FAQ content from `data/gigacorp_faq.txt`.
  - Splits the file into one `Document` per blank-line-separated `Q:` / `A:` block.
  - Adds metadata for `source`, `start_line`, `end_line`, and `citation`.
  - Builds or loads a local FAISS index from `faiss_index/`.
  - Uses `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2`.
  - Creates a retriever with `k=3`.
  - Builds an LCEL chain with `ChatGroq(model="llama-3.3-70b-versatile")`.
  - Wraps the chain with `RunnableWithMessageHistory` using an in-memory `SESSION_HISTORIES` dictionary keyed by `session_id`.

## Request Flow

1. A user opens the app and enters a question in the chat input.
2. `app.py` stores the message in `st.session_state.messages`.
3. `app.py` builds a retrieval query with `_build_retrieval_query(...)`.
4. The FAISS retriever fetches the top 3 matching chunks.
5. The chain formats those chunks into a context block with citations.
6. The system prompt tells the model to answer only from retrieved context and always cite sources as `(Source: <file>, lines X-Y)`.
7. The answer is rendered in the UI, and the same retrieved chunks are shown under `Retrieved sources`.
8. Follow-up questions reuse the same `session_id`, so prior turns remain available through message history.

## Local Setup

### 1. Open the project

```powershell
cd "C:\Users\PARTH SHARMA\OneDrive\Desktop\Documents\scaffold\gigacorp-support"
```

### 2. Create a virtual environment

If you do not already have one:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you already use a shorter-path environment, activate that instead.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

You can also use the `GROQ_API_KEY` environment variable, but `secrets.toml` is the simplest local setup.

### 5. Run the app

```powershell
streamlit run app.py
```

If `streamlit` is not on your PATH:

```powershell
python -m streamlit run app.py
```

## Deploy To Streamlit Community Cloud

### 1. Push the project to a public GitHub repository

Make sure the repository is public and contains:

- `app.py`
- `rag.py`
- `requirements.txt`
- `data/gigacorp_faq.txt`
- `.streamlit/config.toml`

Do **not** commit your real `.streamlit/secrets.toml`.

### 2. Sign in to Streamlit Community Cloud

Go to:

- [https://share.streamlit.io](https://share.streamlit.io)

Sign in with your GitHub account.

### 3. Create a new app

In Streamlit Community Cloud:

1. Click `New app`.
2. Select your GitHub repository.
3. Choose the branch, usually `main`.
4. Set the main file path to:

```text
app.py
```

### 4. Add the Groq secret

Before or after the first deploy, open:

- `App settings`
- `Secrets`

Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

This secret stays in Streamlit Cloud settings and is **not** stored in git.

### 5. Deploy

Click `Deploy`.

Streamlit Community Cloud will:

- install packages from `requirements.txt`
- start the Streamlit app
- build the FAISS index on first run if `faiss_index/` is not present in the deployment environment

## Free-Tier Cold Start Note

On the free tier, the app may sleep when inactive. The first visitor after a period of inactivity may see:

- a short startup delay while the app wakes up
- extra delay on the first retrieval request while the embedding model loads
- extra delay if the FAISS index has to be rebuilt in the cloud environment

This is normal for Streamlit Community Cloud free hosting. After warm-up, the app should respond much faster.

## Notes

- Retrieved source chunks are shown separately from the final answer so users can verify where the answer came from.
- Citations are line-based and come directly from the FAQ chunk metadata.
- Reset clears both the visible transcript and the in-memory session history for the active browser session.
