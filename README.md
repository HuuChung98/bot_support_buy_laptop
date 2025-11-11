# Laptop Recommendation Chatbot

An experimental conversational assistant that recommends laptops using:

- Azure OpenAI (for embeddings + chat)
- Pinecone (vector store)
- LangChain (retrieval + chain wiring)

This project demonstrates a small Retrieval-Augmented Generation (RAG) pipeline combined with function calling to produce accurate, context-aware recommendations.

## Quick summary

- Input: user conversation about needs (usage, budget, preferred features)
- Retrieval: nearest-neighbor search over laptop descriptions (Pinecone)
- Response: Azure OpenAI chat model with optional function calls

## Requirements

- Python 3.8+
- Access to Azure OpenAI (embeddings + chat deployment)
- Pinecone account and API key

## Environment variables

Create a `.env` file in the project root and set the variables below. Replace `your_*` with actual values.

```env
# Embeddings (Azure OpenAI)
AZURE_OPENAI_EMBEDDING_API_KEY=your_embedding_api_key
AZURE_OPENAI_EMBEDDING_ENDPOINT=your_embedding_endpoint
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Chat / LLM (Azure OpenAI)
AZURE_OPENAI_LLM_API_KEY=your_llm_api_key
AZURE_OPENAI_LLM_ENDPOINT=your_llm_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
```

Notes:
- The exact embedding model name and deployment name depend on your Azure configuration.

## Installation

1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

If you don't have `requirements.txt`, install at minimum:

```powershell
pip install langchain-pinecone langchain-openai pinecone-client python-dotenv openai streamlit
```

## Files you care about

- `bot_support_by_laptop.py` — main script: builds embeddings, upserts to Pinecone, sets up LangChain retriever and function metadata, runs example queries, and exposes `process_user_message` used by the UI.
- `ui.py` — Streamlit UI that provides a simple chat interface and calls `process_user_message`.
- `data/laptops.json` — (optional) laptop catalog used to build the vector store.

## How to run

1. Ensure `.env` is configured and your virtual environment is active.

2. To run the CLI/script demo (predefined queries):

```powershell
python bot_support_by_laptop.py
```

3. To run the Streamlit UI (recommended for interactive chat):

```powershell
streamlit run ui.py
```

Open the Streamlit URL printed in your terminal (usually `http://localhost:8501`).

## Expected behavior

- On first run the script creates a Pinecone index (if missing), computes embeddings for the small laptop dataset, and upserts them.
- The demo queries show RAG answers and may also invoke function-calling logic.
- The Streamlit UI stores chat history in the session and displays RAG + function-call answers.

## Troubleshooting

- Authentication errors: confirm the correct API keys/endpoints in `.env`.
- Pinecone index errors: ensure the `PINECONE_API_KEY` is valid and your Pinecone project/region match the script assumptions.
- Azure deployment errors: check the `AZURE_OPENAI_DEPLOYMENT_NAME` and model availability.

## Next improvements (suggested)

- Add a small script to re-index `data/laptops.json` on-demand.
- Add unit tests for `process_user_message` and function handlers.
- Improve error handling and logging for API calls.

## License

MIT-style. Use as you wish for experimentation and learning.

---
If you'd like, I can also:
- scan other project files and translate remaining non-English comments to English, or
- add a short `requirements.txt` and a simple re-index script.
Tell me which you'd prefer.