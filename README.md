#  RAG-System-Zero-Data-Leakage-LangGraph

Retrieval-Augmented Generation (RAG) System using [LangGraph](https://github.com/langchain-ai/langgraph), designed to run **fully locally** with **zero data leakage**.

## Why this project?

Most RAG systems today rely on cloud-based LLMs or external embedding services, which might expose your sensitive documents.

This project solves that:

-  100% local: Your documents never leave your machine.
-  Runs on local LLM (e.g., `llama3:8b`) via [Ollama](https://ollama.com).
-  Uses local embeddings via HuggingFace or other open models.
-  Vector database: [`Chroma`](https://github.com/chroma-core/chroma) stored locally.
-  Memory: Powered by `ConversationSummaryMemory` to retain context.

---

##  Features

-  **LangGraph-powered**: Modular, state-based agent workflow.
-  **Summary memory**: Keeps conversation flow smooth.
-  **PDF Support**: Load and index local documents easily.
-  **Local VectorStore**: Efficient and isolated using `Chroma`.
-  **Zero API keys required**: Fully air-gapped if needed.
-  **Production-ready structure**: Designed for easy expansion.

---

##  Installation

> Make sure [Python 3.10+](https://www.python.org/downloads/) is installed.

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/RAG-System-Zero-Data-Leakage-LangGraph.git
cd RAG-System-Zero-Data-Leakage-LangGraph

# Install uv if you haven't
pip install uv

# Sync dependencies
uv sync
