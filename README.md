#  RAG-System-Zero-Data-Leakage-LangGraph

Retrieval-Augmented Generation (RAG) System using [LangGraph](https://github.com/langchain-ai/langgraph), designed to run **fully locally** with **zero data leakage**.

##  Why this project?

Most RAG systems today rely on **cloud-based LLMs** or **external embedding services**, which may expose your **private or sensitive documents**.

This project was built to **completely eliminate that risk**:

-  **100% local-first**: Your files never leave your machine.
-  **Runs on a local LLM**, such as `llama3:8b`, using [Ollama](https://ollama.com).
-  **Uses local embedding models** from HuggingFace or other open-source alternatives.
-  **Stores your vectors locally** with [Chroma](https://github.com/chroma-core/chroma).
-  **Includes memory support** via `ConversationSummaryMemory` to keep context flowing across turns.
-  **Easily configurable**:
  - You can **switch to a more powerful LLM model** (e.g., `llama3:70b`) with a single line in `llm.py`.
  - You can **swap embedding models** in `embeddings.py` (e.g., `intfloat/e5-large-v2`, `all-MiniLM`, etc.).
  - ⚠️ Just **make sure the embedding dimension matches the vector store or delte the old one** (e.g., 1024, 384, or 1536), otherwise you'll face runtime errors.




##  Features

-  **LangGraph-powered**: Modular, state-based agent workflow.
-  **Summary memory**: Keeps conversation flow smooth.
-  **PDF Support**: Load and index local documents easily.
-  **Local VectorStore**: Efficient and isolated using `Chroma`.
-  **Zero API keys required**: Fully air-gapped if needed.
-  **Production-ready structure**: Designed for easy expansion.
  
## Need a Custom Version?
- Contact me directly for custom development

---

##  Installation

> Make sure [Python 3.10+](https://www.python.org/downloads/) is installed.

```bash
# Clone the repo
git clone https://github.com/HUSSAMX7/RAG-System-Zero-Data-Leakage-LangGraph.git

# Install uv if you haven't
pip install uv

# Sync dependencies
uv sync
