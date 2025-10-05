# Agentic RAG PoC

A proof-of-concept (PoC) for an agentic Retrieval-Augmented Generation (RAG) system using LangChain, LangGraph, and Ollama LLMs. This project demonstrates how to build a modular, agent-driven workflow for document retrieval, grading, rewriting, and answer generation.

## Features
- **Document Loading & Splitting:** Fetches and splits web documents for processing.
- **Vector Store Setup:** Embeds and stores document chunks for retrieval using ChromaDB and Ollama embeddings.
- **Agent Node:** Handles user queries, invokes retrieval tools, and interacts with the LLM.
- **Document Grading:** Grades retrieved documents for relevance.
- **Rewrite Node:** Improves user questions if needed.
- **Answer Generation:** Generates answers using only the provided context.
- **Graph Workflow:** Orchestrates the above steps using LangGraph.

## Installation

### Requirements
- Python 3.12
- [Ollama](https://ollama.com/) (for local LLM inference)

### Install dependencies

You can install all dependencies using pip:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```
```pswl
powershell -c "irm https://astral.sh/uv/install.ps1 | sh"
uv sync
```

## Project Structure

- `main.py` — Example entry point
- `agents/` — Core agent, node, and workflow logic
    - `load_documents.py` — Loads and splits documents
    - `vectorstore_setup.py` — Sets up vector store and retriever
    - `nodes.py` — Agent, grading, rewrite, and generate nodes
    - `build_graph.py` — Workflow graph construction
    - `state_schema.py` — State schema for LangGraph
- `pyproject.toml` — Dependencies

## Key Dependencies
- `langchain`, `langchain-community`, `langchain-core`, `langchain-ollama`, `langchain-text-splitters`
- `langgraph`, `langgraph-prebuilt`, `langgraph-sdk`
- `chromadb`, `ollama`
- `pydantic`, `typer`, `rich`, `requests`, `beautifulsoup4`

See `pyproject.toml` for the full list.


