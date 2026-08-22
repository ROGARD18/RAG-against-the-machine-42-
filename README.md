# RAG Against the Machine

*This project has been created as part of the 42 curriculum by arogard.*

## Description

This project implements a local Retrieval-Augmented Generation (RAG) system from scratch. The goal is to ingest technical documentation and codebase files (specifically the vLLM project), index them, retrieve the most relevant snippets based on user queries, and generate accurate, context-aware answers using a local Large Language Model.

## System Architecture

The pipeline consists of three main interacting components:

- **Indexer** (`indexer_utils.py`): Parses raw `.md` and `.py` files, splits them into manageable text chunks, tokenizes the text, and builds a sparse search index saved locally.
- **Retriever** (`cli_interface.py`): Takes a user query, tokenizes it, and queries the index to fetch the top *k* most relevant chunks.
- **Generator** (`llm_model.py`): Formats the retrieved chunks as context and prompts a local LLM to generate a concise, factual answer based exclusively on that context.

## Chunking Strategy

Documents are segmented using LangChain's `RecursiveCharacterTextSplitter` to ensure chunks remain coherent:

- **Markdown files**: Split with a maximum chunk size of 2000 characters and a 10% overlap to preserve contextual continuity across paragraphs.
- **Python files**: Split using the Python-specific language splitter. This respects code semantics (functions, classes) with a chunk overlap of 20% to avoid cutting critical logic in half.

## Retrieval Method

Retrieval relies on the **BM25Okapi** algorithm.

Queries and documents are tokenized by removing special characters, converting to lowercase, and keeping words longer than 2 characters. BM25 ranks the chunks based on term frequency and inverse document frequency (TF-IDF principles). This provides fast, keyword-driven retrieval that is highly effective for technical documentation and code.

## Performance Analysis

The system's retrieval performance was evaluated against a strict ground truth dataset, requiring at least a 5% character index overlap between the predicted chunk and the expected chunk:

| Metric | Score |
|--------|-------|
| Recall@1 | 31.3% |
| Recall@3 | 45.5% |
| Recall@5 | 50.5% |
| Recall@10 | 57.6% |

These scores establish a solid baseline for a purely sparse, keyword-based retrieval method without the computational overhead of semantic dense embeddings.

## Design Decisions

- **BM25 over embeddings**: Chosen for its lightweight, fast CPU execution. It eliminates the need for GPU acceleration during the retrieval phase.
- **Local LLM (`Qwen/Qwen3-0.6B`)**: Selected for its excellent balance between a small memory footprint and capable technical reasoning.
- **Strict generation prompting**: The system prompt forces the LLM to reply exactly with `"Information not found in context."` if it cannot confidently answer. This drastically reduces hallucinations.
- **CLI framework**: Utilized `Fire` for a clean, minimal command-line interface.

## Challenges Faced

- **Recall calculation**: Replicating the evaluation logic of the grading system was complex. Initially, a naive file-path matching approach artificially inflated the score to 64%. The solution required implementing a precise mathematical overlap calculation (minimum 5% character overlap) to accurately reflect true retrieval performance.
- **Type hinting & linting**: Ensuring strict adherence to PEP 8 (Flake8) and static typing (Mypy) required refactoring union types and handling `Optional` values carefully, particularly around the LLM loading phases.

## Instructions

### Prerequisites

- Python 3.10+
- `uv` package manager

### Installation

```bash
# Setup the virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt
```

### Execution

All commands are run through the `src` module using `uv`.

```bash
# 1. Index the raw data
uv run python -m src index

# 2. Search for a query (returns top 5 chunks)
uv run python -m src search "What is vLLM?" --k 5

# 3. Ask a question to the LLM
uv run python -m src answer "How do you enable expert parallelism in vLLM?" --k 5

# 4. Run automated datasets and evaluate
uv run python -m src search_dataset data/datasets/UnansweredQuestions/dataset.json --k 10 --save_directory data/output/
uv run python -m src evaluate data/output/dataset.json data/datasets/AnsweredQuestions/dataset.json
```

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [LangChain Text Splitters](https://python.langchain.com/docs/how_to/#text-splitters)
- [BM25 Algorithm (Okapi)](https://en.wikipedia.org/wiki/Okapi_BM25)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)

## AI Usage

AI was used as a pair-programming assistant during this project. Specifically, it was utilized to:

- Debug and align the custom recall calculation logic with the strict requirements of the automated grading tool.
- Perform static code analysis to resolve Mypy typing errors and Flake8 formatting inconsistencies.