# Gunner RAG: Local Wikipedia Retrieval-Augmented Generation

Gunner RAG is a production-grade, local question-answering and conversational search system built over Wikipedia domain knowledge for Arsenal F.C.

The pipeline ingests raw prose directly via the MediaWiki Action API, cleans and structures text with section-aware chunking, indexes embeddings in PostgreSQL using pgvector, and executes grounded answering through local Ollama models with verifiable citations.

---

## Key Features

- **Clean Wikipedia Ingestion**: Downloads articles directly through the official Wikipedia API. Drops tables, infoboxes, and references so the database stores only clear, readable paragraphs.
- **Section-Aware Chunking**: Splits text according to Wikipedia sections rather than blind character counts. Combines tiny sections to preserve flow and tags every chunk with its article title and section name for context.
- **Smart Section Routing**: If a question mentions a specific section (such as match months or team kits), the system routes directly to that section instead of guessing with vector search.
- **Relative Result Filtering**: Compares search results against the top match rather than using a rigid score cutoff, keeping relevant passages from being dropped.
- **Fallback Query Rewriting**: If the first search returns weak matches, the system rewrites the question, searches again, and merges the best results using Reciprocal Rank Fusion (RRF).
- **Season Guarding**: Automatically detects seasons mentioned in questions and filters search results to that season, preventing stats and matches from different years from mixing together.
- **Real Citations**: The AI only writes numbered markers like `[1]`. The actual article title, section name, and Wikipedia link are created by code, ensuring sources are never made up.
- **Fact-Checking (Grounding Check)**: Compares the generated answer against the source passages to flag any names or numbers that do not appear in the retrieved text.

---

## System Architecture

```
Wikipedia (MediaWiki Action API)
        |
        v
src/ingestion/wikipedia_loader.py       [Extract raw page text & sections]
        |
        v
src/ingestion/document_cleaner.py       [Strip wiki templates, HTML, junk sections]
        |
        v
src/ingestion/section_assembler.py      [Filter boilerplate, merge short sections]
        |
        v
src/ingestion/text_splitter.py          [Split into overlapping chunks with header]
        |
        v
src/ingestion/index_builder.py          [Generate embeddings via Ollama -> pgvector]
        |
        +-----------------------------------+
        |                                   |
        v                                   v
src/retrieval/section_router.py     src/retrieval/retriever.py
[Deterministic section targeting]   [Vector search, RRF fusion & relative gating]
        |                                   |
        +-----------------+-----------------+
                          |
                          v
        src/generation/answer_chain.py
        [Prompt assembly, local LLM generation, citation formatting]
                          |
                          v
        src/generation/grounding.py
        [Entity and numerical grounding verification]
                          |
        +-----------------+-----------------+
        |                 |                 |
        v                 v                 v
     CLI UI          FastAPI API       Streamlit Web App
   (ask.py / main.py)   (api.py)           (app.py)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language & Runtime | Python 3.13 |
| Orchestration Framework | LangChain |
| Chat Model (Local LLM) | Llama 3.1 8B via Ollama |
| Embedding Model | Qwen3-Embedding-0.6B via Ollama |
| Vector Store | PostgreSQL with pgvector extension |
| Content Source | MediaWiki Action API |
| Backend API | FastAPI, Pydantic, Uvicorn |
| Web Interface | Streamlit |

---

## Project Structure

```
Arsenal_Wiki/
├── config/
│   └── settings.yaml
├── src/
│   ├── config.py
│   ├── text_utils.py
│   ├── ingestion/
│   │   ├── wikipedia_loader.py
│   │   ├── fetch_cache.py
│   │   ├── document_cleaner.py
│   │   ├── section_assembler.py
│   │   ├── text_splitter.py
│   │   ├── index_builder.py
│   │   └── run_ingestion.py
│   ├── retrieval/
│   │   ├── retriever.py
│   │   └── section_router.py
│   └── generation/
│       ├── answer_chain.py
│       └── grounding.py
├── eval/
├── tests/
├── api.py
├── app.py
├── ask.py
├── main.py
├── requirements.txt
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.13+
- PostgreSQL instance with the `vector` extension enabled
- Ollama service running locally with the required models:
  ```bash
  ollama pull llama3.1:8b
  ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0
  ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ezekielkharkongor08/Gunner-RAG.git
   cd Gunner-RAG
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Update `.env` with your PostgreSQL database credentials, Ollama endpoint, and collection names.

4. Verify system dependencies:
   ```bash
   python main.py
   ```

---

## Usage

### 1. Ingest Data

Fetch configured articles, clean text, split into sections, and build the vector index:

```bash
python src/ingestion/run_ingestion.py
python src/ingestion/run_ingestion.py --titles "Arsenal F.C." "History of Arsenal F.C."
```

### 2. Command Line Interface

```bash
python ask.py "Who did Arsenal play against in the 2025-26 EFL Cup final?"
python ask.py --check "What was Arsenal's average possession?"
python main.py --chat
python main.py --summarize "2025–26 Arsenal F.C. season"
```

### 3. FastAPI Service

Start the REST API server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Key endpoints:
- `GET  /api/status` - Database, Ollama, and vector index health metrics.
- `POST /api/ask` - Single-turn Q&A with structured citations and grounding validation.
- `POST /api/chat` - Multi-turn conversational Q&A with context memory.
- `POST /api/summarize` - Retrieve-and-summarize pipeline for full articles.
- `POST /api/ingest` - Dynamic ingestion of new Wikipedia titles.

### 4. Streamlit Web Interface

Launch the interactive dashboard:

```bash
streamlit run app.py
```

---

## Evaluation and Benchmarks

Run retrieval performance evaluation and data quality checks:

```bash
python tests/run_quality_checks.py --collection arsenal_documents
python eval/retrieval_eval.py --collection arsenal_documents
```
