SUGGESTED PROJECT STRUCTURE
arsenal-weki/
├── README.md
├── .env.example
├── .env
├── requirements.txt
├── pyproject.toml
├── config/
│   └── settings.yaml
├── src/
│   ├── __init__.py
│   ├── config.py                 # settings from .env + config/settings.yaml
│   ├── main.py                   # smoke check (settings, PostgreSQL, Ollama)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── wikipedia_loader.py   # fetch Wikipedia via MediaWiki Action API, prose only
│   │   ├── document_cleaner.py   # clean prose (no tables)
│   │   ├── text_splitter.py      # split prose into chunks (no tables)
│   │   ├── index_builder.py      # embed + store prose chunks in pgvector
│   │   └── run_ingestion.py      # orchestrator: load -> clean -> split -> build_index
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py          # pgvector similarity search + citations (prose only)
│   └── generation/               # (not implemented yet)
│       └── answer_chain.py       # (placeholder)
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_wikipedia_loader.py
│   ├── test_document_cleaner.py
│   ├── test_text_splitter.py
│   ├── test_index_builder.py
│   ├── test_retrieval.py
│   └── test_integration.py       # optional, gated on ARSENAL_WEKI_INTEGRATION=1
└── data/                         # no longer used for ingestion (Wikipedia-only)

TECH STACK
- LLM: llama3.1:8b through Ollama
- Embeddings: Qwen3-Embedding-0.6B via Ollama
- Vector database: PostgreSQL with the pgvector extension
- Framework: LangChain
- Content source: Wikipedia only, via the MediaWiki Action API (action=query&prop=extracts)
- Extraction: prose only — no HTML scraping, no BeautifulSoup, no tables/infoboxes
