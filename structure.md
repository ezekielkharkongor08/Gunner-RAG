SUGGESTED PROJECT STRUCTURE
arsenal-wikipedia-rag/
├── README.md
├── .env.example
├── requirements.txt
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── ingestion/
│   │   ├── wikipedia_loader.py
│   │   ├── document_cleaner.py
│   │   ├── text_splitter.py
│   │   └── index_builder.py
│   ├── retrieval/
│   │   └── retriever.py
│   ├── generation/
│   │   └── answer_chain.py
│   ├── database/
│   │   └── vector_store.py
│   └── app.py
└── tests/
    ├── test_wikipedia_loader.py
    └── test_retrieval.py

when downloading always use uv pip install 


TECH STACK
- LLM: llama3.1:8b through Ollama
- Embeddings: nomic-embed-text:v1.5 via ollama
- Vector database: PostgreSQL with the pgvector extension
- Framework: LangChain
- Sole external content source: Wikipedia API
- use the langchain in build wikipedia things