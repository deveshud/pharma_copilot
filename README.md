# PharmaRAG

A Python-based Retrieval-Augmented Generation (RAG) project built from scratch to learn and implement the core mechanics behind modern AI systems.

This project is designed to help understand and build:

- document ingestion
- text cleaning
- chunking
- embeddings
- vector databases
- semantic retrieval
- prompt grounding
- response generation
- citations
- evaluation
- workflow orchestration
- agentic extensions later using LangGraph

The project intentionally avoids relying on managed AI platforms in the beginning so the underlying concepts can be learned hands-on.

---

## Project Objective

Build a local-first RAG system that can:

1. read and process domain documents
2. split them into meaningful chunks
3. convert chunks into embeddings
4. store them in a vector database
5. retrieve the most relevant chunks for a user query
6. generate grounded answers using retrieved context
7. display citations for traceability
8. later extend into workflow-based and agentic systems

---

## Why this project

The goal is not just to "use AI", but to understand how AI applications are actually built.

This project focuses on learning the internals of:

- how embeddings represent meaning
- why chunk size affects retrieval quality
- how similarity search works
- how LAG/LLM grounding changes responses
- where hallucinations come from
- how RAG pipelines are evaluated
- how workflows can later evolve into agents

---

## Use Case Theme

The initial domain focus is pharma / enterprise / data documentation.

Example documents:
- BRDs
- SOWs
- architecture notes
- KPI documents
- process documentation
- data dictionaries
- source-to-target mappings
- project notes
- business rules

Example questions:
- What are the key assumptions in this proposal?
- Which section explains the KPI logic?
- What dependencies are mentioned for access or environments?
- Which document discusses scope?
- What are the out-of-scope items?

---

## Core Features

### Phase 1
- Load local PDF/DOCX/TXT files
- Clean and normalize extracted text
- Split text into chunks
- Generate embeddings
- Store chunk vectors in a local vector database
- Retrieve top-k relevant chunks
- Generate answers using retrieved context
- Show chunk-level citations

### Phase 2
- Add metadata filtering
- Add retrieval tuning
- Compare chunking strategies
- Compare embedding models
- Add evaluation set
- Add answer quality checks

### Phase 3
- Add multi-step retrieval workflows
- Add query rewriting
- Add routing logic
- Add LangGraph orchestration
- Add optional agentic behavior

---

## Architecture

```mermaid
flowchart TD
    A[Documents: PDF / DOCX / TXT] --> B[Document Loader]
    B --> C[Text Cleaner]
    C --> D[Chunking Module]
    D --> E[Embedding Model]
    E --> F[Vector Database]

    Q[User Query] --> Q1[Query Preprocessing]
    Q1 --> Q2[Query Embedding]
    Q2 --> G[Similarity Search]
    F --> G

    G --> H[Top-K Relevant Chunks]
    H --> I[Prompt Builder]
    Q --> I
    I --> J[LLM Response Generator]
    J --> K[Answer with Citations]

    K --> L[Streamlit UI]

    H --> M[Evaluation Layer]
    J --> M