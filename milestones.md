# PharmaRAG Weekly Milestones

## Week 1 — Setup and foundation
**Goal:** Create the project base and understand the end-to-end RAG flow at a high level.

### Tasks
- Create Git repo
- Create project folder structure
- Create virtual environment
- Install dependencies
- Add `README.md`
- Add `requirements.txt`
- Add `.gitignore`
- Add `.env.example`
- Create `data/sample_docs/`
- Collect 5–10 sample documents
- Create a basic Streamlit app shell
- Create initial document loader

### Concepts to learn
- What RAG is
- RAG vs normal LLM chat
- What tokens are
- Why chunking is needed
- What embeddings do
- High-level vector search flow

### Deliverable
- A running repo with sample documents and basic app structure

---

## Week 2 — Document parsing and text cleaning
**Goal:** Extract readable text from files and clean it for downstream processing.

### Tasks
- Build PDF parser
- Build DOCX parser
- Build TXT reader
- Normalize line breaks
- Remove extra spaces and noisy text
- Save cleaned text output for inspection
- Handle document metadata such as file name and section source

### Concepts to learn
- Unstructured text ingestion
- Why preprocessing matters
- Common parsing issues in PDFs
- Metadata preservation

### Deliverable
- Clean extracted text from all sample files

---

## Week 3 — Chunking and metadata design
**Goal:** Split documents into useful chunks for retrieval.

### Tasks
- Build chunking logic
- Add chunk overlap
- Add chunk IDs
- Preserve file-level metadata
- Preserve chunk source references
- Compare chunk sizes manually
- Save chunks into a structured format

### Concepts to learn
- Chunk size tradeoffs
- Overlap and context continuity
- Precision vs context coverage
- Why chunk boundaries matter

### Deliverable
- Chunked document dataset with metadata

---

## Week 4 — Embeddings and vector indexing
**Goal:** Convert chunks into vectors and create a searchable vector index.

### Tasks
- Select an embedding model
- Generate embeddings for all chunks
- Store embeddings in FAISS
- Store chunk metadata alongside the index
- Run basic manual similarity searches
- Inspect nearest chunk results

### Concepts to learn
- What embeddings are
- Semantic similarity
- Vector search basics
- Cosine similarity / nearest neighbor intuition

### Deliverable
- Searchable vector index over project documents

---

## Week 5 — Retrieval pipeline
**Goal:** Build the retrieval layer for user questions.

### Tasks
- Convert user query into embedding
- Search vector store
- Return top-k chunks
- Format retrieval results clearly
- Add chunk scores
- Add source references
- Validate whether retrieved chunks are actually relevant

### Concepts to learn
- Query embeddings
- Top-k retrieval
- Retrieval quality
- Similarity scores
- Common retrieval failure modes

### Deliverable
- Working retriever that fetches relevant chunks for a question

---

## Week 6 — Grounded answer generation
**Goal:** Build the first complete RAG system.

### Tasks
- Create prompt template
- Pass retrieved chunks into prompt
- Generate answer from context
- Add citations
- Display question, answer, and sources in UI
- Prevent answers when retrieval quality is weak
- Add fallback message when context is insufficient

### Concepts to learn
- Grounding
- Prompt context assembly
- Hallucination control
- Answer formatting
- Citation strategy

### Deliverable
- Working document Q&A app with citations

---

## Week 7 — Evaluation and tuning
**Goal:** Improve reliability and understand where the system fails.

### Tasks
- Create evaluation question set
- Create expected-answer notes manually
- Test retrieval quality
- Test answer quality
- Compare chunking approaches
- Compare embedding models if needed
- Tune top-k
- Tune prompt design
- Analyze bad examples

### Concepts to learn
- Evaluation mindset
- Retrieval precision vs recall
- Prompt tuning
- Error analysis
- Why systems fail even when code works

### Deliverable
- Measured improvement in retrieval and answer quality

---

## Week 8 — Smarter workflow and LangGraph prep
**Goal:** Move from a linear pipeline toward a workflow-based system.

### Tasks
- Add query classification
- Route different question types differently
- Add source filtering
- Add fallback logic
- Define state object
- Identify future graph nodes
- Build a basic LangGraph workflow skeleton

### Concepts to learn
- Chains vs workflows
- Routing logic
- Stateful orchestration
- Graph-based AI systems
- Early agentic design patterns

### Deliverable
- Smarter workflow and LangGraph-ready design