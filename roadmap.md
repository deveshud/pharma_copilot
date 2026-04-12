# Pharma Copilot Roadmap

This is the canonical roadmap for Pharma Copilot. It consolidates the previous weekly milestones, improvements checklist, and feature implementation roadmap into one source of truth.

## Product Goal

Build a local-first RAG system that can ingest pharma and enterprise documents, retrieve the right evidence from a local vector store, and generate grounded answers through a local LLM.

## Implemented Features

### Foundation

- [x] Python 3.12 project structure.
- [x] `uv` dependency management.
- [x] Local-first architecture with no managed AI platform requirement.
- [x] Project documentation and runnable app entrypoint.
- [x] Unit test suite for core pipeline components.

### Document Ingestion

- [x] PDF parser.
- [x] DOCX parser.
- [x] PPTX parser.
- [x] XLSX parser.
- [x] Consolidated ingestion runner for files under `data/sample_docs`.
- [x] Parser-specific metadata preservation.
- [x] Normalized JSON outputs for parsed document blocks.

### Text Cleaning And Chunking

- [x] Text normalization utilities.
- [x] Structure-aware chunking.
- [x] Section, source, ordering, and block metadata preservation.
- [x] Token overlap within sections.
- [x] Table-preserving chunk behavior.
- [x] Chunk output saved to `outputs/structural_chunks.json`.

### Embeddings And Vector Storage

- [x] Sentence Transformers retrieval encoder.
- [x] Embedding metadata stored with model name, dimension, and normalization status.
- [x] Embeddings saved to `outputs/retrieval_embeddings.json`.
- [x] Persistent ChromaDB vector store.
- [x] Chroma collection defaults to `pharma_copilot_chunks`.
- [x] Query embedding model inferred from Chroma metadata in the UI.

### Retrieval

- [x] ChromaDB semantic search.
- [x] Metadata-aware reranking.
- [x] Boilerplate demotion for non-admin scope queries.
- [x] Scope-aware boosting.
- [x] CLI retrieval debug runner.
- [x] Associated-context expansion for same-section and nearby source chunks.
- [x] Retrieved chunk metadata, scores, and reranking reasons available for inspection.

### Answer Generation

- [x] Grounded prompt builder.
- [x] Retrieved context appears before the user question in the prompt.
- [x] Local Ollama adapter.
- [x] Default Ollama model setting: `llama3.2:3b`.
- [x] Prompt instructs the LLM to synthesize from retrieved evidence without inventing facts.
- [x] Citation-style answer instruction.
- [x] Streaming Ollama answer generation.
- [x] Empty-context fallback response.

### Streamlit UI

- [x] Single `Answer question` workflow.
- [x] ChromaDB path and collection settings in sidebar.
- [x] Ollama model, host, temperature, and context-window settings in sidebar.
- [x] Low-level retrieval controls hidden from the user.
- [x] Automatic retrieval before answer generation.
- [x] Clean answer-first layout without vector candidate, chunk count, or embedding-model boxes.
- [x] Optional retrieved evidence expander.
- [x] Optional grounded prompt expander.
- [x] Streaming answer display.
- [x] Spinner with interim status updates during retrieval and answer generation.
- [x] Streamlit watcher disabled to avoid optional `torchvision` import noise from `transformers`.

## Implemented Milestone Summary

### Milestone 1: Setup And Foundation

- [x] Repository structure.
- [x] Virtual environment and dependency setup.
- [x] Basic Streamlit app.
- [x] README and project notes.

### Milestone 2: Parsing And Cleaning

- [x] PDF, DOCX, PPTX, and XLSX parsing.
- [x] Normalized text outputs.
- [x] Document and block metadata preservation.

### Milestone 3: Chunking And Metadata

- [x] Structure-aware chunks.
- [x] Chunk IDs.
- [x] Source references.
- [x] Overlap and metadata-rich chunk output.

### Milestone 4: Embeddings And Indexing

- [x] Embedding model integration.
- [x] Chunk embeddings.
- [x] ChromaDB vector index.
- [x] Persisted metadata alongside vectors.

### Milestone 5: Retrieval Pipeline

- [x] Query embedding.
- [x] Vector search.
- [x] Reranking.
- [x] Debug output.
- [x] Source references.

### Milestone 6: Grounded Answer Generation

- [x] Prompt template.
- [x] Retrieved context passed into prompt.
- [x] Ollama answer generation.
- [x] Citation instruction.
- [x] Streamlit answer and evidence display.

## Will Be Implemented Later

### Immediate Stabilization

- [ ] Add one command or runner for the full pipeline: ingestion, chunking, embedding, and Chroma storage.
- [ ] Add a Streamlit startup health panel for ChromaDB, collection count, inferred embedding model, Ollama availability, and selected Ollama model.
- [ ] Create an answer-quality triage workflow for incorrect answers.
- [ ] Improve error handling when the selected Ollama model has not been pulled.
- [ ] Improve error handling when ChromaDB exists but the expected collection is missing.
- [ ] Move retrieval constants into a config file.
- [ ] Add tests for Streamlit helper functions that do not require launching the UI.
- [ ] Document sample data setup more explicitly because `data/` is gitignored.

### Answer Quality Recovery

- [ ] Build a wrong-answer log with question, generated answer, expected answer, retrieved chunks, missing evidence, and failure reason.
- [ ] Create a small failure-case benchmark from real incorrect answers before tuning prompts or retrieval settings.
- [ ] Add retrieval hit-rate checks before answer generation so the system can detect when the right evidence was not retrieved.
- [ ] Add an insufficient-context guard that blocks confident answers when retrieved evidence is weak, unrelated, or contradictory.
- [ ] Add citation validation so every generated citation maps back to a retrieved chunk ID.
- [ ] Add groundedness checks that flag claims not supported by retrieved context.
- [ ] Track whether answer failures came from parsing, text normalization, chunking, embeddings, retrieval, reranking, or generation.

### Ingestion

- [ ] Add TXT ingestion.
- [ ] Add Markdown ingestion.
- [ ] Add CSV ingestion.
- [ ] Add HTML ingestion.
- [ ] Add Streamlit file upload.
- [ ] Add UI-triggered indexing for uploaded files.
- [ ] Add duplicate detection by file hash.
- [ ] Add document deletion from ChromaDB.
- [ ] Add document version tracking.
- [ ] Add parser warnings for skipped, protected, or malformed files.
- [ ] Add OCR support for scanned PDFs.
- [ ] Add optional image extraction only behind explicit vision dependencies.

### Chunking

- [ ] Improve text normalization for broken line breaks, split words, repeated whitespace, bullets, headings, and punctuation artifacts.
- [ ] Add normalization tests using real snippets from incorrect-answer cases.
- [ ] Preserve table rows, column labels, section headings, and nearby explanatory text in the same retrieval unit where possible.
- [ ] Avoid tiny heading-only chunks and oversized mixed-topic chunks.
- [ ] Add parent-child chunking so retrieval can match a precise child chunk while passing the broader parent section to the LLM.
- [ ] Add parent-child retrieval.
- [ ] Add configurable section boundary rules.
- [ ] Add a chunk quality report.
- [ ] Report chunk length distribution, duplicate/noisy chunks, section coverage, table coverage, and source-document coverage.
- [ ] Add chunk previews in an index inspection page.
- [ ] Benchmark chunk sizes.
- [ ] Benchmark overlap settings.

### Embeddings

- [ ] Add embedding model comparison reports.
- [ ] Add model-specific retrieval benchmark results.
- [ ] Compare embedding models against the wrong-answer benchmark, not only generic examples.
- [ ] Evaluate domain-friendly query and document prefixes for embedding inputs.
- [ ] Normalize retrieval text consistently before embedding and querying.
- [ ] Add top-k recall measurements for expected source sections.
- [ ] Add local model cache validation before embedding jobs.
- [ ] Add embedding dimension checks before writing to existing Chroma collections.
- [ ] Add batch progress reporting for large embedding runs.
- [ ] Add incremental embedding for new or changed chunks.

### Retrieval

- [ ] Add hybrid keyword plus vector retrieval.
- [ ] Add metadata filters for document type, source file, section, page, sheet, or slide.
- [ ] Add source-specific routing for questions that name a document.
- [ ] Add query rewriting for vague questions.
- [ ] Add multi-query retrieval for broad questions.
- [ ] Add cross-document comparison retrieval.
- [ ] Add retrieval confidence scoring.
- [ ] Add insufficient-context gating before LLM generation.
- [ ] Add deduplication by similar text, not only chunk ID.
- [ ] Add citation grouping by source document.

### Answer Generation

- [ ] Add answer modes: concise, detailed, executive summary, and bullet format.
- [ ] Add source-first answers where citations appear before claims.
- [ ] Add a structured answer schema with answer, evidence, gaps, and citations.
- [ ] Add automatic "not enough evidence" behavior based on retrieval confidence.
- [ ] Validate generated citations against retrieved chunk IDs.
- [ ] Add answer regeneration with adjusted retrieval context.
- [ ] Populate the model selector from local `ollama list`.
- [ ] Add additional local runtimes behind `LocalLLMAdapter`, such as llama.cpp or vLLM.

### Streamlit Product

- [ ] Add chat history for the current session.
- [ ] Add saved conversations.
- [ ] Add answer feedback buttons.
- [ ] Add copy buttons for answer and citations.
- [ ] Add source document panel with filters.
- [ ] Add document upload and indexing status.
- [ ] Add index rebuild controls behind a confirmation step.
- [ ] Add visual health checks for ChromaDB and Ollama.
- [ ] Add stronger empty-state onboarding.
- [ ] Persist user settings.

### Evaluation

- [ ] Create a benchmark question set.
- [ ] Include known wrong-answer questions as regression cases.
- [ ] Add expected answer notes.
- [ ] Add expected source chunk or section labels.
- [ ] Measure retrieval hit rate.
- [ ] Measure whether expected evidence appears in top 3, top 5, and final associated context.
- [ ] Measure citation accuracy.
- [ ] Track answer groundedness.
- [ ] Log failed queries and retrieved context.
- [ ] Add regression tests for known hard questions.
- [ ] Add an evaluation runner that writes `outputs/evaluation_report.json`.
- [ ] Track retrieval and answer quality over time.

### Workflow And Agentic Extensions

- [ ] Add query classification.
- [ ] Route scope, KPI logic, assumption, timeline, contact, and out-of-scope questions differently.
- [ ] Add multi-step retrieval workflows.
- [ ] Add fallback workflow when retrieval confidence is low.
- [ ] Add self-check step before answering.
- [ ] Add LangGraph state object.
- [ ] Add LangGraph workflow skeleton.
- [ ] Add tool abstractions for retrieval, answer generation, evaluation, and export.
- [ ] Add human-in-the-loop checkpoints for uncertain answers.

### Engineering

- [ ] Add structured logging.
- [ ] Add application config file support.
- [ ] Add stronger type checking.
- [ ] Add linting and formatting commands.
- [ ] Add pre-commit hooks.
- [ ] Add CI for tests.
- [ ] Add fixtures for representative sample documents.
- [ ] Add ChromaDB integration tests with a temporary local collection.
- [ ] Add mocked Ollama error-handling tests.
- [ ] Add dependency groups for app, dev, and optional vision/OCR features.

### Enterprise

- [ ] Add authentication.
- [ ] Add role-based access control.
- [ ] Add document-level access permissions.
- [ ] Add audit logging for questions and retrieved sources.
- [ ] Add encrypted storage guidance.
- [ ] Add enterprise metadata catalog.
- [ ] Add Snowflake or warehouse-backed structured data Q&A.
- [ ] Add deployment packaging for a local server or intranet app.
- [ ] Add backup and restore guidance for ChromaDB indexes.

### Documentation

- [x] README describes the current implemented system.
- [x] Consolidated roadmap exists.
- [ ] Add screenshots of the Streamlit UI.
- [ ] Add sample document setup guide.
- [ ] Add troubleshooting guide for Ollama, ChromaDB, and embedding model issues.
- [ ] Add architecture decision records for ChromaDB, Ollama, and Streamlit.
- [ ] Add glossary for RAG, embeddings, chunking, reranking, grounding, and citations.
