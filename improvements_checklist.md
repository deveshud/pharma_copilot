# PharmaRAG Future Improvements Checklist

## Retrieval improvements
- [ ] Compare small vs large chunk sizes
- [ ] Compare chunk overlap values
- [ ] Add metadata-based filtering
- [ ] Add document-type filtering
- [ ] Add source-level retrieval controls
- [ ] Add hybrid retrieval later
- [ ] Add reranking later

---

## Embedding improvements
- [ ] Compare two or more embedding models
- [ ] Measure retrieval quality across models
- [ ] Try a lightweight local model first
- [ ] Evaluate domain-specific embedding performance

---

## Generation improvements
- [ ] Improve prompt structure
- [ ] Force citation-first answers
- [ ] Add “insufficient context” response behavior
- [ ] Add concise vs detailed answer modes
- [ ] Add answer confidence notes

---

## Evaluation improvements
- [ ] Create benchmark question set
- [ ] Add expected source references
- [ ] Track retrieval hit quality
- [ ] Track grounded vs ungrounded answers
- [ ] Maintain a log of failure cases
- [ ] Re-test after each major change

---

## Product / UI improvements
- [ ] Improve Streamlit layout
- [ ] Add file upload support
- [ ] Add retrieved chunk preview
- [ ] Add score display
- [ ] Add source document panel
- [ ] Add chat history
- [ ] Add export of answer + citations

---

## Workflow improvements
- [ ] Add query classification
- [ ] Add fallback logic
- [ ] Add source routing
- [ ] Add document-type routing
- [ ] Add multi-step retrieval
- [ ] Add LangGraph workflow
- [ ] Add retry and control nodes

---

## Engineering improvements
- [ ] Add structured logging
- [ ] Add config file support
- [ ] Add unit tests
- [ ] Add test sample documents
- [ ] Add reusable utility modules
- [ ] Add pre-commit hooks
- [ ] Add GitHub Actions later
- [ ] Add proper error handling

---

## Advanced RAG improvements
- [ ] Add contextual chunking
- [ ] Add section-aware chunking
- [ ] Add parent-child retrieval
- [ ] Add query rewriting
- [ ] Add answer synthesis across multiple chunks
- [ ] Add multi-document comparison
- [ ] Add grounded summarization

---

## Agentic / advanced AI improvements
- [ ] Add LangGraph orchestration
- [ ] Add tool abstraction
- [ ] Add query planner
- [ ] Add retrieval decision node
- [ ] Add evaluation node
- [ ] Add self-check step before answering
- [ ] Add human-in-the-loop checkpoints

---

## Long-term enterprise improvements
- [ ] Replace local vector DB with scalable vector platform
- [ ] Add authentication
- [ ] Add document versioning
- [ ] Add enterprise metadata catalog
- [ ] Add Snowflake integration later
- [ ] Compare local RAG with managed retrieval systems
- [ ] Add structured data Q&A layer