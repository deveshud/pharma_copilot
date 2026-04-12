from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.chroma_store import ChromaEmbeddingStore


class FakeCollection:
    def __init__(self) -> None:
        self.upserts: list[dict[str, object]] = []

    def upsert(self, **kwargs: object) -> None:
        self.upserts.append(kwargs)


class FakeClient:
    def __init__(self) -> None:
        self.collection = FakeCollection()
        self.deleted_collections: list[str] = []
        self.requested_collections: list[tuple[str, dict[str, object]]] = []

    def delete_collection(self, name: str) -> None:
        self.deleted_collections.append(name)

    def get_or_create_collection(self, name: str, metadata: dict[str, object]) -> FakeCollection:
        self.requested_collections.append((name, metadata))
        return self.collection


def make_payload() -> dict[str, object]:
    return {
        "source_path": "outputs/structural_chunks.json",
        "model": {
            "name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "normalized_embeddings": True,
        },
        "records": [
            {
                "chunk_id": "chunk_1",
                "source_file": "proposal.docx",
                "file_name": "proposal.docx",
                "doc_type": "docx",
                "chunk_type": "section",
                "section_title": "Project Scope",
                "section_path": ["Project Scope"],
                "text": "Project Scope: Requirements mapping is in scope.",
                "retrieval_text": "Source File: proposal.docx\nContent:\nProject Scope",
                "embedding": [0.1, 0.2, 0.3],
                "block_ids": ["b1", "b2"],
            },
            {
                "chunk_id": "chunk_bad",
                "text": "Missing embedding should be skipped.",
            },
        ],
    }


def test_chroma_store_upserts_generated_embeddings() -> None:
    client = FakeClient()
    store = ChromaEmbeddingStore(
        persist_path="outputs/chroma_db",
        collection_name="test_collection",
        client=client,
    )

    summary = store.store_embeddings_output(make_payload(), reset_collection=True, batch_size=1)

    assert summary["records_seen"] == 2
    assert summary["records_stored"] == 1
    assert client.deleted_collections == ["test_collection"]
    assert client.requested_collections == [("test_collection", {"hnsw:space": "cosine"})]
    assert client.collection.upserts[0]["ids"] == ["chunk_1"]
    assert client.collection.upserts[0]["embeddings"] == [[0.1, 0.2, 0.3]]
    assert client.collection.upserts[0]["documents"] == ["Source File: proposal.docx\nContent:\nProject Scope"]

    metadata = client.collection.upserts[0]["metadatas"][0]  # type: ignore[index]
    assert metadata["chunk_id"] == "chunk_1"
    assert metadata["section_title"] == "Project Scope"
    assert metadata["section_path"] == '["Project Scope"]'
    assert metadata["block_ids"] == '["b1", "b2"]'
    assert metadata["embedding_model"] == "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    assert metadata["normalized_embeddings"] is True
