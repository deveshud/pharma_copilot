from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_CHROMA_COLLECTION = "pharma_copilot_chunks"


class ChromaEmbeddingStore:
    """Persist generated retrieval embeddings into a ChromaDB collection."""

    def __init__(
        self,
        *,
        persist_path: str | Path,
        collection_name: str = DEFAULT_CHROMA_COLLECTION,
        client: Any | None = None,
    ) -> None:
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self.client = client or self._build_client(self.persist_path)

    @staticmethod
    def load_embeddings_output(input_path: str | Path) -> dict[str, Any]:
        path = Path(input_path)
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError("Embeddings output must be a JSON object.")
        if not isinstance(payload.get("records"), list):
            raise ValueError("Embeddings output must contain a 'records' list.")

        return payload

    def store_embeddings_output(
        self,
        embeddings_payload: Mapping[str, Any],
        *,
        reset_collection: bool = False,
        batch_size: int = 500,
    ) -> dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        records = self._valid_records(embeddings_payload.get("records", []))
        collection = self._collection(reset_collection=reset_collection)

        stored_count = 0
        for batch in self._batched(records, batch_size):
            ids = [self._record_id(record, index=stored_count + offset) for offset, record in enumerate(batch)]
            collection.upsert(
                ids=ids,
                embeddings=[record["embedding"] for record in batch],
                documents=[self._record_document(record) for record in batch],
                metadatas=[
                    self._record_metadata(
                        record,
                        source_path=embeddings_payload.get("source_path"),
                        model=embeddings_payload.get("model", {}),
                    )
                    for record in batch
                ],
            )
            stored_count += len(batch)

        return {
            "persist_path": str(self.persist_path),
            "collection_name": self.collection_name,
            "records_seen": len(embeddings_payload.get("records", [])),
            "records_stored": stored_count,
            "reset_collection": reset_collection,
        }

    def store_file(
        self,
        input_path: str | Path,
        *,
        reset_collection: bool = False,
        batch_size: int = 500,
    ) -> dict[str, Any]:
        payload = self.load_embeddings_output(input_path)
        return self.store_embeddings_output(
            payload,
            reset_collection=reset_collection,
            batch_size=batch_size,
        )

    def _collection(self, *, reset_collection: bool) -> Any:
        if reset_collection:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass

        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _build_client(persist_path: Path) -> Any:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("chromadb is required to store embeddings in the vector database.") from exc

        persist_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(persist_path))

    @staticmethod
    def _valid_records(records: Any) -> list[dict[str, Any]]:
        if not isinstance(records, list):
            raise ValueError("Embeddings output 'records' must be a list.")

        valid_records: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            embedding = record.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                continue
            if not all(isinstance(value, (int, float)) for value in embedding):
                continue
            valid_records.append(record)

        return valid_records

    @staticmethod
    def _record_id(record: Mapping[str, Any], *, index: int) -> str:
        chunk_id = str(record.get("chunk_id") or "").strip()
        if chunk_id:
            return chunk_id

        source_file = str(record.get("source_file") or record.get("file_name") or "unknown")
        return f"{source_file}::record_{index:06d}"

    @staticmethod
    def _record_document(record: Mapping[str, Any]) -> str:
        return str(record.get("retrieval_text") or record.get("text") or "").strip()

    @classmethod
    def _record_metadata(
        cls,
        record: Mapping[str, Any],
        *,
        source_path: Any,
        model: Any,
    ) -> dict[str, str | int | float | bool]:
        metadata = {
            "chunk_id": record.get("chunk_id"),
            "file_name": record.get("file_name") or record.get("source_file"),
            "source_file": record.get("source_file"),
            "source_path": record.get("source_path"),
            "doc_type": record.get("doc_type"),
            "chunk_type": record.get("chunk_type"),
            "section_title": record.get("section_title") or record.get("heading") or record.get("section_key"),
            "section_key": record.get("section_key"),
            "heading": record.get("heading"),
            "page_number": record.get("page_number"),
            "order_start": record.get("order_start"),
            "order_end": record.get("order_end"),
            "char_count": record.get("char_count"),
            "embedding_source_path": source_path,
            "embedding_model": model.get("name") if isinstance(model, Mapping) else None,
            "normalized_embeddings": model.get("normalized_embeddings") if isinstance(model, Mapping) else None,
            "section_path": record.get("section_path"),
            "page_numbers": record.get("page_numbers"),
            "slide_numbers": record.get("slide_numbers"),
            "block_ids": record.get("block_ids"),
            "block_types": record.get("block_types"),
        }

        return {
            key: cls._metadata_scalar(value)
            for key, value in metadata.items()
            if cls._metadata_scalar(value) not in ("", None)
        }

    @staticmethod
    def _metadata_scalar(value: Any) -> str | int | float | bool | None:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _batched(records: Sequence[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
        for start in range(0, len(records), batch_size):
            yield list(records[start:start + batch_size])
