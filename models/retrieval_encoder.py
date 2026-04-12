from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer


DEFAULT_RETRIEVAL_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


class RetrievalEncoder:
    """Encode structural chunks into retrieval-oriented embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_RETRIEVAL_MODEL,
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.model = model or SentenceTransformer(model_name)

    def load_chunked_output(self, input_path: str | Path) -> dict[str, list[dict[str, Any]]]:
        path = Path(input_path)
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError("Chunked output must be a JSON object keyed by source file.")

        return payload

    def flatten_chunked_output(
        self,
        chunked_output: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []

        for source_file, chunks in chunked_output.items():
            if not isinstance(chunks, list):
                raise ValueError(f"Chunk list for {source_file!r} must be a JSON array.")

            for chunk in chunks:
                if not isinstance(chunk, dict):
                    raise ValueError(f"Chunk entry for {source_file!r} must be a JSON object.")
                flattened.append(chunk)

        return flattened

    def build_retrieval_text(self, chunk: dict[str, Any]) -> str:
        section_path = chunk.get("section_path") or []
        section_text = " > ".join(str(part) for part in section_path if part)
        explicit_section_title = str(chunk.get("section_title") or "").strip()
        section_title = explicit_section_title or str(chunk.get("heading") or "").strip()
        heading = chunk.get("heading")
        text = str(chunk.get("text") or "").strip()

        lines = [
            f"Source File: {chunk.get('file_name') or chunk.get('source_file', '')}",
            f"Chunk Type: {chunk.get('chunk_type', '')}",
        ]

        if explicit_section_title:
            lines.append(f"Section Title: {section_title}")
        if section_text:
            lines.append(f"Section Path: {section_text}")
        if heading and not self._text_contains_heading(text, str(heading)):
            lines.append(f"Heading: {heading}")

        lines.append("Content:")
        lines.append(text)
        return "\n".join(lines).strip()

    @staticmethod
    def _text_contains_heading(text: str, heading: str) -> bool:
        normalized_heading = " ".join(heading.lower().split())
        if not normalized_heading:
            return False

        first_non_empty_line = next(
            (" ".join(line.lower().split()) for line in text.splitlines() if line.strip()),
            "",
        )
        return first_non_empty_line == normalized_heading

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        show_progress_bar = len(texts) > self.batch_size
        encode_document = getattr(self.model, "encode_document", None)

        if callable(encode_document):
            embeddings = encode_document(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=show_progress_bar,
            )
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=show_progress_bar,
                prompt_name="document",
            )

        return [vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in embeddings]

    def encode_query_text(self, query: str) -> list[float]:
        encode_query = getattr(self.model, "encode_query", None)

        if callable(encode_query):
            embedding = encode_query(
                query,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
        else:
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                prompt_name="query",
            )

        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    def encode_chunked_output(
        self,
        chunked_output: dict[str, list[dict[str, Any]]],
        *,
        source_path: str | Path | None = None,
    ) -> dict[str, Any]:
        flattened_chunks = self.flatten_chunked_output(chunked_output)
        retrieval_texts = [self.build_retrieval_text(chunk) for chunk in flattened_chunks]
        embeddings = self.encode_documents(retrieval_texts)

        records: list[dict[str, Any]] = []
        for chunk, retrieval_text, embedding in zip(flattened_chunks, retrieval_texts, embeddings, strict=True):
            record = dict(chunk)
            record["retrieval_text"] = retrieval_text
            record["embedding"] = embedding
            records.append(record)

        return {
            "output_format": "retrieval_embeddings/v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_path": str(source_path) if source_path else None,
            "model": {
                "name": self.model_name,
                "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                "normalized_embeddings": self.normalize_embeddings,
            },
            "stats": {
                "source_document_count": len(chunked_output),
                "chunk_count": len(records),
            },
            "records": records,
        }

    def encode_file(self, input_path: str | Path) -> dict[str, Any]:
        chunked_output = self.load_chunked_output(input_path)
        return self.encode_chunked_output(chunked_output, source_path=input_path)

    def save_embeddings_output(
        self,
        encoded_output: dict[str, Any],
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(encoded_output, file, indent=2, ensure_ascii=False)

        return path
