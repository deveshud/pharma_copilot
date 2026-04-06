from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from models.retrieval_encoder import RetrievalEncoder


class LocalRetriever:
    """Retrieve the most relevant chunks from a saved embeddings payload."""

    def __init__(self, encoder: RetrievalEncoder | None = None) -> None:
        self.encoder = encoder

    def load_embeddings_output(self, input_path: str | Path) -> dict[str, Any]:
        path = Path(input_path)
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError("Embeddings output must be a JSON object.")
        if not isinstance(payload.get("records"), list):
            raise ValueError("Embeddings output must contain a 'records' list.")

        return payload

    def retrieve(
        self,
        query: str,
        embeddings_payload: dict[str, Any],
        *,
        top_k: int = 5,
    ) -> list[str]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_embedding = self._get_encoder(embeddings_payload).encode_query_text(query)
        normalized_embeddings = bool(embeddings_payload.get("model", {}).get("normalized_embeddings"))

        scored_results: list[tuple[float, dict[str, Any]]] = []
        for record in embeddings_payload["records"]:
            embedding = record.get("embedding")
            if not isinstance(embedding, list):
                continue

            score = self._similarity_score(
                query_embedding=query_embedding,
                chunk_embedding=embedding,
                normalized_embeddings=normalized_embeddings,
            )
            scored_results.append((score, record))

        scored_results.sort(key=lambda item: item[0], reverse=True)

        return [
            str(record.get("text") or "")
            for _, record in scored_results[:top_k]
        ]

    def retrieve_from_file(
        self,
        query: str,
        input_path: str | Path,
        *,
        top_k: int = 5,
    ) -> list[str]:
        embeddings_payload = self.load_embeddings_output(input_path)
        return self.retrieve(query, embeddings_payload, top_k=top_k)

    def _get_encoder(self, embeddings_payload: dict[str, Any]) -> RetrievalEncoder:
        if self.encoder is not None:
            return self.encoder

        model_name = str(embeddings_payload.get("model", {}).get("name") or "").strip()
        if not model_name:
            raise ValueError("Embeddings payload is missing the model name required for query encoding.")

        local_model_path = self._resolve_local_model_path(model_name)
        if local_model_path is not None:
            model = SentenceTransformer(str(local_model_path), local_files_only=True)
            self.encoder = RetrievalEncoder(model_name=model_name, model=model)
            return self.encoder

        self.encoder = RetrievalEncoder(model_name=model_name)
        return self.encoder

    @staticmethod
    def _resolve_local_model_path(model_name: str) -> Path | None:
        sanitized_name = model_name.replace("/", "--")
        snapshot_root = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--{sanitized_name}"
            / "snapshots"
        )
        if not snapshot_root.exists():
            return None

        snapshots = sorted(
            [path for path in snapshot_root.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not snapshots:
            return None

        return snapshots[0]

    @staticmethod
    def _similarity_score(
        *,
        query_embedding: list[float],
        chunk_embedding: list[float],
        normalized_embeddings: bool,
    ) -> float:
        if len(query_embedding) != len(chunk_embedding):
            raise ValueError("Query and chunk embeddings must have the same dimension.")

        dot_product = sum(query * chunk for query, chunk in zip(query_embedding, chunk_embedding, strict=True))
        if normalized_embeddings:
            return dot_product

        query_norm = math.sqrt(sum(value * value for value in query_embedding))
        chunk_norm = math.sqrt(sum(value * value for value in chunk_embedding))
        if query_norm == 0.0 or chunk_norm == 0.0:
            return 0.0

        return dot_product / (query_norm * chunk_norm)
