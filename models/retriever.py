from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from sentence_transformers import SentenceTransformer

from models.chroma_store import DEFAULT_CHROMA_COLLECTION
from models.retrieval_encoder import RetrievalEncoder
from utils.text_normalization import normalize_narrative_text, normalize_retrieved_text


@dataclass(frozen=True)
class RerankConfig:
    scope_title_phrases: tuple[str, ...] = (
        "project scope",
        "out of scope",
        "in scope",
        "scope",
    )
    boilerplate_phrases: tuple[str, ...] = (
        "individual project agreement",
        "master services agreement",
        "customer project contact",
        "vendor project contact",
        "effective upon acceptance",
        "terms and conditions",
        "capitalized terms",
        "exit clause",
        "termination",
    )
    admin_query_terms: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "agreement",
                "contract",
                "customer",
                "vendor",
                "contact",
                "effective",
                "expiration",
                "termination",
                "msa",
                "ipa",
            }
        )
    )
    scope_title_boost: float = 0.34
    title_overlap_weight: float = 0.22
    file_overlap_weight: float = 0.06
    text_overlap_weight: float = 0.04
    phrase_overlap_weight: float = 0.10
    exact_query_boost: float = 0.12
    heading_body_boost: float = 0.07
    adjacent_section_boost: float = 0.04
    tiny_chunk_penalty: float = 0.22
    boilerplate_penalty: float = 0.55
    broad_boilerplate_penalty: float = 0.20
    project_scope_title_boost: float = 0.46
    in_scope_title_boost: float = 0.38
    out_scope_title_boost: float = 0.34
    generic_scope_title_boost: float = 0.24


class LocalRetriever:
    """Retrieve the most relevant chunks from a saved embeddings payload."""

    def __init__(self, encoder: RetrievalEncoder | None = None, rerank_config: RerankConfig | None = None) -> None:
        self.encoder = encoder
        self.rerank_config = rerank_config or RerankConfig()

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
        return [result["text"] for result in self.retrieve_debug(query, embeddings_payload, top_k=top_k)]

    def retrieve_debug(
        self,
        query: str,
        embeddings_payload: dict[str, Any],
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        ranked_results = self._rank_records(query, embeddings_payload, top_k=top_k)
        return [self._format_debug_result(result) for result in ranked_results]

    def retrieve_from_file(
        self,
        query: str,
        input_path: str | Path,
        *,
        top_k: int = 5,
    ) -> list[str]:
        embeddings_payload = self.load_embeddings_output(input_path)
        return self.retrieve(query, embeddings_payload, top_k=top_k)

    def retrieve_debug_from_file(
        self,
        query: str,
        input_path: str | Path,
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        embeddings_payload = self.load_embeddings_output(input_path)
        return self.retrieve_debug(query, embeddings_payload, top_k=top_k)

    def retrieve_from_chroma(
        self,
        query: str,
        collection: Any,
        *,
        model_name: str,
        top_k: int = 5,
        candidate_k: int = 25,
    ) -> list[str]:
        return [
            result["text"]
            for result in self.retrieve_debug_from_chroma(
                query,
                collection,
                model_name=model_name,
                top_k=top_k,
                candidate_k=candidate_k,
            )
        ]

    def retrieve_debug_from_chroma(
        self,
        query: str,
        collection: Any,
        *,
        model_name: str,
        top_k: int = 5,
        candidate_k: int = 25,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k.")

        query_embedding = self._get_encoder({"model": {"name": model_name}}).encode_query_text(query)
        chroma_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )
        records = self._records_from_chroma_result(chroma_result)
        ranked_results = self._rerank_records(query=query, scored_records=records, top_k=top_k)
        return [self._format_debug_result(result) for result in ranked_results]

    def retrieve_associated_debug_from_chroma(
        self,
        query: str,
        collection: Any,
        *,
        model_name: str,
        seed_k: int = 8,
        candidate_k: int = 80,
        associated_window: int = 2,
        max_context_chunks: int = 24,
    ) -> list[dict[str, Any]]:
        if seed_k <= 0:
            raise ValueError("seed_k must be a positive integer.")
        if candidate_k < seed_k:
            raise ValueError("candidate_k must be greater than or equal to seed_k.")
        if associated_window < 0:
            raise ValueError("associated_window cannot be negative.")
        if max_context_chunks <= 0:
            raise ValueError("max_context_chunks must be a positive integer.")

        seed_results = self.retrieve_debug_from_chroma(
            query,
            collection,
            model_name=model_name,
            top_k=seed_k,
            candidate_k=candidate_k,
        )

        ordered: dict[str, dict[str, Any]] = {}
        group_order: dict[tuple[str, str], int] = {}
        for group_index, seed in enumerate(seed_results):
            source_file = self._result_source_file(seed)
            section_title = self._result_section_title(seed)
            group_key = (source_file, section_title)
            group_order.setdefault(group_key, group_index)

            seed = dict(seed)
            seed["relationship"] = "semantic_seed"
            seed["group_rank"] = group_order[group_key]
            self._add_unique_result(ordered, seed)

            for associated in self._associated_results_for_seed(
                collection=collection,
                seed=seed,
                associated_window=associated_window,
                group_rank=group_order[group_key],
            ):
                self._add_unique_result(ordered, associated)

        results = list(ordered.values())
        results.sort(
            key=lambda item: (
                int(item.get("group_rank", max_context_chunks)),
                str(item.get("file_name") or ""),
                self._sort_order(item.get("metadata", {}).get("order_start")),
                0 if item.get("relationship") == "semantic_seed" else 1,
                str(item.get("chunk_id") or ""),
            )
        )
        return results[:max_context_chunks]

    def _rank_records(
        self,
        query: str,
        embeddings_payload: dict[str, Any],
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_embedding = self._get_encoder(embeddings_payload).encode_query_text(query)
        normalized_embeddings = bool(embeddings_payload.get("model", {}).get("normalized_embeddings"))
        records = [record for record in embeddings_payload["records"] if isinstance(record.get("embedding"), list)]

        scored_results: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            semantic_score = self._similarity_score(
                query_embedding=query_embedding,
                chunk_embedding=record["embedding"],
                normalized_embeddings=normalized_embeddings,
            )
            rerank = self._rerank_record(query=query, record=record, raw_vector_score=semantic_score, records=records, index=index)
            scored_results.append(rerank)

        scored_results.sort(key=lambda item: item["final_score"], reverse=True)
        return scored_results[:top_k]

    def _rerank_records(
        self,
        *,
        query: str,
        scored_records: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        reranked_results = [
            self._rerank_record(
                query=query,
                record=record,
                raw_vector_score=float(record.get("_raw_vector_score", 0.0)),
                records=scored_records,
                index=index,
            )
            for index, record in enumerate(scored_records)
        ]
        reranked_results.sort(key=lambda item: item["final_score"], reverse=True)
        return reranked_results[:top_k]

    @classmethod
    def _records_from_chroma_result(cls, chroma_result: Mapping[str, Any]) -> list[dict[str, Any]]:
        ids = cls._first_query_result(chroma_result.get("ids"))
        documents = cls._first_query_result(chroma_result.get("documents"))
        metadatas = cls._first_query_result(chroma_result.get("metadatas"))
        distances = cls._first_query_result(chroma_result.get("distances"))

        records: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            document = str(documents[index] or "") if index < len(documents) else ""
            distance = float(distances[index]) if index < len(distances) and distances[index] is not None else 1.0

            record = cls._record_from_chroma_metadata(metadata, document=document, chunk_id=str(chunk_id))
            record["_raw_vector_score"] = cls._score_from_chroma_distance(distance)
            record["_chroma_distance"] = distance
            records.append(record)

        return records

    def _associated_results_for_seed(
        self,
        *,
        collection: Any,
        seed: dict[str, Any],
        associated_window: int,
        group_rank: int,
    ) -> list[dict[str, Any]]:
        metadata = dict(seed.get("metadata") or {})
        source_file = self._result_source_file(seed)
        section_title = self._result_section_title(seed)
        seed_order = self._safe_int(metadata.get("order_start"))
        if not source_file:
            return []

        try:
            chroma_result = collection.get(
                where={"source_file": source_file},
                include=["documents", "metadatas"],
            )
        except Exception:
            return []

        records = self._records_from_chroma_result(chroma_result)
        associated: list[dict[str, Any]] = []
        for record in records:
            record_section = self._record_section_title(record)
            record_metadata = dict(record.get("metadata") or {}).get("chroma_metadata", {})
            record_order = self._safe_int(record_metadata.get("order_start"))

            same_section = bool(section_title and record_section.lower() == section_title.lower())
            near_seed = (
                seed_order is not None
                and record_order is not None
                and abs(record_order - seed_order) <= associated_window
            )
            if not same_section and not near_seed:
                continue

            result = self._format_associated_result(
                record=record,
                relationship="same_section" if same_section else "nearby_context",
                group_rank=group_rank,
            )
            associated.append(result)

        return associated

    @classmethod
    def _format_associated_result(
        cls,
        *,
        record: dict[str, Any],
        relationship: str,
        group_rank: int,
    ) -> dict[str, Any]:
        section_title = cls._record_section_title(record)
        text = normalize_retrieved_text(str(record.get("text") or ""), section_title)
        metadata = dict(record.get("metadata") or {})
        chroma_metadata = metadata.get("chroma_metadata", {})
        return {
            "score": round(float(record.get("_raw_vector_score", 0.0)), 6),
            "final_score": round(float(record.get("_raw_vector_score", 0.0)), 6),
            "raw_vector_score": round(float(record.get("_raw_vector_score", 0.0)), 6),
            "rerank_delta": 0.0,
            "reasons": [f"associated:{relationship}"],
            "relationship": relationship,
            "group_rank": group_rank,
            "chroma_distance": record.get("_chroma_distance"),
            "chunk_id": record.get("chunk_id"),
            "file_name": record.get("file_name") or record.get("source_file"),
            "section_title": section_title,
            "page_number": record.get("page_number"),
            "chunk_length": len(text),
            "preview": cls._preview(text),
            "metadata": chroma_metadata,
            "text": text,
        }

    @staticmethod
    def _add_unique_result(results: dict[str, dict[str, Any]], result: dict[str, Any]) -> None:
        chunk_id = str(result.get("chunk_id") or "").strip()
        if not chunk_id:
            chunk_id = f"{result.get('file_name')}::{result.get('section_title')}::{len(results)}"
        if chunk_id not in results:
            results[chunk_id] = result
        elif result.get("relationship") == "semantic_seed":
            results[chunk_id] = result

    @staticmethod
    def _result_source_file(result: dict[str, Any]) -> str:
        metadata = dict(result.get("metadata") or {})
        return str(metadata.get("source_file") or result.get("file_name") or "").strip()

    @staticmethod
    def _result_section_title(result: dict[str, Any]) -> str:
        return normalize_narrative_text(str(result.get("section_title") or "")).rstrip(":")

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _sort_order(cls, value: Any) -> int:
        order = cls._safe_int(value)
        return order if order is not None else 1_000_000_000

    @staticmethod
    def _first_query_result(value: Any) -> list[Any]:
        if not value:
            return []
        if isinstance(value, list) and value and isinstance(value[0], list):
            return value[0]
        if isinstance(value, list):
            return value
        return []

    @classmethod
    def _record_from_chroma_metadata(cls, metadata: dict[str, Any], *, document: str, chunk_id: str) -> dict[str, Any]:
        text = cls._content_from_document(document)
        record = {
            "chunk_id": metadata.get("chunk_id") or chunk_id,
            "file_name": metadata.get("file_name") or metadata.get("source_file"),
            "source_file": metadata.get("source_file") or metadata.get("file_name"),
            "source_path": metadata.get("source_path"),
            "doc_type": metadata.get("doc_type"),
            "chunk_type": metadata.get("chunk_type"),
            "section_title": metadata.get("section_title"),
            "section_key": metadata.get("section_key"),
            "heading": metadata.get("heading"),
            "page_number": metadata.get("page_number"),
            "order_start": metadata.get("order_start"),
            "order_end": metadata.get("order_end"),
            "char_count": metadata.get("char_count") or len(text),
            "text": text,
            "retrieval_text": document,
            "metadata": {
                "source_block_count": cls._count_metadata_items(metadata.get("block_ids")),
                "embedding_model": metadata.get("embedding_model"),
                "normalized_embeddings": metadata.get("normalized_embeddings"),
                "chroma_metadata": metadata,
            },
            "block_types": cls._json_list(metadata.get("block_types")),
        }
        return record

    @staticmethod
    def _content_from_document(document: str) -> str:
        marker = "Content:"
        if marker in document:
            return document.split(marker, 1)[1].strip()
        return document.strip()

    @staticmethod
    def _score_from_chroma_distance(distance: float) -> float:
        # The collection is created with cosine space, where Chroma returns distance.
        return 1.0 - distance

    @staticmethod
    def _json_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if not isinstance(value, str) or not value.strip():
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []

    @classmethod
    def _count_metadata_items(cls, value: Any) -> int:
        parsed = cls._json_list(value)
        return len(parsed) if parsed else 1

    @staticmethod
    def infer_chroma_model_name(collection: Any, default_model_name: str) -> str:
        try:
            result = collection.get(limit=1, include=["metadatas"])
        except Exception:
            return default_model_name

        metadatas = result.get("metadatas") if isinstance(result, dict) else None
        if isinstance(metadatas, list) and metadatas:
            model_name = str((metadatas[0] or {}).get("embedding_model") or "").strip()
            if model_name:
                return model_name

        return default_model_name

    @staticmethod
    def open_chroma_collection(
        *,
        persist_path: str | Path,
        collection_name: str = DEFAULT_CHROMA_COLLECTION,
    ) -> Any:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("chromadb is required to query the vector database.") from exc

        client = chromadb.PersistentClient(path=str(persist_path))
        return client.get_collection(name=collection_name)

    def _format_debug_result(self, reranked_result: dict[str, Any]) -> dict[str, Any]:
        record = reranked_result["record"]
        section_title = self._record_section_title(record)
        text = normalize_retrieved_text(str(record.get("text") or ""), section_title)
        metadata = dict(record.get("metadata") or {})
        return {
            "score": round(reranked_result["final_score"], 6),
            "final_score": round(reranked_result["final_score"], 6),
            "raw_vector_score": round(reranked_result["raw_vector_score"], 6),
            "rerank_delta": round(reranked_result["rerank_delta"], 6),
            "reasons": reranked_result["reasons"],
            "chroma_distance": record.get("_chroma_distance"),
            "chunk_id": record.get("chunk_id"),
            "file_name": record.get("file_name") or record.get("source_file"),
            "section_title": section_title,
            "page_number": record.get("page_number"),
            "chunk_length": len(text),
            "preview": self._preview(text),
            "metadata": metadata.get("chroma_metadata", {}),
            "text": text,
        }

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

    def _rerank_record(
        self,
        *,
        query: str,
        record: dict[str, Any],
        raw_vector_score: float,
        records: list[dict[str, Any]],
        index: int,
    ) -> dict[str, Any]:
        rerank_delta, reasons = self._metadata_rerank_delta(query=query, record=record, records=records, index=index)
        return {
            "final_score": raw_vector_score + rerank_delta,
            "raw_vector_score": raw_vector_score,
            "rerank_delta": rerank_delta,
            "reasons": reasons,
            "record": record,
        }

    def _metadata_rerank_delta(
        self,
        *,
        query: str,
        record: dict[str, Any],
        records: list[dict[str, Any]],
        index: int,
    ) -> tuple[float, list[str]]:
        config = self.rerank_config
        query_text = normalize_narrative_text(query).lower()
        query_tokens = self._tokens(query_text)
        if not query_tokens:
            return 0.0, []

        reasons: list[str] = []
        section_title = self._record_section_title(record)
        text = normalize_retrieved_text(str(record.get("text") or ""), section_title).lower()
        file_name = normalize_narrative_text(str(record.get("file_name") or record.get("source_file") or "")).lower()
        combined_text = " ".join(part for part in [section_title.lower(), file_name, text] if part)

        title_overlap = self._token_overlap(query_tokens, self._tokens(section_title.lower()))
        file_overlap = self._token_overlap(query_tokens, self._tokens(file_name))
        text_overlap = self._token_overlap(query_tokens, self._tokens(text))
        phrase_overlap = self._phrase_overlap(query_tokens, combined_text)
        delta = 0.0

        scope_boost = self._scope_section_boost(query_tokens, section_title)
        if scope_boost:
            delta += scope_boost
            reasons.append(f"boost:scope_section(+{scope_boost:.2f})")

        title_boost = config.title_overlap_weight * title_overlap
        if title_boost > 0:
            delta += title_boost
            reasons.append(f"boost:title_overlap(+{title_boost:.2f})")

        file_boost = config.file_overlap_weight * file_overlap
        if file_boost > 0:
            delta += file_boost
            reasons.append(f"boost:file_overlap(+{file_boost:.2f})")

        text_boost = config.text_overlap_weight * text_overlap
        if text_boost > 0:
            delta += text_boost
            reasons.append(f"boost:text_overlap(+{text_boost:.2f})")

        phrase_boost = config.phrase_overlap_weight * phrase_overlap
        if phrase_boost > 0:
            delta += phrase_boost
            reasons.append(f"boost:phrase_overlap(+{phrase_boost:.2f})")

        if query_text and query_text in combined_text:
            delta += config.exact_query_boost
            reasons.append(f"boost:exact_query(+{config.exact_query_boost:.2f})")

        text_len = len(text)
        has_heading_and_body = bool(section_title and text_len > max(120, len(section_title) + 40))
        if has_heading_and_body:
            delta += config.heading_body_boost
            reasons.append(f"boost:heading_body(+{config.heading_body_boost:.2f})")

        if self._has_adjacent_section_match(records, index, section_title) and self._query_matches_section(query_tokens, section_title):
            delta += config.adjacent_section_boost
            reasons.append(f"boost:adjacent_section(+{config.adjacent_section_boost:.2f})")

        if text_len < 80 and int(dict(record.get("metadata") or {}).get("source_block_count", 1)) <= 1:
            delta -= config.tiny_chunk_penalty
            reasons.append(f"penalty:tiny_chunk(-{config.tiny_chunk_penalty:.2f})")

        if self._is_boilerplate_content(combined_text) and not self._is_admin_query(query_tokens):
            delta -= config.boilerplate_penalty
            reasons.append(f"penalty:boilerplate(-{config.boilerplate_penalty:.2f})")

        if self._is_broad_boilerplate(record=record, text=text, section_title=section_title) and not self._is_admin_query(query_tokens):
            delta -= config.broad_boilerplate_penalty
            reasons.append(f"penalty:broad_boilerplate(-{config.broad_boilerplate_penalty:.2f})")

        return delta, reasons

    def _is_scope_section(self, section_title: str) -> bool:
        normalized_title = section_title.lower()
        return any(phrase in normalized_title for phrase in self.rerank_config.scope_title_phrases)

    def _scope_section_boost(self, query_tokens: list[str], section_title: str) -> float:
        if not self._is_scope_query(query_tokens) or not self._is_scope_section(section_title):
            return 0.0

        config = self.rerank_config
        normalized_title = section_title.lower()
        query_set = set(query_tokens)
        if "project scope" in normalized_title:
            return config.project_scope_title_boost
        if "out of scope" in normalized_title:
            return config.out_scope_title_boost if "out" in query_set else config.generic_scope_title_boost
        if "in scope" in normalized_title:
            return config.in_scope_title_boost
        return config.generic_scope_title_boost

    @staticmethod
    def _is_scope_query(query_tokens: list[str]) -> bool:
        query_set = set(query_tokens)
        return bool(query_set.intersection({"scope", "scoped", "inscope"}))

    def _query_matches_section(self, query_tokens: list[str], section_title: str) -> bool:
        return self._token_overlap(query_tokens, self._tokens(section_title)) >= 0.25

    def _has_adjacent_section_match(self, records: list[dict[str, Any]], index: int, section_title: str) -> bool:
        if not section_title:
            return False
        for adjacent_index in (index - 1, index + 1):
            if 0 <= adjacent_index < len(records) and self._record_section_title(records[adjacent_index]).lower() == section_title.lower():
                return True
        return False

    def _is_boilerplate_content(self, combined_text: str) -> bool:
        return any(phrase in combined_text for phrase in self.rerank_config.boilerplate_phrases)

    def _is_admin_query(self, query_tokens: list[str]) -> bool:
        return bool(set(query_tokens).intersection(self.rerank_config.admin_query_terms))

    def _is_broad_boilerplate(self, *, record: dict[str, Any], text: str, section_title: str) -> bool:
        title = section_title.lower()
        block_types = {str(block_type).lower() for block_type in record.get("block_types", [])}
        boilerplate_title = title in {"individual project agreement", "master services agreement"} or title.startswith(
            ("title of project", "customer", "vendor", "effective date", "expiration date")
        )
        broad_text = len(text) > 500 and len(self._tokens(text)) > 90
        sparse_section = not self._is_scope_section(section_title) and not block_types.intersection({"table", "row"})
        return boilerplate_title or (broad_text and sparse_section and self._is_boilerplate_content(text))

    @staticmethod
    def _record_section_title(record: dict[str, Any]) -> str:
        return normalize_narrative_text(
            str(record.get("section_title") or record.get("heading") or record.get("section_key") or "")
        ).rstrip(":")

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _token_overlap(query_tokens: list[str], candidate_tokens: list[str]) -> float:
        if not query_tokens or not candidate_tokens:
            return 0.0
        query_set = set(query_tokens)
        return len(query_set.intersection(candidate_tokens)) / len(query_set)

    @staticmethod
    def _phrase_overlap(query_tokens: list[str], candidate_text: str) -> float:
        if len(query_tokens) < 2:
            return 0.0

        phrases: list[str] = []
        for size in (3, 2):
            phrases.extend(" ".join(query_tokens[index:index + size]) for index in range(len(query_tokens) - size + 1))

        if not phrases:
            return 0.0

        normalized_candidate = " ".join(LocalRetriever._tokens(candidate_text))
        matches = sum(1 for phrase in phrases if phrase in normalized_candidate)
        return matches / len(phrases)

    @staticmethod
    def _preview(text: str, max_chars: int = 260) -> str:
        text = normalize_narrative_text(text)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

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
