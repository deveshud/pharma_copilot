from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.retrieval_encoder import RetrievalEncoder


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.last_encoded_texts: list[str] = []

    def encode_document(
        self,
        texts: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        self.last_encoded_texts = texts
        return [[0.1, 0.2, 0.3] for _ in texts]

    def encode_query(
        self,
        query: str,
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> list[float]:
        return [0.7, 0.2, 0.1]

    def get_sentence_embedding_dimension(self) -> int:
        return 3


def make_chunk(
    *,
    chunk_id: str,
    source_file: str = "sample.docx",
    doc_type: str = "docx",
    chunk_type: str = "section",
    section_path: list[str] | None = None,
    heading: str | None = None,
    text: str = "Base content",
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_file": source_file,
        "source_path": f"C:/docs/{source_file}",
        "doc_type": doc_type,
        "chunk_type": chunk_type,
        "section_path": section_path or [],
        "section_key": "",
        "heading": heading,
        "text": text,
        "char_count": len(text),
        "block_ids": [f"{chunk_id}_block"],
        "block_types": ["paragraph"],
        "order_start": 1,
        "order_end": 1,
        "page_numbers": [],
        "slide_numbers": [],
        "shape_indices": [],
        "sheet_names": [],
        "row_numbers": [],
        "metadata": {"max_chars": 1200, "source_block_count": 1},
    }


def test_build_retrieval_text_includes_structural_context() -> None:
    encoder = RetrievalEncoder(model_name="fake-model", model=FakeSentenceTransformer())
    chunk = make_chunk(
        chunk_id="chunk_1",
        section_path=["Executive Summary", "Scope"],
        heading="Scope",
        text="This chunk explains the deployment scope.",
    )

    retrieval_text = encoder.build_retrieval_text(chunk)

    assert "Source File: sample.docx" in retrieval_text
    assert "Chunk Type: section" in retrieval_text
    assert "Section Path: Executive Summary > Scope" in retrieval_text
    assert "Heading: Scope" in retrieval_text
    assert retrieval_text.endswith("This chunk explains the deployment scope.")


def test_build_retrieval_text_skips_heading_when_text_already_contains_it() -> None:
    encoder = RetrievalEncoder(model_name="fake-model", model=FakeSentenceTransformer())
    chunk = make_chunk(
        chunk_id="chunk_2",
        section_path=["Executive Summary"],
        heading="Executive Summary",
        text="Executive Summary\nThis chunk explains the deployment scope.",
    )

    retrieval_text = encoder.build_retrieval_text(chunk)

    assert "Section Path: Executive Summary" in retrieval_text
    assert "Heading: Executive Summary" not in retrieval_text


def test_encode_chunked_output_adds_embeddings_and_stats() -> None:
    fake_model = FakeSentenceTransformer()
    encoder = RetrievalEncoder(model_name="fake-model", model=fake_model)
    chunked_output = {
        "sample.docx": [
            make_chunk(chunk_id="chunk_1", text="Alpha content"),
            make_chunk(chunk_id="chunk_2", text="Beta content"),
        ]
    }

    encoded = encoder.encode_chunked_output(chunked_output, source_path="outputs/structural_chunks.json")

    assert encoded["model"]["name"] == "fake-model"
    assert encoded["model"]["embedding_dimension"] == 3
    assert encoded["stats"]["source_document_count"] == 1
    assert encoded["stats"]["chunk_count"] == 2
    assert len(encoded["records"]) == 2
    assert encoded["records"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert fake_model.last_encoded_texts[0].endswith("Alpha content")


def test_save_embeddings_output_writes_json_payload(tmp_path: Path) -> None:
    encoder = RetrievalEncoder(model_name="fake-model", model=FakeSentenceTransformer())
    output_path = tmp_path / "retrieval_embeddings.json"
    payload = {
        "output_format": "retrieval_embeddings/v1",
        "created_at": "2026-04-06T00:00:00+00:00",
        "source_path": "outputs/structural_chunks.json",
        "model": {
            "name": "fake-model",
            "embedding_dimension": 3,
            "normalized_embeddings": True,
        },
        "stats": {
            "source_document_count": 1,
            "chunk_count": 1,
        },
        "records": [
            {
                "chunk_id": "chunk_1",
                "retrieval_text": "Example text",
                "embedding": [0.1, 0.2, 0.3],
            }
        ],
    }

    saved_path = encoder.save_embeddings_output(payload, output_path)
    saved_payload = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path == output_path
    assert saved_payload["records"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_encode_query_text_uses_query_encoder_when_available() -> None:
    encoder = RetrievalEncoder(model_name="fake-model", model=FakeSentenceTransformer())

    query_embedding = encoder.encode_query_text("What metrics are available?")

    assert query_embedding == [0.7, 0.2, 0.1]
