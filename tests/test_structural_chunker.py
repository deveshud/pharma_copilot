from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chunkers.structural_chunker import StructuralChunker, chunk_consolidated_blocks


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True, truncation: bool = False) -> list[str]:
        tokens = text.split()
        if add_special_tokens:
            return ["[CLS]", *tokens, "[SEP]"]
        return tokens


def make_block(
    *,
    block_id: str,
    order: int,
    text: str,
    block_type: str = "paragraph",
    source_file: str = "sample.docx",
    source_path: str = "C:/docs/sample.docx",
    doc_type: str = "docx",
    section_path: list[str] | None = None,
    page_number: int | None = None,
    sheet_name: str | None = None,
    row_number: int | None = None,
) -> dict[str, object]:
    return {
        "block_id": block_id,
        "source_file": source_file,
        "source_path": source_path,
        "doc_type": doc_type,
        "block_type": block_type,
        "order": order,
        "text": text,
        "section_path": section_path or [],
        "page_number": page_number,
        "slide_number": None,
        "shape_index": None,
        "sheet_name": sheet_name,
        "row_number": row_number,
        "metadata": {},
    }


def test_structural_chunker_groups_body_by_section_and_keeps_tables_separate() -> None:
    blocks = [
        make_block(block_id="p2", order=4, text="Overview follow-up paragraph.", section_path=["Overview"]),
        make_block(block_id="h1", order=1, text="Overview", block_type="heading", section_path=["Overview"]),
        make_block(block_id="p1", order=2, text="Overview opening paragraph.", section_path=["Overview"]),
        make_block(block_id="t1", order=3, text="Columns: Name, Value\nRow 1: Revenue, 20", block_type="table", section_path=["Overview"]),
        make_block(block_id="h2", order=5, text="Details", block_type="heading", section_path=["Details"]),
        make_block(block_id="p3", order=6, text="Detailed section body.", section_path=["Details"]),
    ]

    chunks = StructuralChunker(max_chars=120, tokenizer=FakeTokenizer()).chunk_file_blocks(blocks)

    assert [chunk["chunk_type"] for chunk in chunks] == ["section", "table", "section", "section"]
    assert chunks[0]["section_path"] == ["Overview"]
    assert chunks[0]["block_ids"] == ["h1", "p1"]
    assert chunks[1]["block_ids"] == ["t1"]
    assert chunks[2]["section_path"] == ["Overview"]
    assert chunks[2]["heading"] == "Overview"
    assert chunks[2]["block_ids"] == ["h1", "p2"]
    assert chunks[3]["section_path"] == ["Details"]
    assert chunks[3]["block_ids"] == ["h2", "p3"]


def test_structural_chunker_splits_large_sections_without_losing_traceability() -> None:
    long_text = (
        "Sentence one keeps the topic grounded. "
        "Sentence two adds more retrieval context. "
        "Sentence three pushes the section over the configured budget. "
        "Sentence four ensures a second split is required."
    )
    blocks = [
        make_block(block_id="h1", order=1, text="Safety Review", block_type="heading", section_path=["Safety Review"]),
        make_block(block_id="p1", order=2, text=long_text, section_path=["Safety Review"]),
    ]

    chunks = StructuralChunker(max_chars=100, tokenizer=FakeTokenizer()).chunk_file_blocks(blocks)

    assert len(chunks) >= 2
    assert all(chunk["chunk_type"] == "section" for chunk in chunks)
    assert all(chunk["heading"] == "Safety Review" for chunk in chunks)
    assert all("h1" in chunk["block_ids"] for chunk in chunks)
    assert all("p1" in chunk["block_ids"] for chunk in chunks)
    assert all(chunk["char_count"] <= 100 for chunk in chunks)


def test_chunk_consolidated_blocks_keeps_files_separate() -> None:
    consolidated = {
        "file_a.docx": [
            make_block(block_id="a1", order=1, text="Alpha", block_type="heading", source_file="file_a.docx", section_path=["Alpha"]),
            make_block(block_id="a2", order=2, text="Alpha body", source_file="file_a.docx", section_path=["Alpha"]),
        ],
        "file_b.docx": [
            make_block(block_id="b1", order=1, text="Beta", block_type="heading", source_file="file_b.docx", section_path=["Beta"]),
            make_block(block_id="b2", order=2, text="Beta body", source_file="file_b.docx", section_path=["Beta"]),
        ],
    }

    output = chunk_consolidated_blocks(consolidated, max_chars=100)

    assert set(output) == {"file_a.docx", "file_b.docx"}
    assert all(chunk["source_file"] == "file_a.docx" for chunk in output["file_a.docx"])
    assert all(chunk["source_file"] == "file_b.docx" for chunk in output["file_b.docx"])
    assert output["file_a.docx"][0]["block_ids"] == ["a1", "a2"]
    assert output["file_b.docx"][0]["block_ids"] == ["b1", "b2"]


def test_chunk_file_blocks_rejects_mixed_source_files() -> None:
    blocks = [
        make_block(block_id="a1", order=1, text="A", source_file="file_a.docx"),
        make_block(block_id="b1", order=2, text="B", source_file="file_b.docx"),
    ]

    with pytest.raises(ValueError):
        StructuralChunker(tokenizer=FakeTokenizer()).chunk_file_blocks(blocks)


def test_structural_chunker_does_not_duplicate_heading_after_table_boundary() -> None:
    blocks = [
        make_block(block_id="h1", order=1, text="Overview", block_type="heading", section_path=["Overview"]),
        make_block(block_id="p1", order=2, text="Overview body.", section_path=["Overview"]),
        make_block(block_id="t1", order=3, text="Columns: Name, Value\nRow 1: Revenue, 20", block_type="table", section_path=["Overview"]),
        make_block(block_id="h2", order=4, text="Next Section", block_type="heading", section_path=["Next Section"]),
        make_block(block_id="p2", order=5, text="Next body.", section_path=["Next Section"]),
    ]

    chunks = StructuralChunker(max_chars=120, tokenizer=FakeTokenizer()).chunk_file_blocks(blocks)

    assert [chunk["heading"] for chunk in chunks] == ["Overview", "Overview", "Next Section"]
    assert [chunk["chunk_type"] for chunk in chunks] == ["section", "table", "section"]


def test_structural_chunker_preserves_overlap_when_splitting_long_sections() -> None:
    blocks = [
        make_block(block_id="h1", order=1, text="Clinical Notes", block_type="heading", section_path=["Clinical Notes"]),
        make_block(
            block_id="p1",
            order=2,
            text=(
                "Sentence one introduces the patient context. "
                "Sentence two records the baseline symptom history. "
                "Sentence three explains the response to treatment. "
                "Sentence four captures the next-step monitoring plan."
            ),
            section_path=["Clinical Notes"],
        ),
    ]

    chunks = StructuralChunker(
        max_chars=500,
        max_tokens=24,
        overlap_tokens=10,
        tokenizer=FakeTokenizer(),
    ).chunk_file_blocks(blocks)

    assert len(chunks) >= 2
    assert "Sentence three explains the response to treatment." in chunks[0]["text"]
    assert "Sentence three explains the response to treatment." in chunks[1]["text"]
    assert all(chunk["heading"] == "Clinical Notes" for chunk in chunks)
