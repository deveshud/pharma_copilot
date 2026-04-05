from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parsers.docx_parser import DocxParser, DocumentBlock
from parsers.ppt_parser import PptParser, PptBlock, build_slide_level_text
from parsers.xlsx_parser import XlsxParser, XlsxBlock, XlsxParserError


SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"

NORMALIZED_FIELDS = {
    "block_id",
    "source_file",
    "source_path",
    "doc_type",
    "block_type",
    "order",
    "text",
    "section_path",
    "page_number",
    "slide_number",
    "shape_index",
    "sheet_name",
    "row_number",
    "metadata",
}


def first_sample_file(extension: str) -> Path:
    matches = sorted(
        path for path in SAMPLE_DOCS.glob(f"*{extension}")
        if path.is_file() and not path.name.startswith("~$")
    )
    assert matches, f"No sample file found for extension {extension}"
    return matches[0]


def test_docx_parser_returns_normalized_blocks() -> None:
    blocks = DocxParser().parse(str(first_sample_file(".docx")))

    assert blocks
    assert all(isinstance(block, DocumentBlock) for block in blocks)
    assert all(block.doc_type == "docx" for block in blocks)
    assert all(NORMALIZED_FIELDS.issubset(set(block.__dict__.keys())) for block in blocks)
    assert any(block.block_type == "paragraph" for block in blocks)


def test_ppt_parser_returns_normalized_blocks_and_slide_summary() -> None:
    parsed_results = {"sample.pptx": PptParser().parse(str(first_sample_file(".pptx")))}
    blocks = parsed_results["sample.pptx"]

    assert blocks
    assert all(isinstance(block, PptBlock) for block in blocks)
    assert all(block.doc_type == "pptx" for block in blocks)
    assert all(NORMALIZED_FIELDS.issubset(set(block.__dict__.keys())) for block in blocks)
    assert all(block.slide_number is not None for block in blocks)

    slide_summary = build_slide_level_text(parsed_results)
    assert slide_summary["sample.pptx"]
    assert all("slide_number" in slide for slide in slide_summary["sample.pptx"])


def test_xlsx_parser_returns_normalized_blocks() -> None:
    blocks = XlsxParser().parse(str(first_sample_file(".xlsx")))

    assert blocks
    assert all(isinstance(block, XlsxBlock) for block in blocks)
    assert all(block.doc_type == "xlsx" for block in blocks)
    assert all(NORMALIZED_FIELDS.issubset(set(block.__dict__.keys())) for block in blocks)
    assert any(block.block_type == "row" for block in blocks)
    assert any(block.sheet_name for block in blocks if block.block_type != "workbook_metadata")


def test_xlsx_parser_raises_error_for_corrupted_file(tmp_path: Path) -> None:
    bad_xlsx = tmp_path / "corrupted.xlsx"
    bad_xlsx.write_text("not a valid xlsx", encoding="utf-8")

    with pytest.raises(XlsxParserError):
        XlsxParser().parse(str(bad_xlsx))
