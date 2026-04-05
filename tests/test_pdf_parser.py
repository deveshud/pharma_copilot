from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parsers.pdf_parser import PdfBlock, PdfParser, PdfParserError


SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"


def first_sample_file(extension: str) -> Path:
    matches = sorted(
        path for path in SAMPLE_DOCS.glob(f"*{extension}")
        if path.is_file() and not path.name.startswith("~$")
    )
    assert matches, f"No sample file found for extension {extension}"
    return matches[0]


def test_pdf_parser_extracts_metadata_and_text_blocks() -> None:
    parser = PdfParser(include_empty_pages=True)
    sample_pdf = first_sample_file(".pdf")

    blocks = parser.parse(str(sample_pdf))

    assert blocks
    assert all(isinstance(block, PdfBlock) for block in blocks)
    assert blocks[0].block_type == "document_metadata"
    assert any(block.block_type == "page_text" for block in blocks)
    assert all(block.doc_type == "pdf" for block in blocks)


def test_pdf_parser_raises_clear_error_for_corrupted_pdf(tmp_path: Path) -> None:
    bad_pdf = tmp_path / "corrupted.pdf"
    bad_pdf.write_text("not a valid pdf", encoding="utf-8")

    with pytest.raises(PdfParserError, match="Could not read PDF"):
        PdfParser().parse(str(bad_pdf))
