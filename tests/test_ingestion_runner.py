from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runners.run_ingestion import (
    discover_supported_files,
    ingest_documents,
    save_all_outputs,
    save_ingestion_output,
)


SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"


def first_sample_file(extension: str) -> Path:
    matches = sorted(
        path for path in SAMPLE_DOCS.glob(f"*{extension}")
        if path.is_file() and not path.name.startswith("~$")
    )
    assert matches, f"No sample file found for extension {extension}"
    return matches[0]


def test_ingestion_runner_dispatches_supported_extensions_and_saves_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "input_docs"
    input_dir.mkdir()

    sample_paths = [
        first_sample_file(".docx"),
        first_sample_file(".pdf"),
        first_sample_file(".pptx"),
        first_sample_file(".xlsx"),
    ]
    sample_files = [path.name for path in sample_paths]

    for sample_path in sample_paths:
        shutil.copy2(sample_path, input_dir / sample_path.name)

    (input_dir / "~$temp.xlsx").write_text("temporary lock file", encoding="utf-8")
    (input_dir / "notes.txt").write_text("unsupported", encoding="utf-8")

    discovered_files = discover_supported_files(str(input_dir))
    assert [path.name for path in discovered_files] == sorted(sample_files)

    parsed_results = ingest_documents(str(input_dir))
    assert sorted(parsed_results) == sorted(sample_files)
    assert all(parsed_results[file_name] for file_name in sample_files)

    doc_types = {
        file_name: {block.doc_type for block in blocks}
        for file_name, blocks in parsed_results.items()
    }
    expected_doc_types = {
        sample_path.name: {sample_path.suffix.lstrip(".")}
        for sample_path in sample_paths
    }
    assert doc_types == expected_doc_types

    output_file = tmp_path / "normalized_ingestion_blocks.json"
    save_ingestion_output(parsed_results, str(output_file))
    assert output_file.exists()

    saved_files = save_all_outputs(parsed_results, tmp_path)
    assert (tmp_path / "normalized_docx_blocks.json").exists()
    assert (tmp_path / "normalized_pdf_blocks.json").exists()
    assert (tmp_path / "normalized_ppt_blocks.json").exists()
    assert (tmp_path / "normalized_ppt_slides.json").exists()
    assert (tmp_path / "normalized_xlsx_blocks.json").exists()
    assert (tmp_path / "normalized_ingestion_blocks.json").exists()
    assert "consolidated" in saved_files
