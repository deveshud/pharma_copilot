from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_DOCS_DIR = PROJECT_ROOT / "data" / "sample_docs"
PARSED_OUTPUTS_DIR = PROJECT_ROOT / "parsed_outputs"

from parsers.docx_parser import DocxParser
from parsers.pdf_parser import PdfParser
from parsers.ppt_parser import PptParser
from parsers.ppt_parser import build_slide_level_text
from parsers.xlsx_parser import XlsxParser


def build_parser_registry() -> Dict[str, object]:
    return {
        ".docx": DocxParser(heading_threshold=3),
        ".pdf": PdfParser(include_document_metadata=True, include_empty_pages=True),
        ".pptx": PptParser(include_notes=True, include_images=True),
        ".xlsx": XlsxParser(
            include_workbook_metadata=True,
            include_sheet_metadata=True,
            include_empty_rows=False,
            infer_header_row=True,
        ),
    }


def build_output_registry() -> Dict[str, str]:
    return {
        ".docx": "normalized_docx_blocks.json",
        ".pdf": "normalized_pdf_blocks.json",
        ".pptx": "normalized_ppt_blocks.json",
        ".xlsx": "normalized_xlsx_blocks.json",
    }


def discover_supported_files(folder_path: str) -> List[Path]:
    folder = Path(folder_path)
    registry = build_parser_registry()

    return [
        file_path
        for file_path in sorted(folder.rglob("*"))
        if file_path.is_file()
        and file_path.suffix.lower() in registry
        and not file_path.name.startswith("~$")
    ]


def ingest_documents(folder_path: str) -> Dict[str, List[Any]]:
    folder = Path(folder_path)
    registry = build_parser_registry()
    parsed_results: Dict[str, List[Any]] = {}

    for file_path in discover_supported_files(folder_path):
        parser = registry[file_path.suffix.lower()]
        relative_key = str(file_path.relative_to(folder)).replace("\\", "/")
        parsed_results[relative_key] = parser.parse(str(file_path))

    return parsed_results


def save_ingestion_output(
    parsed_results: Dict[str, List[Any]],
    output_file: str = "parsed_outputs\\normalized_ingestion_blocks.json",
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_ready = {
        file_name: [asdict(block) for block in blocks]
        for file_name, blocks in parsed_results.items()
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2, ensure_ascii=False)


def partition_results_by_extension(parsed_results: Dict[str, List[Any]]) -> Dict[str, Dict[str, List[Any]]]:
    partitioned: Dict[str, Dict[str, List[Any]]] = {}

    for file_name, blocks in parsed_results.items():
        extension = Path(file_name).suffix.lower()
        partitioned.setdefault(extension, {})[file_name] = blocks

    return partitioned


def save_separate_outputs(parsed_results: Dict[str, List[Any]], output_dir: Path | None = None) -> Dict[str, Path]:
    output_dir = output_dir or PARSED_OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_registry = build_output_registry()
    partitioned_results = partition_results_by_extension(parsed_results)
    saved_files: Dict[str, Path] = {}

    for extension, file_results in partitioned_results.items():
        output_name = output_registry.get(extension)
        if not output_name:
            continue

        output_path = output_dir / output_name
        save_ingestion_output(file_results, str(output_path))
        saved_files[extension] = output_path

    ppt_results = partitioned_results.get(".pptx", {})
    if ppt_results:
        ppt_slide_output = output_dir / "normalized_ppt_slides.json"
        with ppt_slide_output.open("w", encoding="utf-8") as f:
            json.dump(build_slide_level_text(ppt_results), f, indent=2, ensure_ascii=False)
        saved_files[".pptx_slides"] = ppt_slide_output

    return saved_files


def save_all_outputs(parsed_results: Dict[str, List[Any]], output_dir: Path | None = None) -> Dict[str, Path]:
    output_dir = output_dir or PARSED_OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = save_separate_outputs(parsed_results, output_dir)

    consolidated_output = output_dir / "normalized_ingestion_blocks.json"
    save_ingestion_output(parsed_results, str(consolidated_output))
    saved_files["consolidated"] = consolidated_output

    return saved_files


def print_ingestion_summary(parsed_results: Dict[str, List[Any]]) -> None:
    print("INGESTED FILES:")
    for file_name, blocks in parsed_results.items():
        print(f"{file_name}: {len(blocks)} blocks")


def print_sample_output(parsed_results: Dict[str, List[Any]], sample_size: int = 3) -> None:
    print("\nSAMPLE OUTPUT:")
    for file_name, blocks in parsed_results.items():
        print(f"\n--- {file_name} ---")
        for block in blocks[:sample_size]:
            print(json.dumps(asdict(block), ensure_ascii=True))
        break


def print_saved_outputs(saved_files: Dict[str, Path]) -> None:
    print("\nSaved outputs:")
    for key, output_path in sorted(saved_files.items()):
        print(f"- {output_path.name}")


if __name__ == "__main__":
    folder = SAMPLE_DOCS_DIR
    print(f"Reading supported documents from: {folder.resolve()}")

    if not folder.exists():
        print(f"Folder not found: {folder.resolve()}")
    else:
        files = discover_supported_files(str(folder))
        print("Supported files found:")
        for file_path in files:
            print("-", file_path.name)

        parsed_results = ingest_documents(str(folder))
        print_ingestion_summary(parsed_results)
        print_sample_output(parsed_results)

        saved_files = save_all_outputs(parsed_results, PARSED_OUTPUTS_DIR)
        print_saved_outputs(saved_files)
