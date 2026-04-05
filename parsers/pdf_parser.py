from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re
import uuid

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass
class PdfBlock:
    block_id: str
    source_file: str
    source_path: str
    doc_type: str
    block_type: str
    order: int
    text: str
    section_path: List[str] = field(default_factory=list)
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    shape_index: Optional[int] = None
    sheet_name: Optional[str] = None
    row_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PdfParserError(Exception):
    """Raised when a PDF cannot be parsed safely."""


class PdfParser:
    def __init__(
        self,
        include_document_metadata: bool = True,
        include_empty_pages: bool = False,
    ) -> None:
        self.include_document_metadata = include_document_metadata
        self.include_empty_pages = include_empty_pages

    def _make_block_id(self) -> str:
        return f"pdf_{uuid.uuid4().hex[:12]}"

    def _normalize_text(self, text: Optional[str]) -> str:
        if not text:
            return ""

        text = text.replace("\x00", " ")
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _safe_stringify(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _extract_document_metadata(self, reader: PdfReader) -> Dict[str, Any]:
        raw_metadata = reader.metadata or {}
        metadata: Dict[str, Any] = {
            "page_count": len(reader.pages),
            "is_encrypted": bool(reader.is_encrypted),
        }

        for key, value in raw_metadata.items():
            clean_key = str(key).lstrip("/") if key is not None else "unknown"
            metadata[clean_key] = self._safe_stringify(value)

        return metadata

    def _metadata_to_text(self, metadata: Dict[str, Any]) -> str:
        lines = []

        for key, value in metadata.items():
            if value in (None, "", []):
                continue
            pretty_key = key.replace("_", " ").title()
            lines.append(f"{pretty_key}: {value}")

        return "\n".join(lines)

    def _page_image_count(self, page: Any) -> int:
        try:
            return len(list(page.images))
        except Exception:
            return 0

    def _page_dimensions(self, page: Any) -> Dict[str, float]:
        mediabox = getattr(page, "mediabox", None)
        if mediabox is None:
            return {}

        try:
            width = float(mediabox.width)
            height = float(mediabox.height)
        except Exception:
            return {}

        return {
            "width_points": round(width, 2),
            "height_points": round(height, 2),
        }

    def parse(self, file_path: str) -> List[PdfBlock]:
        path = Path(file_path)
        source_path = str(path.resolve())

        if not path.exists():
            raise PdfParserError(f"PDF file not found: {source_path}")

        if path.is_dir():
            raise PdfParserError(f"Expected a PDF file but received a directory: {source_path}")

        try:
            with path.open("rb") as pdf_file:
                reader = PdfReader(pdf_file)
                document_metadata = self._extract_document_metadata(reader)

                blocks: List[PdfBlock] = []
                order = 0

                if self.include_document_metadata:
                    metadata_text = self._metadata_to_text(document_metadata)
                    if metadata_text:
                        order += 1
                        blocks.append(
                            PdfBlock(
                                block_id=self._make_block_id(),
                                source_file=path.name,
                                source_path=source_path,
                                doc_type="pdf",
                                block_type="document_metadata",
                                order=order,
                                text=metadata_text,
                                metadata=document_metadata,
                            )
                        )

                for page_number, page in enumerate(reader.pages, start=1):
                    try:
                        extracted_text = self._normalize_text(page.extract_text())
                    except Exception as exc:
                        extracted_text = ""
                        page_error = str(exc)
                    else:
                        page_error = None

                    image_count = self._page_image_count(page)
                    page_metadata = {
                        "page_number": page_number,
                        "image_count": image_count,
                        "rotation": self._safe_stringify(getattr(page, "rotation", 0)),
                        **self._page_dimensions(page),
                    }

                    if page_error:
                        page_metadata["extraction_error"] = page_error

                    if extracted_text:
                        page_metadata["extraction_status"] = "text_extracted"
                        page_metadata["character_count"] = len(extracted_text)
                        if image_count:
                            page_metadata["contains_embedded_images"] = True

                        order += 1
                        blocks.append(
                            PdfBlock(
                                block_id=self._make_block_id(),
                                source_file=path.name,
                                source_path=source_path,
                                doc_type="pdf",
                                block_type="page_text",
                                order=order,
                                text=extracted_text,
                                page_number=page_number,
                                metadata=page_metadata,
                            )
                        )
                        continue

                    if image_count:
                        page_metadata["extraction_status"] = "image_only_or_scanned"
                        placeholder = (
                            f"Page {page_number} appears to be image-based or scanned. "
                            "No embedded text was extracted."
                        )
                    else:
                        page_metadata["extraction_status"] = "no_text_detected"
                        placeholder = f"Page {page_number} did not yield extractable text."

                    if self.include_empty_pages or image_count or page_error:
                        order += 1
                        blocks.append(
                            PdfBlock(
                                block_id=self._make_block_id(),
                                source_file=path.name,
                                source_path=source_path,
                                doc_type="pdf",
                                block_type="page_notice",
                                order=order,
                                text=placeholder,
                                page_number=page_number,
                                metadata=page_metadata,
                            )
                        )

                return blocks

        except PdfReadError as exc:
            raise PdfParserError(f"Could not read PDF '{source_path}'. The file may be corrupted: {exc}") from exc
        except OSError as exc:
            raise PdfParserError(f"Could not open PDF '{source_path}': {exc}") from exc
        except Exception as exc:
            raise PdfParserError(f"Unexpected error while parsing PDF '{source_path}': {exc}") from exc


def parse_pdf_folder(folder_path: str, parser: Optional[PdfParser] = None) -> Dict[str, List[PdfBlock]]:
    parser = parser or PdfParser()
    folder = Path(folder_path)
    parsed_results: Dict[str, List[PdfBlock]] = {}

    for file_path in sorted(folder.glob("*.pdf")):
        if file_path.name.startswith("~$"):
            continue
        parsed_results[file_path.name] = parser.parse(str(file_path))

    return parsed_results


def save_results_to_json(
    parsed_results: Dict[str, List[PdfBlock]],
    output_file: str = "parsed_pdf_output.json",
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_ready: Dict[str, Any] = {}

    for file_name, blocks in parsed_results.items():
        json_ready[file_name] = [asdict(block) for block in blocks]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2, ensure_ascii=False)
