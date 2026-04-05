from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from pathlib import Path
import uuid


@dataclass
class DocumentBlock:
    block_id: str
    source_file: str
    source_path: str
    doc_type: str
    block_type: str                  # heading | paragraph | table
    order: int
    text: str
    section_path: List[str] = field(default_factory=list)

    style_name: Optional[str] = None
    heading_score: Optional[int] = None
    paragraph_index: Optional[int] = None
    table_row_count: Optional[int] = None
    table_col_count: Optional[int] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


class DocxParser:
    def __init__(self, heading_threshold: int = 3):
        self.heading_threshold = heading_threshold

    def _make_block_id(self) -> str:
        return f"block_{uuid.uuid4().hex[:12]}"

    def _clean_text(self, text: str) -> str:
        return text.strip() if text else ""

    def _is_title_case_like(self, text: str) -> bool:
        words = text.split()
        if not words:
            return False

        meaningful_words = [w for w in words if w.isalpha()]
        if not meaningful_words:
            return False

        title_like_count = sum(1 for w in meaningful_words if w[:1].isupper())
        return (title_like_count / len(meaningful_words)) >= 0.7

    def _paragraph_has_bold(self, para: Paragraph) -> bool:
        return any(run.bold for run in para.runs)

    def _score_heading_likelihood(self, para: Paragraph) -> tuple[int, dict]:
        text = self._clean_text(para.text)
        style_name = para.style.name if para.style else ""

        score = 0
        reasons = {}

        if not text:
            return score, {"empty_text": True}

        word_count = len(text.split())
        has_bold = self._paragraph_has_bold(para)
        lower_style = style_name.lower()

        if word_count <= 12:
            score += 2
            reasons["short_text"] = True

        if has_bold:
            score += 2
            reasons["has_bold"] = True

        if not text.endswith((".", ";", ":", ",")):
            score += 1
            reasons["no_sentence_punctuation"] = True

        if self._is_title_case_like(text):
            score += 1
            reasons["title_case_like"] = True

        if text.isupper() and len(text) <= 40:
            score += 1
            reasons["all_caps_short"] = True

        if word_count > 20:
            score -= 3
            reasons["long_text"] = True

        if "list" in lower_style or "bullet" in lower_style:
            score -= 2
            reasons["list_or_bullet_style"] = True

        if "annotation" in lower_style:
            score -= 2
            reasons["annotation_style"] = True

        if text.startswith(("•", "-", "*")):
            score -= 2
            reasons["starts_like_bullet"] = True

        return score, reasons

    def _table_to_text(self, table: Table) -> tuple[str, int, int]:
        rows_data = []

        for row in table.rows:
            row_values = []
            for cell in row.cells:
                cell_text = self._clean_text(cell.text)
                row_values.append(cell_text)

            if any(row_values):
                rows_data.append(row_values)

        if not rows_data:
            return "", 0, 0

        row_count = len(rows_data)
        col_count = max(len(row) for row in rows_data)

        lines = []

        header = rows_data[0]
        lines.append(f"Columns: {', '.join(header)}")

        for idx, row in enumerate(rows_data[1:], start=1):
            lines.append(f"Row {idx}: {', '.join(row)}")

        table_text = "\n".join(lines)
        return table_text, row_count, col_count

    def _iter_block_items(self, doc: Document):
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, doc)
            elif isinstance(child, CT_Tbl):
                yield Table(child, doc)

    def parse(self, file_path: str) -> List[DocumentBlock]:
        doc = Document(file_path)

        source_file = Path(file_path).name
        source_path = str(Path(file_path).resolve())

        blocks: List[DocumentBlock] = []
        current_section = None
        order_counter = 0
        paragraph_counter = 0

        for item in self._iter_block_items(doc):
            if isinstance(item, Paragraph):
                text = self._clean_text(item.text)
                if not text:
                    continue

                paragraph_counter += 1
                order_counter += 1

                style_name = item.style.name if item.style else "No Style"
                score, reasons = self._score_heading_likelihood(item)
                is_heading_like = score >= self.heading_threshold

                if is_heading_like:
                    current_section = text

                block_type = "heading" if is_heading_like else "paragraph"

                block = DocumentBlock(
                    block_id=self._make_block_id(),
                    source_file=source_file,
                    source_path=source_path,
                    doc_type="docx",
                    block_type=block_type,
                    order=order_counter,
                    text=text,
                    section_path=[current_section] if current_section else [],
                    style_name=style_name,
                    heading_score=score,
                    paragraph_index=paragraph_counter,
                    metadata={
                        "is_heading_like": is_heading_like,
                        "heading_reasons": reasons
                    }
                )
                blocks.append(block)

            elif isinstance(item, Table):
                table_text, row_count, col_count = self._table_to_text(item)
                if not table_text:
                    continue

                order_counter += 1

                block = DocumentBlock(
                    block_id=self._make_block_id(),
                    source_file=source_file,
                    source_path=source_path,
                    doc_type="docx",
                    block_type="table",
                    order=order_counter,
                    text=table_text,
                    section_path=[current_section] if current_section else [],
                    table_row_count=row_count,
                    table_col_count=col_count,
                    metadata={}
                )
                blocks.append(block)

        return blocks

