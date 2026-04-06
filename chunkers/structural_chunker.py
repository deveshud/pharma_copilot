from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import json
import re
import textwrap

ChunkDict = Dict[str, Any]
ConsolidatedBlocks = Mapping[str, Sequence[Mapping[str, Any]]]


@dataclass(frozen=True)
class NormalizedBlock:
    block_id: str
    source_file: str
    source_path: str
    doc_type: str
    block_type: str
    order: int
    text: str
    section_path: Tuple[str, ...]
    page_number: int | None = None
    slide_number: int | None = None
    shape_index: int | None = None
    sheet_name: str | None = None
    row_number: int | None = None
    metadata: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, block: Mapping[str, Any]) -> "NormalizedBlock":
        section_path = tuple(
            str(part).strip()
            for part in (block.get("section_path") or [])
            if str(part).strip()
        )

        return cls(
            block_id=str(block.get("block_id", "")).strip(),
            source_file=str(block.get("source_file", "")).strip(),
            source_path=str(block.get("source_path", "")).strip(),
            doc_type=str(block.get("doc_type", "")).strip(),
            block_type=str(block.get("block_type", "")).strip().lower(),
            order=int(block.get("order", 0)),
            text=_clean_text(block.get("text", "")),
            section_path=section_path,
            page_number=_coerce_int(block.get("page_number")),
            slide_number=_coerce_int(block.get("slide_number")),
            shape_index=_coerce_int(block.get("shape_index")),
            sheet_name=_coerce_optional_str(block.get("sheet_name")),
            row_number=_coerce_int(block.get("row_number")),
            metadata=dict(block.get("metadata") or {}),
        )


@dataclass(frozen=True)
class _ChunkSegment:
    block_id: str
    block_type: str
    order: int
    text: str
    page_number: int | None
    slide_number: int | None
    shape_index: int | None
    sheet_name: str | None
    row_number: int | None


class StructuralChunker:
    """
    Structure-aware chunker that preserves document semantics.

    Design goals:
    - process one file at a time
    - respect block ordering
    - use heading blocks to establish section context
    - use section_path to avoid cross-section merges
    - keep tables isolated from narrative text
    - split large sections with a max_chars safeguard
    - preserve block_ids for traceability
    """

    def __init__(
        self,
        max_chars: int = 2000,
        heading_block_types: Iterable[str] | None = None,
        table_block_types: Iterable[str] | None = None,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be a positive integer.")

        heading_types = heading_block_types or {"heading", "title", "header"}
        table_types = table_block_types or {"table"}

        self.max_chars = max_chars
        self.heading_block_types = {block_type.lower() for block_type in heading_types}
        self.table_block_types = {block_type.lower() for block_type in table_types}

    def chunk_file_blocks(self, blocks: Sequence[Mapping[str, Any]]) -> List[ChunkDict]:
        normalized_blocks = self._prepare_blocks(blocks)
        if not normalized_blocks:
            return []

        file_context = self._build_file_context(normalized_blocks)
        chunks: List[ChunkDict] = []
        chunk_counter = 0

        current_section_path: Tuple[str, ...] = tuple()
        current_heading: NormalizedBlock | None = None
        current_body_segments: List[_ChunkSegment] = []
        heading_emitted = False

        def flush_section(*, clear_context: bool) -> None:
            nonlocal chunk_counter, current_section_path, current_heading, current_body_segments, heading_emitted

            if current_heading is None and not current_body_segments:
                return

            if current_heading is not None and not current_body_segments and heading_emitted:
                return

            chunk_counter += 1
            chunks.append(
                self._build_section_chunk(
                    chunk_index=chunk_counter,
                    file_context=file_context,
                    section_path=current_section_path,
                    heading=current_heading,
                    body_segments=current_body_segments,
                )
            )
            current_body_segments = []
            if current_heading is not None:
                heading_emitted = True

            if clear_context:
                current_heading = None
                current_section_path = tuple()
                heading_emitted = False

        for block in normalized_blocks:
            if not block.text:
                continue

            block_section_path = block.section_path or current_section_path

            if block.block_type in self.heading_block_types:
                flush_section(clear_context=True)
                current_heading = block
                current_section_path = block.section_path or (block.text,)
                heading_emitted = False
                continue

            if block.block_type in self.table_block_types:
                table_section_path = block.section_path or current_section_path
                same_section = table_section_path == current_section_path

                flush_section(clear_context=not same_section)

                if not same_section:
                    current_section_path = table_section_path
                    current_heading = None
                    heading_emitted = False

                chunk_counter += 1
                chunks.append(
                    self._build_table_chunk(
                        chunk_index=chunk_counter,
                        file_context=file_context,
                        block=block,
                        heading=current_heading if same_section else None,
                        section_path=table_section_path,
                    )
                )
                continue

            if block_section_path != current_section_path:
                flush_section(clear_context=True)
                current_section_path = block_section_path
                current_heading = None
                heading_emitted = False

            allowed_body_chars = self._body_char_budget(current_heading)
            for piece in self._split_text(block.text, allowed_body_chars):
                segment = self._segment_from_block(block, piece)
                candidate_segments = current_body_segments + [segment]

                if current_body_segments and self._render_section_text(current_heading, candidate_segments):
                    if len(self._render_section_text(current_heading, candidate_segments)) > self.max_chars:
                        flush_section(clear_context=False)

                current_body_segments.append(segment)

        flush_section(clear_context=True)
        return chunks

    def chunk_consolidated_blocks(self, consolidated_blocks: ConsolidatedBlocks) -> Dict[str, List[ChunkDict]]:
        chunked_output: Dict[str, List[ChunkDict]] = {}

        for file_name, blocks in consolidated_blocks.items():
            chunked_output[file_name] = self.chunk_file_blocks(blocks)

        return chunked_output

    @staticmethod
    def load_consolidated_blocks(input_path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
        path = Path(input_path)
        with path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        if not isinstance(data, dict):
            raise ValueError("Expected consolidated JSON to be an object keyed by source file.")

        return {
            str(file_name): [dict(block) for block in blocks]
            for file_name, blocks in data.items()
        }

    @staticmethod
    def save_chunked_output(
        chunked_output: Mapping[str, Sequence[Mapping[str, Any]]],
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable_output = {
            str(file_name): [dict(chunk) for chunk in chunks]
            for file_name, chunks in chunked_output.items()
        }

        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(serializable_output, file_handle, indent=2, ensure_ascii=False)

        return path

    def _prepare_blocks(self, blocks: Sequence[Mapping[str, Any]]) -> List[NormalizedBlock]:
        normalized_blocks = [NormalizedBlock.from_dict(block) for block in blocks]
        normalized_blocks.sort(key=lambda block: (block.order, block.block_id))

        file_names = {block.source_file for block in normalized_blocks if block.source_file}
        if len(file_names) > 1:
            raise ValueError("chunk_file_blocks received blocks from multiple files.")

        return normalized_blocks

    def _build_file_context(self, blocks: Sequence[NormalizedBlock]) -> Dict[str, Any]:
        first_block = blocks[0]
        return {
            "source_file": first_block.source_file,
            "source_path": first_block.source_path,
            "doc_type": first_block.doc_type,
        }

    def _build_section_chunk(
        self,
        chunk_index: int,
        file_context: Mapping[str, Any],
        section_path: Tuple[str, ...],
        heading: NormalizedBlock | None,
        body_segments: Sequence[_ChunkSegment],
    ) -> ChunkDict:
        text = self._render_section_text(heading, body_segments)
        ordered_segments = self._ordered_segments(heading, body_segments)

        return self._build_chunk_dict(
            chunk_index=chunk_index,
            file_context=file_context,
            chunk_type="section",
            section_path=section_path,
            heading=heading.text if heading else None,
            text=text,
            segments=ordered_segments,
            metadata={
                "max_chars": self.max_chars,
                "source_block_count": len(self._ordered_unique([segment.block_id for segment in ordered_segments])),
            },
        )

    def _build_table_chunk(
        self,
        chunk_index: int,
        file_context: Mapping[str, Any],
        block: NormalizedBlock,
        heading: NormalizedBlock | None,
        section_path: Tuple[str, ...],
    ) -> ChunkDict:
        segment = self._segment_from_block(block, block.text)

        return self._build_chunk_dict(
            chunk_index=chunk_index,
            file_context=file_context,
            chunk_type="table",
            section_path=section_path,
            heading=heading.text if heading else None,
            text=block.text,
            segments=[segment],
            metadata={
                "max_chars": self.max_chars,
                "source_block_count": 1,
                "original_block_type": block.block_type,
            },
        )

    def _build_chunk_dict(
        self,
        chunk_index: int,
        file_context: Mapping[str, Any],
        chunk_type: str,
        section_path: Tuple[str, ...],
        heading: str | None,
        text: str,
        segments: Sequence[_ChunkSegment],
        metadata: Mapping[str, Any],
    ) -> ChunkDict:
        return {
            "chunk_id": f"{file_context['source_file']}::chunk_{chunk_index:04d}",
            "source_file": file_context["source_file"],
            "source_path": file_context["source_path"],
            "doc_type": file_context["doc_type"],
            "chunk_type": chunk_type,
            "section_path": list(section_path),
            "section_key": " > ".join(section_path),
            "heading": heading,
            "text": text,
            "char_count": len(text),
            "block_ids": self._ordered_unique([segment.block_id for segment in segments]),
            "block_types": self._ordered_unique([segment.block_type for segment in segments]),
            "order_start": min(segment.order for segment in segments),
            "order_end": max(segment.order for segment in segments),
            "page_numbers": sorted({segment.page_number for segment in segments if segment.page_number is not None}),
            "slide_numbers": sorted({segment.slide_number for segment in segments if segment.slide_number is not None}),
            "shape_indices": sorted({segment.shape_index for segment in segments if segment.shape_index is not None}),
            "sheet_names": self._ordered_unique(
                [segment.sheet_name for segment in segments if segment.sheet_name]
            ),
            "row_numbers": sorted({segment.row_number for segment in segments if segment.row_number is not None}),
            "metadata": dict(metadata),
        }

    def _segment_from_block(self, block: NormalizedBlock, text: str) -> _ChunkSegment:
        return _ChunkSegment(
            block_id=block.block_id,
            block_type=block.block_type,
            order=block.order,
            text=text,
            page_number=block.page_number,
            slide_number=block.slide_number,
            shape_index=block.shape_index,
            sheet_name=block.sheet_name,
            row_number=block.row_number,
        )

    def _ordered_segments(
        self,
        heading: NormalizedBlock | None,
        body_segments: Sequence[_ChunkSegment],
    ) -> List[_ChunkSegment]:
        ordered = list(body_segments)
        if heading is not None:
            ordered.insert(0, self._segment_from_block(heading, heading.text))
        return ordered

    def _render_section_text(
        self,
        heading: NormalizedBlock | None,
        body_segments: Sequence[_ChunkSegment],
    ) -> str:
        parts: List[str] = []

        if heading is not None and heading.text:
            parts.append(heading.text)

        parts.extend(segment.text for segment in body_segments if segment.text)
        return "\n\n".join(parts)

    def _body_char_budget(self, heading: NormalizedBlock | None) -> int:
        if heading is None or not heading.text:
            return self.max_chars

        heading_prefix = f"{heading.text}\n\n"
        remaining = self.max_chars - len(heading_prefix)

        if remaining <= 0:
            return self.max_chars

        return remaining

    def _split_text(self, text: str, max_chars: int) -> List[str]:
        cleaned_text = _clean_text(text)
        if not cleaned_text:
            return []

        if len(cleaned_text) <= max_chars:
            return [cleaned_text]

        paragraph_units = [part.strip() for part in re.split(r"\n{2,}", cleaned_text) if part.strip()]
        if len(paragraph_units) > 1:
            return self._pack_units(paragraph_units, max_chars, separator="\n\n")

        line_units = [part.strip() for part in cleaned_text.splitlines() if part.strip()]
        if len(line_units) > 1:
            return self._pack_units(line_units, max_chars, separator="\n")

        sentence_units = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned_text) if part.strip()]
        if len(sentence_units) > 1:
            return self._pack_units(sentence_units, max_chars, separator=" ")

        return textwrap.wrap(
            cleaned_text,
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [cleaned_text]

    def _pack_units(self, units: Sequence[str], max_chars: int, separator: str) -> List[str]:
        chunks: List[str] = []
        current_parts: List[str] = []

        for unit in units:
            if len(unit) > max_chars:
                if current_parts:
                    chunks.append(separator.join(current_parts))
                    current_parts = []

                chunks.extend(self._split_text(unit, max_chars))
                continue

            candidate = separator.join(current_parts + [unit]) if current_parts else unit
            if current_parts and len(candidate) > max_chars:
                chunks.append(separator.join(current_parts))
                current_parts = [unit]
            else:
                current_parts.append(unit)

        if current_parts:
            chunks.append(separator.join(current_parts))

        return chunks

    @staticmethod
    def _ordered_unique(values: Sequence[Any]) -> List[Any]:
        unique_values: List[Any] = []
        seen = set()

        for value in values:
            if value in seen:
                continue
            seen.add(value)
            unique_values.append(value)

        return unique_values


def chunk_file_blocks(
    blocks: Sequence[Mapping[str, Any]],
    max_chars: int = 2000,
) -> List[ChunkDict]:
    return StructuralChunker(max_chars=max_chars).chunk_file_blocks(blocks)


def chunk_consolidated_blocks(
    consolidated_blocks: ConsolidatedBlocks,
    max_chars: int = 2000,
) -> Dict[str, List[ChunkDict]]:
    return StructuralChunker(max_chars=max_chars).chunk_consolidated_blocks(consolidated_blocks)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_str(value: Any) -> str | None:
    text = _clean_text(value)
    return text or None
