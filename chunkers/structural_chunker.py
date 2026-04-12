from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import json
import re
import textwrap

from utils.text_normalization import clean_text, join_narrative_parts, normalize_narrative_text, render_heading_with_body

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None

ChunkDict = Dict[str, Any]
ConsolidatedBlocks = Mapping[str, Sequence[Mapping[str, Any]]]
DEFAULT_TOKENIZER_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


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
            text=clean_text(block.get("text", "")),
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
    - split large sections with character and token safeguards
    - carry overlap between adjacent split chunks to preserve context
    - preserve block_ids for traceability
    """

    def __init__(
        self,
        max_chars: int = 2000,
        max_tokens: int | None = 380,
        overlap_tokens: int = 40,
        heading_block_types: Iterable[str] | None = None,
        table_block_types: Iterable[str] | None = None,
        min_section_chars: int = 120,
        tokenizer_name: str = DEFAULT_TOKENIZER_MODEL,
        tokenizer: Any | None = None,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be a positive integer.")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer when provided.")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be zero or greater.")
        if min_section_chars < 0:
            raise ValueError("min_section_chars must be zero or greater.")

        heading_types = heading_block_types or {"heading", "title", "header"}
        table_types = table_block_types or {"table"}

        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.heading_block_types = {block_type.lower() for block_type in heading_types}
        self.table_block_types = {block_type.lower() for block_type in table_types}
        self.min_section_chars = min_section_chars
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer or self._load_tokenizer(tokenizer_name)

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

            for body_group in self._group_section_segments(current_heading, current_body_segments):
                chunk_counter += 1
                chunks.append(
                    self._build_section_chunk(
                        chunk_index=chunk_counter,
                        file_context=file_context,
                        section_path=current_section_path,
                        heading=current_heading,
                        body_segments=body_group,
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
            if self._is_inferred_heading(block):
                block = replace(block, block_type="heading", section_path=block.section_path or (block.text,))

            block_section_path = block.section_path or current_section_path

            if self._is_heading_block(block):
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

                for table_text in self._split_table_text(
                    block.text,
                    sticky_header=self._extract_table_header_line(block.text),
                ):
                    chunk_counter += 1
                    chunks.append(
                        self._build_table_chunk(
                            chunk_index=chunk_counter,
                            file_context=file_context,
                            block=block,
                            text=table_text,
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
            allowed_body_tokens = self._body_token_budget(current_heading)

            for piece in self._split_text_into_segments(
                block.text,
                max_chars=allowed_body_chars,
                max_tokens=allowed_body_tokens,
            ):
                segment = self._segment_from_block(block, piece)
                current_body_segments.append(segment)

        flush_section(clear_context=True)
        return self._merge_short_context_chunks(chunks)

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
                "max_tokens": self.max_tokens,
                "overlap_tokens": self.overlap_tokens,
                "source_block_count": len(self._ordered_unique([segment.block_id for segment in ordered_segments])),
            },
        )

    def _build_table_chunk(
        self,
        chunk_index: int,
        file_context: Mapping[str, Any],
        block: NormalizedBlock,
        text: str,
        heading: NormalizedBlock | None,
        section_path: Tuple[str, ...],
    ) -> ChunkDict:
        segment = self._segment_from_block(block, text)

        return self._build_chunk_dict(
            chunk_index=chunk_index,
            file_context=file_context,
            chunk_type="table",
            section_path=section_path,
            heading=heading.text if heading else None,
            text=text,
            segments=[segment],
            metadata={
                "max_chars": self.max_chars,
                "max_tokens": self.max_tokens,
                "overlap_tokens": self.overlap_tokens,
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
            "file_name": file_context["source_file"],
            "source_file": file_context["source_file"],
            "source_path": file_context["source_path"],
            "doc_type": file_context["doc_type"],
            "chunk_type": chunk_type,
            "section_path": list(section_path),
            "section_key": " > ".join(section_path),
            "section_title": self._section_title(heading, section_path),
            "heading": heading,
            "text": text,
            "char_count": len(text),
            "block_ids": self._ordered_unique([segment.block_id for segment in segments]),
            "block_types": self._ordered_unique([segment.block_type for segment in segments]),
            "order_start": min(segment.order for segment in segments),
            "order_end": max(segment.order for segment in segments),
            "page_number": self._first_or_none(
                sorted({segment.page_number for segment in segments if segment.page_number is not None})
            ),
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
        body_parts = [segment.text for segment in body_segments if segment.text]
        return render_heading_with_body(heading.text if heading else None, body_parts)

    def _body_char_budget(self, heading: NormalizedBlock | None) -> int:
        if heading is None or not heading.text:
            return self.max_chars

        heading_prefix = f"{heading.text}\n\n"
        remaining = self.max_chars - len(heading_prefix)

        if remaining <= 0:
            return self.max_chars

        return remaining

    def _body_token_budget(self, heading: NormalizedBlock | None) -> int | None:
        if self.max_tokens is None:
            return None
        if heading is None or not heading.text:
            return self.max_tokens

        heading_prefix = f"{heading.text}\n\n"
        remaining = self.max_tokens - self._token_count(heading_prefix)

        if remaining <= 0:
            return self.max_tokens

        return remaining

    def _group_section_segments(
        self,
        heading: NormalizedBlock | None,
        body_segments: Sequence[_ChunkSegment],
    ) -> List[List[_ChunkSegment]]:
        if not body_segments:
            return [[]]

        groups: List[List[_ChunkSegment]] = []
        current_group: List[_ChunkSegment] = []

        for segment in body_segments:
            candidate_group = current_group + [segment]
            if current_group and not self._fits_within_budget(
                self._render_section_text(heading, candidate_group)
            ):
                groups.append(current_group)
                overlap_group = self._select_overlap_segments(current_group)
                current_group = self._trim_overlap_to_fit(
                    heading=heading,
                    overlap_segments=overlap_group,
                    next_segment=segment,
                )
                candidate_group = current_group + [segment]

            current_group = candidate_group

        if current_group:
            groups.append(current_group)

        return groups

    def _merge_short_context_chunks(self, chunks: Sequence[ChunkDict]) -> List[ChunkDict]:
        merged_chunks: List[ChunkDict] = []
        pending_prefixes: List[ChunkDict] = []

        for chunk in chunks:
            if self._is_short_standalone_section(chunk):
                # Colon headings usually introduce the next section; other short orphan
                # headings attach backward unless they are part of an existing prefix chain.
                if pending_prefixes or self._should_prefix_short_section(chunk) or not merged_chunks:
                    pending_prefixes.append(chunk)
                elif self._fits_within_budget(self._merged_text(merged_chunks[-1], chunk)):
                    merged_chunks[-1] = self._merge_chunk_pair(merged_chunks[-1], chunk, position="suffix")
                else:
                    pending_prefixes.append(chunk)
                continue

            if pending_prefixes:
                chunk = self._merge_prefix_chunks(pending_prefixes, chunk)
                pending_prefixes = []

            merged_chunks.append(chunk)

        if pending_prefixes:
            if merged_chunks:
                for prefix in pending_prefixes:
                    if self._fits_within_budget(self._merged_text(merged_chunks[-1], prefix)):
                        merged_chunks[-1] = self._merge_chunk_pair(merged_chunks[-1], prefix, position="suffix")
                    else:
                        merged_chunks.append(prefix)
            else:
                merged_chunks.extend(pending_prefixes)

        return self._renumber_chunks(merged_chunks)

    def _select_overlap_segments(self, segments: Sequence[_ChunkSegment]) -> List[_ChunkSegment]:
        if self.overlap_tokens <= 0 or not segments:
            return []

        overlap_segments: List[_ChunkSegment] = []
        token_total = 0

        for segment in reversed(segments):
            segment_tokens = self._token_count(segment.text)
            projected_total = token_total + segment_tokens

            if overlap_segments and projected_total > self.overlap_tokens:
                break
            if not overlap_segments and segment_tokens > self.overlap_tokens:
                break

            overlap_segments.insert(0, segment)
            token_total = projected_total

        return overlap_segments

    def _trim_overlap_to_fit(
        self,
        *,
        heading: NormalizedBlock | None,
        overlap_segments: Sequence[_ChunkSegment],
        next_segment: _ChunkSegment,
    ) -> List[_ChunkSegment]:
        trimmed_overlap = list(overlap_segments)
        while trimmed_overlap and not self._fits_within_budget(
            self._render_section_text(heading, trimmed_overlap + [next_segment])
        ):
            trimmed_overlap.pop(0)
        return trimmed_overlap

    def _split_table_text(self, text: str, sticky_header: str | None = None) -> List[str]:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return []

        if sticky_header and cleaned_text.startswith(sticky_header):
            body_text = cleaned_text[len(sticky_header):].lstrip("\n")
            if not body_text:
                return [sticky_header]

            header_chars = max(self.max_chars - len(f"{sticky_header}\n"), 1)
            header_tokens = self._subtract_token_headroom(self.max_tokens, sticky_header)
            body_pieces = self._split_text(
                body_text,
                max_chars=header_chars,
                max_tokens=header_tokens,
                prefer_lines=True,
            )
            return [f"{sticky_header}\n{piece}" for piece in body_pieces]

        return self._split_text(
            cleaned_text,
            max_chars=self.max_chars,
            max_tokens=self.max_tokens,
            prefer_lines=True,
        )

    def _split_text_into_segments(
        self,
        text: str,
        *,
        max_chars: int,
        max_tokens: int | None,
        prefer_lines: bool = False,
    ) -> List[str]:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return []
        if self._fits_within_budget(cleaned_text, max_chars=max_chars, max_tokens=max_tokens):
            return [cleaned_text]

        unit_builders = [
            [part.strip() for part in re.split(r"\n{2,}", cleaned_text) if part.strip()],
            [part.strip() for part in cleaned_text.splitlines() if part.strip()],
            [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned_text) if part.strip()],
            cleaned_text.split(),
        ]

        if prefer_lines:
            unit_builders[0], unit_builders[1] = unit_builders[1], unit_builders[0]

        for units in unit_builders:
            if len(units) > 1:
                atomic_segments: List[str] = []
                for unit in units:
                    if self._fits_within_budget(unit, max_chars=max_chars, max_tokens=max_tokens):
                        atomic_segments.append(unit)
                    else:
                        atomic_segments.extend(
                            self._split_text_into_segments(
                                unit,
                                max_chars=max_chars,
                                max_tokens=max_tokens,
                                prefer_lines=(units is unit_builders[1]),
                            )
                        )
                return atomic_segments

        return textwrap.wrap(
            cleaned_text,
            width=max(1, min(max_chars, self._fallback_wrap_width(max_tokens))),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [cleaned_text]

    def _split_text(
        self,
        text: str,
        *,
        max_chars: int,
        max_tokens: int | None,
        prefer_lines: bool = False,
    ) -> List[str]:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return []

        if self._fits_within_budget(cleaned_text, max_chars=max_chars, max_tokens=max_tokens):
            return [cleaned_text]

        unit_builders = [
            ([part.strip() for part in re.split(r"\n{2,}", cleaned_text) if part.strip()], "\n\n"),
            ([part.strip() for part in cleaned_text.splitlines() if part.strip()], "\n"),
            ([part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned_text) if part.strip()], " "),
            (cleaned_text.split(), " "),
        ]

        if prefer_lines:
            unit_builders[0], unit_builders[1] = unit_builders[1], unit_builders[0]

        for units, separator in unit_builders:
            if len(units) > 1:
                return self._pack_units(
                    units,
                    max_chars=max_chars,
                    max_tokens=max_tokens,
                    separator=separator,
                    prefer_lines=(separator == "\n"),
                )

        return textwrap.wrap(
            cleaned_text,
            width=max(1, min(max_chars, self._fallback_wrap_width(max_tokens))),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [cleaned_text]

    def _pack_units(
        self,
        units: Sequence[str],
        *,
        max_chars: int,
        max_tokens: int | None,
        separator: str,
        prefer_lines: bool,
    ) -> List[str]:
        chunks: List[str] = []
        current_parts: List[str] = []

        for unit in units:
            if not self._fits_within_budget(unit, max_chars=max_chars, max_tokens=max_tokens):
                if current_parts:
                    chunks.append(separator.join(current_parts))
                    current_parts = []

                chunks.extend(
                    self._split_text(
                        unit,
                        max_chars=max_chars,
                        max_tokens=max_tokens,
                        prefer_lines=prefer_lines,
                    )
                )
                continue

            candidate = separator.join(current_parts + [unit]) if current_parts else unit
            if current_parts and not self._fits_within_budget(
                candidate,
                max_chars=max_chars,
                max_tokens=max_tokens,
            ):
                chunks.append(separator.join(current_parts))
                current_parts = [unit]
            else:
                current_parts.append(unit)

        if current_parts:
            chunks.append(separator.join(current_parts))

        return chunks

    def _fits_within_budget(
        self,
        text: str,
        *,
        max_chars: int | None = None,
        max_tokens: int | None = None,
    ) -> bool:
        char_budget = self.max_chars if max_chars is None else max_chars
        token_budget = self.max_tokens if max_tokens is None else max_tokens

        if char_budget is not None and len(text) > char_budget:
            return False
        if token_budget is not None and self._token_count(text) > token_budget:
            return False

        return True

    def _token_count(self, text: str) -> int:
        if not text:
            return 0

        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))

        token_like_parts = re.findall(r"\w+|[^\w\s]", text)
        return max(len(token_like_parts), math.ceil(len(text) / 4))

    def _load_tokenizer(self, tokenizer_name: str) -> Any | None:
        if AutoTokenizer is None:
            return None

        try:
            return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        except Exception:
            return None

    def _is_short_standalone_section(self, chunk: Mapping[str, Any]) -> bool:
        if chunk.get("chunk_type") != "section":
            return False

        block_types = {str(block_type).lower() for block_type in chunk.get("block_types", [])}
        if not block_types or not block_types.issubset(self.heading_block_types):
            return False

        text = normalize_narrative_text(str(chunk.get("text") or ""))
        if not text:
            return False

        return len(text) < self.min_section_chars

    def _is_heading_block(self, block: NormalizedBlock) -> bool:
        return block.block_type in self.heading_block_types

    @staticmethod
    def _is_inferred_heading(block: NormalizedBlock) -> bool:
        text = normalize_narrative_text(block.text)
        if not text.endswith(":"):
            return False
        if block.block_type in {"table", "row", "page_text", "document_metadata", "workbook_metadata"}:
            return False
        return len(text) <= 120 and len(text.split()) <= 12

    @staticmethod
    def _should_prefix_short_section(chunk: Mapping[str, Any]) -> bool:
        text = normalize_narrative_text(str(chunk.get("text") or ""))
        lowered = text.lower().rstrip(":")
        return text.endswith(":") or lowered.startswith(("the following", "following ", "scope includes"))

    def _merge_prefix_chunks(self, prefixes: Sequence[ChunkDict], chunk: ChunkDict) -> ChunkDict:
        merged = dict(chunk)
        for prefix in reversed(prefixes):
            merged = self._merge_chunk_pair(prefix, merged, position="prefix")
        return merged

    def _merge_chunk_pair(self, left: Mapping[str, Any], right: Mapping[str, Any], *, position: str) -> ChunkDict:
        left_text = normalize_narrative_text(str(left.get("text") or ""))
        right_text = normalize_narrative_text(str(right.get("text") or ""))
        merged_text = join_narrative_parts([left_text, right_text])

        left_path = [str(part) for part in left.get("section_path", []) if str(part).strip()]
        right_path = [str(part) for part in right.get("section_path", []) if str(part).strip()]
        merged_path = self._ordered_unique([*left_path, *right_path]) if position == "prefix" else left_path
        heading = self._merged_heading(left, right, position=position)

        page_numbers = sorted(
            {
                *[int(page) for page in left.get("page_numbers", []) if page is not None],
                *[int(page) for page in right.get("page_numbers", []) if page is not None],
            }
        )
        metadata = {
            **dict(left.get("metadata") or {}),
            **dict(right.get("metadata") or {}),
            "source_block_count": len(self._ordered_unique([*left.get("block_ids", []), *right.get("block_ids", [])])),
            "merged_short_context_chunks": int(dict(left.get("metadata") or {}).get("merged_short_context_chunks", 0))
            + int(dict(right.get("metadata") or {}).get("merged_short_context_chunks", 0))
            + 1,
        }

        merged = dict(right if position == "prefix" else left)
        merged.update(
            {
                "section_path": merged_path,
                "section_key": " > ".join(merged_path),
                "section_title": self._section_title(heading, tuple(merged_path)),
                "heading": heading,
                "text": merged_text,
                "char_count": len(merged_text),
                "block_ids": self._ordered_unique([*left.get("block_ids", []), *right.get("block_ids", [])]),
                "block_types": self._ordered_unique([*left.get("block_types", []), *right.get("block_types", [])]),
                "order_start": min(int(left.get("order_start", 0)), int(right.get("order_start", 0))),
                "order_end": max(int(left.get("order_end", 0)), int(right.get("order_end", 0))),
                "page_number": self._first_or_none(page_numbers),
                "page_numbers": page_numbers,
                "slide_numbers": sorted(
                    {
                        *[int(slide) for slide in left.get("slide_numbers", []) if slide is not None],
                        *[int(slide) for slide in right.get("slide_numbers", []) if slide is not None],
                    }
                ),
                "shape_indices": sorted(
                    {
                        *[int(index) for index in left.get("shape_indices", []) if index is not None],
                        *[int(index) for index in right.get("shape_indices", []) if index is not None],
                    }
                ),
                "sheet_names": self._ordered_unique([*left.get("sheet_names", []), *right.get("sheet_names", [])]),
                "row_numbers": sorted(
                    {
                        *[int(row) for row in left.get("row_numbers", []) if row is not None],
                        *[int(row) for row in right.get("row_numbers", []) if row is not None],
                    }
                ),
                "metadata": metadata,
            }
        )
        return merged

    @staticmethod
    def _merged_heading(left: Mapping[str, Any], right: Mapping[str, Any], *, position: str) -> str | None:
        if position == "prefix":
            return str(left.get("section_title") or left.get("heading") or "").strip() or str(right.get("heading") or "").strip() or None
        return str(left.get("heading") or left.get("section_title") or "").strip() or str(right.get("heading") or "").strip() or None

    @staticmethod
    def _merged_text(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
        return join_narrative_parts(
            [
                normalize_narrative_text(str(left.get("text") or "")),
                normalize_narrative_text(str(right.get("text") or "")),
            ]
        )

    def _renumber_chunks(self, chunks: Sequence[ChunkDict]) -> List[ChunkDict]:
        renumbered: List[ChunkDict] = []
        for index, chunk in enumerate(chunks, start=1):
            next_chunk = dict(chunk)
            source_file = str(next_chunk.get("source_file") or next_chunk.get("file_name") or "")
            next_chunk["chunk_id"] = f"{source_file}::chunk_{index:04d}"
            renumbered.append(next_chunk)
        return renumbered

    @staticmethod
    def _section_title(heading: str | None, section_path: Sequence[str]) -> str | None:
        title = normalize_narrative_text(heading or "")
        if not title and section_path:
            title = normalize_narrative_text(str(section_path[-1]))
        return title.rstrip(":") or None

    @staticmethod
    def _first_or_none(values: Sequence[int]) -> int | None:
        return values[0] if values else None

    @staticmethod
    def _extract_table_header_line(text: str) -> str | None:
        lines = [line.strip() for line in _clean_text(text).splitlines() if line.strip()]
        if len(lines) < 2:
            return None

        first_line = lines[0]
        lowered = first_line.lower()
        if lowered.startswith(("columns:", "headers:", "header:", "fields:")):
            return first_line

        return None

    @staticmethod
    def _fallback_wrap_width(max_tokens: int | None) -> int:
        if max_tokens is None:
            return 2000
        return max(20, max_tokens * 4)

    @staticmethod
    def _subtract_token_headroom(max_tokens: int | None, prefix_text: str) -> int | None:
        if max_tokens is None:
            return None

        estimated_prefix_tokens = max(
            len(re.findall(r"\w+|[^\w\s]", prefix_text)),
            math.ceil(len(prefix_text) / 4),
        )
        return max(max_tokens - estimated_prefix_tokens, 1)

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
    max_tokens: int | None = 380,
    overlap_tokens: int = 40,
) -> List[ChunkDict]:
    return StructuralChunker(
        max_chars=max_chars,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    ).chunk_file_blocks(blocks)


def chunk_consolidated_blocks(
    consolidated_blocks: ConsolidatedBlocks,
    max_chars: int = 2000,
    max_tokens: int | None = 380,
    overlap_tokens: int = 40,
) -> Dict[str, List[ChunkDict]]:
    return StructuralChunker(
        max_chars=max_chars,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    ).chunk_consolidated_blocks(consolidated_blocks)


def _clean_text(value: Any) -> str:
    return clean_text(value)


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
