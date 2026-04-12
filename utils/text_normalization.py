from __future__ import annotations

import re


_TERMINAL_PUNCTUATION = ".!?:;)]}"
_SOFT_TERMINAL_PUNCTUATION = ".!?)\"']}"
_IMAGE_PLACEHOLDER = "[image]"


def clean_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_narrative_text(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def join_narrative_parts(parts: list[str], *, punctuate: bool = True) -> str:
    normalized_parts = _normalize_parts(parts)
    if not normalized_parts:
        return ""

    if not punctuate:
        return " ".join(normalized_parts).strip()

    rendered_parts: list[str] = []
    for index, part in enumerate(normalized_parts):
        is_last = index == len(normalized_parts) - 1
        rendered_parts.append(_with_sentence_ending(part, force=is_last))

    return " ".join(rendered_parts).strip()


def render_heading_with_body(heading: str | None, body_parts: list[str]) -> str:
    normalized_heading = normalize_narrative_text(heading or "")
    body_text = join_narrative_parts(body_parts)

    if not normalized_heading:
        return body_text
    if not body_text:
        return normalized_heading

    heading_prefix = normalized_heading.rstrip(":")
    if _body_starts_with_heading(body_text, heading_prefix):
        return body_text

    if _looks_like_heading_prefix(heading_prefix):
        return f"{heading_prefix}: {body_text}"

    return f"{normalized_heading}\n\n{body_text}"


def normalize_retrieved_text(text: str, heading: str | None = None) -> str:
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return ""

    normalized_heading = normalize_narrative_text(heading or "")
    parts = [part.strip() for part in re.split(r"\n{2,}", cleaned_text) if part.strip()]
    if len(parts) <= 1:
        return normalize_narrative_text(cleaned_text)

    word_split_parts = parts[1:] if normalized_heading and _same_text(parts[0], normalized_heading) else parts
    if _looks_like_word_split_parts(word_split_parts):
        if normalized_heading and _same_text(parts[0], normalized_heading):
            return render_heading_with_body(normalized_heading, [_join_word_split_parts(parts[1:])])

        first_part = normalize_narrative_text(parts[0])
        if _looks_like_heading_prefix(first_part):
            return render_heading_with_body(first_part, [_join_word_split_parts(parts[1:])])

        return _join_word_split_parts(parts)

    if normalized_heading and _same_text(parts[0], normalized_heading):
        return render_heading_with_body(normalized_heading, parts[1:])

    first_part = normalize_narrative_text(parts[0])
    rest = parts[1:]
    if _looks_like_heading_prefix(first_part):
        return render_heading_with_body(first_part, rest)

    return join_narrative_parts(parts)


def _with_sentence_ending(text: str, *, force: bool = False) -> str:
    text = text.strip()
    if not text or text == _IMAGE_PLACEHOLDER:
        return text
    if text[-1] in _TERMINAL_PUNCTUATION:
        return text
    if not force and text[-1] in _SOFT_TERMINAL_PUNCTUATION:
        return text
    return f"{text}."


def _looks_like_heading_prefix(text: str) -> bool:
    if not text or "\n" in text:
        return False
    if len(text) > 120:
        return False
    if text[-1:] in ".!?":
        return False

    word_count = len(text.split())
    return word_count <= 12


def _body_starts_with_heading(body_text: str, heading: str) -> bool:
    if not heading:
        return False

    normalized_body = normalize_narrative_text(body_text).lower()
    normalized_heading = normalize_narrative_text(heading).lower().rstrip(":")
    return normalized_body == normalized_heading or normalized_body.startswith(f"{normalized_heading}:")


def _same_text(left: str, right: str) -> bool:
    return normalize_narrative_text(left).lower().rstrip(":") == normalize_narrative_text(right).lower().rstrip(":")


def _looks_like_word_split_parts(parts: list[str]) -> bool:
    normalized_parts = _normalize_parts(parts)
    if len(normalized_parts) < 4:
        return False

    short_parts = sum(1 for part in normalized_parts if len(part.split()) <= 2)
    return short_parts / len(normalized_parts) >= 0.75


def _join_word_split_parts(parts: list[str]) -> str:
    text = " ".join(_normalize_parts(parts))
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _with_sentence_ending(text, force=True)


def _normalize_parts(parts: list[str]) -> list[str]:
    normalized_parts: list[str] = []
    for part in parts:
        normalized_part = normalize_narrative_text(part)
        if normalized_part:
            normalized_parts.append(normalized_part)
    return normalized_parts
