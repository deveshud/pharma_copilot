from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CitationContext:
    """A normalized retrieval result that can be sent to an answer generator."""

    index: int
    source: str
    section: str
    text: str
    score: float | None = None


class LocalLLMAdapter(Protocol):
    """Small contract for plugging in Ollama, llama.cpp, vLLM, or another local runtime."""

    def generate(self, prompt: str, *, context: list[CitationContext]) -> str:
        """Return a grounded answer for the supplied prompt and citation context."""

    def generate_stream(self, prompt: str, *, context: list[CitationContext]) -> Iterator[str]:
        """Yield a grounded answer incrementally."""


class LocalLLMNotConfigured(RuntimeError):
    """Raised when answer generation cannot reach a configured local model."""


@dataclass(frozen=True)
class OllamaConfig:
    model: str = "llama3.2:3b"
    host: str | None = None
    temperature: float = 0.1
    num_ctx: int = 8192


ChatFunction = Callable[..., Any]


class OllamaLocalLLMAdapter:
    """Local answer generator backed by the Ollama Python package."""

    def __init__(self, config: OllamaConfig | None = None, chat_fn: ChatFunction | None = None) -> None:
        self.config = config or OllamaConfig()
        self._chat_fn = chat_fn

    def generate(self, prompt: str, *, context: list[CitationContext]) -> str:
        if not context:
            return (
                "I could not generate a grounded answer because no ChromaDB context "
                "was retrieved for this question."
            )

        chat_fn = self._chat_fn or self._load_chat_function()
        response = chat_fn(
            model=self.config.model,
            messages=self._messages(prompt),
            options=self._options(),
        )
        content = _response_content(response)
        if not content:
            raise LocalLLMNotConfigured("Ollama returned an empty response.")
        return content

    def generate_stream(self, prompt: str, *, context: list[CitationContext]) -> Iterator[str]:
        if not context:
            yield (
                "I could not generate a grounded answer because no ChromaDB context "
                "was retrieved for this question."
            )
            return

        chat_fn = self._chat_fn or self._load_chat_function()
        response_stream = chat_fn(
            model=self.config.model,
            messages=self._messages(prompt),
            options=self._options(),
            stream=True,
        )
        for chunk in response_stream:
            content = _response_content(chunk, strip=False)
            if content:
                yield content

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are Pharma Copilot, a local RAG assistant. Combine the "
                    "retrieved ChromaDB context with your language understanding to "
                    "synthesize a useful answer. All factual claims must come from "
                    "the provided context. Provide a concise answer, then brief "
                    "evidence reasoning with citations. Do not reveal private "
                    "chain-of-thought; summarize the evidence instead."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def _options(self) -> dict[str, float | int]:
        return {
            "temperature": self.config.temperature,
            "num_ctx": self.config.num_ctx,
        }

    def _load_chat_function(self) -> ChatFunction:
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover
            raise LocalLLMNotConfigured(
                "The `ollama` package is not installed in this environment. "
                "Install it with `uv add ollama` or `pip install ollama`."
            ) from exc

        if self.config.host:
            return ollama.Client(host=self.config.host).chat
        return ollama.chat


def build_citation_context(results: list[dict[str, Any]]) -> list[CitationContext]:
    context: list[CitationContext] = []
    for index, result in enumerate(results, start=1):
        context.append(
            CitationContext(
                index=index,
                source=str(result.get("file_name") or "Unknown source"),
                section=str(result.get("section_title") or "Untitled section"),
                text=str(result.get("text") or result.get("preview") or ""),
                score=_optional_float(result.get("score")),
            )
        )
    return context


def build_grounded_prompt(query: str, results: list[dict[str, Any]]) -> str:
    context_blocks = []
    for item in build_citation_context(results):
        context_blocks.append(
            f"[{item.index}] Source: {item.source}\n"
            f"Section: {item.section}\n"
            f"Content: {item.text}"
        )

    context = "\n\n".join(context_blocks) or "No retrieved context was provided."
    return (
        "You are a careful local RAG assistant for pharma and enterprise documents. "
        "Use the retrieved embedding context as factual evidence and use the local "
        "LLM only to synthesize, organize, and explain the answer. If the context is "
        "insufficient, say exactly what is missing. Do not invent details.\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Question:\n{query.strip()}\n\n"
        "Answer requirements:\n"
        "- Start with the direct answer.\n"
        "- Add a brief evidence reasoning section grounded in the retrieved chunks.\n"
        "- Cite supporting chunks with bracketed citations like [1] or [2].\n"
        "- Add a short 'Not found in context' note when evidence is missing."
    )


def local_llm_adapter(config: OllamaConfig | None = None) -> LocalLLMAdapter:
    return OllamaLocalLLMAdapter(config=config)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _response_content(response: Any, *, strip: bool = True) -> str:
    if isinstance(response, Mapping):
        message = response.get("message")
        if isinstance(message, Mapping):
            return _maybe_strip(str(message.get("content") or ""), strip)
        return _maybe_strip(str(response.get("content") or ""), strip)

    message = getattr(response, "message", None)
    if isinstance(message, Mapping):
        return _maybe_strip(str(message.get("content") or ""), strip)

    content = getattr(message, "content", None)
    if content is not None:
        return _maybe_strip(str(content), strip)

    return _maybe_strip(str(getattr(response, "content", "") or ""), strip)


def _maybe_strip(value: str, strip: bool) -> str:
    return value.strip() if strip else value
