from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.answering import (
    OllamaConfig,
    OllamaLocalLLMAdapter,
    build_citation_context,
    build_grounded_prompt,
    local_llm_adapter,
)


def test_build_grounded_prompt_includes_cited_retrieval_context() -> None:
    results = [
        {
            "file_name": "proposal.docx",
            "section_title": "Project Scope",
            "text": "Requirements gathering and KPI mapping are in scope.",
            "score": 1.23,
        }
    ]

    prompt = build_grounded_prompt("What is in scope?", results)

    assert "What is in scope?" in prompt
    assert "[1] Source: proposal.docx" in prompt
    assert "Section: Project Scope" in prompt
    assert "Requirements gathering and KPI mapping are in scope." in prompt
    assert "Do not invent details." in prompt
    assert "evidence reasoning" in prompt
    assert "retrieved embedding context as factual evidence" in prompt
    assert "LLM only to synthesize" in prompt
    assert prompt.index("Retrieved context:") < prompt.index("Question:")


def test_build_citation_context_normalizes_missing_fields() -> None:
    context = build_citation_context([{"preview": "Fallback preview"}])

    assert context[0].index == 1
    assert context[0].source == "Unknown source"
    assert context[0].section == "Untitled section"
    assert context[0].text == "Fallback preview"
    assert context[0].score is None


def test_ollama_adapter_sends_grounded_prompt_to_chat_function() -> None:
    calls = {}

    def fake_chat(**kwargs: object) -> dict[str, dict[str, str]]:
        calls.update(kwargs)
        return {"message": {"content": "Answer grounded in context [1]."}}

    adapter = OllamaLocalLLMAdapter(
        config=OllamaConfig(model="test-model", temperature=0.2, num_ctx=4096),
        chat_fn=fake_chat,
    )
    context = build_citation_context(
        [
            {
                "file_name": "proposal.docx",
                "section_title": "Scope",
                "text": "Requirements mapping is in scope.",
            }
        ]
    )

    answer = adapter.generate("Grounded prompt", context=context)

    assert answer == "Answer grounded in context [1]."
    assert calls["model"] == "test-model"
    assert calls["options"] == {"temperature": 0.2, "num_ctx": 4096}
    assert calls["messages"][0]["role"] == "system"  # type: ignore[index]
    assert calls["messages"][1] == {"role": "user", "content": "Grounded prompt"}


def test_ollama_adapter_streams_response_chunks() -> None:
    calls = {}

    def fake_chat(**kwargs: object) -> list[dict[str, dict[str, str]]]:
        calls.update(kwargs)
        return [
            {"message": {"content": "Answer "}},
            {"message": {"content": "grounded in context [1]."}},
        ]

    adapter = OllamaLocalLLMAdapter(
        config=OllamaConfig(model="test-model"),
        chat_fn=fake_chat,
    )
    context = build_citation_context(
        [
            {
                "file_name": "proposal.docx",
                "section_title": "Scope",
                "text": "Requirements mapping is in scope.",
            }
        ]
    )

    chunks = list(adapter.generate_stream("Grounded prompt", context=context))

    assert chunks == ["Answer ", "grounded in context [1]."]
    assert calls["stream"] is True
    assert calls["model"] == "test-model"


def test_local_llm_adapter_defaults_to_ollama_adapter() -> None:
    assert isinstance(local_llm_adapter(), OllamaLocalLLMAdapter)
