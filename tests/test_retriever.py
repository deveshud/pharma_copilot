from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.retriever import LocalRetriever


class FakeEncoder:
    def encode_query_text(self, query: str) -> list[float]:
        if "scope" in query.lower():
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]


def make_payload() -> dict[str, object]:
    return {
        "model": {
            "name": "fake-model",
            "normalized_embeddings": True,
        },
        "records": [
            {
                "chunk_id": "chunk_1",
                "source_file": "proposal.docx",
                "heading": "Scope",
                "text": "Project scope and deliverables.",
                "embedding": [0.95, 0.05, 0.0],
            },
            {
                "chunk_id": "chunk_2",
                "source_file": "proposal.docx",
                "heading": "Timeline",
                "text": "Implementation timeline and milestones.",
                "embedding": [0.25, 0.75, 0.0],
            },
            {
                "chunk_id": "chunk_3",
                "source_file": "proposal.docx",
                "heading": "Risks",
                "text": "Risk register and mitigations.",
                "embedding": [0.6, 0.4, 0.0],
            },
        ],
    }


def test_retriever_returns_top_ranked_chunks_for_query() -> None:
    retriever = LocalRetriever(encoder=FakeEncoder())  # type: ignore[arg-type]

    results = retriever.retrieve("What is in scope?", make_payload(), top_k=2)

    assert results == [
        "Project scope and deliverables.",
        "Risk register and mitigations.",
    ]


def test_retriever_uses_cosine_when_embeddings_are_not_normalized() -> None:
    retriever = LocalRetriever(encoder=FakeEncoder())  # type: ignore[arg-type]
    payload = make_payload()
    payload["model"]["normalized_embeddings"] = False  # type: ignore[index]
    payload["records"][0]["embedding"] = [10.0, 0.0, 0.0]  # type: ignore[index]
    payload["records"][1]["embedding"] = [1.0, 1.0, 0.0]  # type: ignore[index]

    results = retriever.retrieve("What is in scope?", payload, top_k=2)

    assert results == [
        "Project scope and deliverables.",
        "Risk register and mitigations.",
    ]
