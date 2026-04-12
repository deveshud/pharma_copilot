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


class FakeChromaCollection:
    def __init__(self) -> None:
        self.last_query: dict[str, object] | None = None

    def query(self, **kwargs: object) -> dict[str, object]:
        self.last_query = kwargs
        return {
            "ids": [["chunk_admin", "chunk_scope"]],
            "documents": [
                [
                    "Source File: proposal.docx\nContent:\nINDIVIDUAL PROJECT AGREEMENT under Master Services Agreement.",
                    (
                        "Source File: proposal.docx\nSection Title: Project Scope\nContent:\n"
                        "Project Scope: Requirements gathering and mapping are included."
                    ),
                ]
            ],
            "metadatas": [
                [
                    {
                        "chunk_id": "chunk_admin",
                        "file_name": "proposal.docx",
                        "source_file": "proposal.docx",
                        "section_title": "INDIVIDUAL PROJECT AGREEMENT",
                        "heading": "INDIVIDUAL PROJECT AGREEMENT",
                        "block_ids": '["a1"]',
                        "block_types": '["heading"]',
                        "embedding_model": "fake-model",
                    },
                    {
                        "chunk_id": "chunk_scope",
                        "file_name": "proposal.docx",
                        "source_file": "proposal.docx",
                        "section_title": "Project Scope",
                        "heading": "Project Scope:",
                        "block_ids": '["s1", "s2"]',
                        "block_types": '["heading", "paragraph"]',
                        "embedding_model": "fake-model",
                    },
                ]
            ],
            "distances": [[0.01, 0.08]],
        }

    def get(self, limit: int, include: list[str]) -> dict[str, object]:
        return {"metadatas": [{"embedding_model": "fake-model"}]}


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


def test_retriever_normalizes_legacy_word_split_result_text() -> None:
    payload = make_payload()
    payload["records"][0]["heading"] = "Out of scope"  # type: ignore[index]
    payload["records"][0]["text"] = (  # type: ignore[index]
        "Out of scope\n\nAny\n\nnew\n\ndata\n\ningestion,\n\ndata\n\ntransformations,\n\nor\n\n"
        "data\n\nassets\n\n(CADs/RRDs)\n\noutside\n\nthe\n\nscope."
    )
    retriever = LocalRetriever(encoder=FakeEncoder())  # type: ignore[arg-type]

    results = retriever.retrieve("What is in scope?", payload, top_k=1)

    assert results == [
        "Out of scope: Any new data ingestion, data transformations, or data assets (CADs/RRDs) outside the scope."
    ]


def test_retriever_boosts_section_context_and_penalizes_tiny_chunks() -> None:
    payload = make_payload()
    payload["records"] = [
        {
            "chunk_id": "chunk_tiny",
            "source_file": "PSS AR KPI Enablement through ICI Snowflake.docx",
            "section_title": "Project Scope",
            "heading": "Project Scope:",
            "text": "Project Scope:",
            "metadata": {"source_block_count": 1},
            "embedding": [0.99, 0.01, 0.0],
        },
        {
            "chunk_id": "chunk_scope",
            "source_file": "PSS AR KPI Enablement through ICI Snowflake.docx",
            "section_title": "Project Scope",
            "heading": "Project Scope:",
            "text": (
                "Project Scope: The following activities are in scope for this engagement. "
                "Requirements Gathering and Onboarding: Structured onboarding to the Sanofi analytics ecosystem."
            ),
            "metadata": {"source_block_count": 4},
            "embedding": [0.95, 0.05, 0.0],
        },
    ]
    retriever = LocalRetriever(encoder=FakeEncoder())  # type: ignore[arg-type]

    results = retriever.retrieve_debug("PSS A&R KPI Enablement Scope", payload, top_k=2)

    assert results[0]["chunk_id"] == "chunk_scope"
    assert results[0]["section_title"] == "Project Scope"
    assert results[0]["chunk_length"] > 80
    assert "The following activities are in scope" in results[0]["preview"]


def test_retriever_demotes_boilerplate_for_scope_queries() -> None:
    payload = make_payload()
    payload["records"] = [
        {
            "chunk_id": "chunk_admin",
            "source_file": "PSS AR KPI Enablement through ICI Snowflake.docx",
            "section_title": "INDIVIDUAL PROJECT AGREEMENT (IPA)",
            "heading": "INDIVIDUAL PROJECT AGREEMENT (IPA)",
            "text": (
                "INDIVIDUAL PROJECT AGREEMENT (IPA) under Master Services Agreement. "
                "Customer Project Contact: Jane Doe. Vendor Project Contact: John Smith. "
                "The following terms and conditions apply solely to work performed under this IPA."
            ),
            "metadata": {"source_block_count": 7},
            "embedding": [0.99, 0.01, 0.0],
        },
        {
            "chunk_id": "chunk_scope",
            "source_file": "PSS AR KPI Enablement through ICI Snowflake.docx",
            "section_title": "Project Scope",
            "heading": "Project Scope:",
            "text": (
                "Project Scope: The following activities are in scope for this engagement. "
                "Requirements Gathering and Onboarding: Structured onboarding to the Sanofi analytics ecosystem."
            ),
            "metadata": {"source_block_count": 4},
            "embedding": [0.93, 0.07, 0.0],
        },
    ]
    retriever = LocalRetriever(encoder=FakeEncoder())  # type: ignore[arg-type]

    results = retriever.retrieve_debug("PSS A&R KPI Enablement Scope", payload, top_k=2)

    assert results[0]["chunk_id"] == "chunk_scope"
    assert any(reason.startswith("boost:scope_section") for reason in results[0]["reasons"])
    assert results[1]["chunk_id"] == "chunk_admin"
    assert any(reason.startswith("penalty:boilerplate") for reason in results[1]["reasons"])
    assert results[1]["raw_vector_score"] > results[0]["raw_vector_score"]
    assert results[1]["final_score"] < results[0]["final_score"]


def test_retriever_prioritizes_project_scope_for_generic_scope_query() -> None:
    payload = make_payload()
    payload["records"] = [
        {
            "chunk_id": "chunk_out_scope",
            "source_file": "PSS AR KPI Enablement through ICI Snowflake.docx",
            "section_title": "Out of scope",
            "heading": "Out of scope",
            "text": "Out of scope: New data ingestion and new dashboard development are outside scope.",
            "metadata": {"source_block_count": 3},
            "embedding": [0.99, 0.01, 0.0],
        },
        {
            "chunk_id": "chunk_project_scope",
            "source_file": "PSS AR KPI Enablement through ICI Snowflake.docx",
            "section_title": "Project Scope",
            "heading": "Project Scope:",
            "text": (
                "Project Scope: The following activities are in scope for this engagement. "
                "Requirements gathering and mapping are included."
            ),
            "metadata": {"source_block_count": 4},
            "embedding": [0.93, 0.07, 0.0],
        },
    ]
    retriever = LocalRetriever(encoder=FakeEncoder())  # type: ignore[arg-type]

    results = retriever.retrieve_debug("PSS A&R KPI Enablement Scope", payload, top_k=2)

    assert results[0]["chunk_id"] == "chunk_project_scope"
    assert any("boost:scope_section(+0.46)" in reason for reason in results[0]["reasons"])
    assert any("boost:scope_section(+0.24)" in reason for reason in results[1]["reasons"])


def test_retriever_prioritizes_out_of_scope_when_query_says_out() -> None:
    payload = make_payload()
    payload["records"] = [
        {
            "chunk_id": "chunk_out_scope",
            "source_file": "proposal.docx",
            "section_title": "Out of scope",
            "heading": "Out of scope",
            "text": "Out of scope: New data ingestion and dashboard development are excluded.",
            "metadata": {"source_block_count": 3},
            "embedding": [0.95, 0.05, 0.0],
        },
        {
            "chunk_id": "chunk_project_scope",
            "source_file": "proposal.docx",
            "section_title": "Project Scope",
            "heading": "Project Scope:",
            "text": "Project Scope: Requirements gathering and mapping are included.",
            "metadata": {"source_block_count": 3},
            "embedding": [0.95, 0.05, 0.0],
        },
    ]
    retriever = LocalRetriever(encoder=FakeEncoder())  # type: ignore[arg-type]

    results = retriever.retrieve_debug("What is out of scope?", payload, top_k=2)

    assert results[0]["chunk_id"] == "chunk_out_scope"
    assert any("boost:scope_section(+0.34)" in reason for reason in results[0]["reasons"])


def test_retriever_keeps_boilerplate_available_for_admin_queries() -> None:
    payload = make_payload()
    payload["records"] = [
        {
            "chunk_id": "chunk_admin",
            "source_file": "proposal.docx",
            "section_title": "INDIVIDUAL PROJECT AGREEMENT (IPA)",
            "heading": "INDIVIDUAL PROJECT AGREEMENT (IPA)",
            "text": "INDIVIDUAL PROJECT AGREEMENT (IPA) under Master Services Agreement. Customer Project Contact: Jane Doe.",
            "metadata": {"source_block_count": 4},
            "embedding": [0.0, 0.98, 0.0],
        },
        {
            "chunk_id": "chunk_scope",
            "source_file": "proposal.docx",
            "section_title": "Project Scope",
            "heading": "Project Scope:",
            "text": "Project Scope: Requirements mapping and validation are in scope.",
            "metadata": {"source_block_count": 3},
            "embedding": [0.0, 0.9, 0.0],
        },
    ]
    retriever = LocalRetriever(encoder=FakeEncoder())  # type: ignore[arg-type]

    results = retriever.retrieve_debug("Who is the customer project contact?", payload, top_k=2)

    assert results[0]["chunk_id"] == "chunk_admin"
    assert not any(reason.startswith("penalty:boilerplate") for reason in results[0]["reasons"])


def test_retriever_queries_chroma_and_reranks_candidates() -> None:
    collection = FakeChromaCollection()
    retriever = LocalRetriever(encoder=FakeEncoder())  # type: ignore[arg-type]

    results = retriever.retrieve_debug_from_chroma(
        "PSS A&R KPI Enablement Scope",
        collection,
        model_name="fake-model",
        top_k=1,
        candidate_k=2,
    )

    assert collection.last_query is not None
    assert collection.last_query["n_results"] == 2
    assert collection.last_query["query_embeddings"] == [[1.0, 0.0, 0.0]]
    assert results[0]["chunk_id"] == "chunk_scope"
    assert results[0]["raw_vector_score"] < 1.0
    assert results[0]["metadata"]["embedding_model"] == "fake-model"
    assert any(reason.startswith("boost:scope_section") for reason in results[0]["reasons"])


def test_retriever_infers_chroma_model_name_from_metadata() -> None:
    collection = FakeChromaCollection()

    assert LocalRetriever.infer_chroma_model_name(collection, "fallback-model") == "fake-model"
