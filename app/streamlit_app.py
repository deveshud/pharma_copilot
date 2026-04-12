from __future__ import annotations

import html
from pathlib import Path
import sys
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.answering import (  # noqa: E402
    LocalLLMNotConfigured,
    OllamaConfig,
    build_citation_context,
    build_grounded_prompt,
    local_llm_adapter,
)
from models.chroma_store import DEFAULT_CHROMA_COLLECTION  # noqa: E402
from models.retrieval_encoder import DEFAULT_RETRIEVAL_MODEL  # noqa: E402
from models.retriever import LocalRetriever  # noqa: E402


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_PERSIST_PATH = OUTPUTS_DIR / "chroma_db"
DEFAULT_SEED_CHUNKS = 8
DEFAULT_CANDIDATE_CHUNKS = 80
DEFAULT_ASSOCIATED_WINDOW = 2
DEFAULT_MAX_CONTEXT_CHUNKS = 24


st.set_page_config(
    page_title="Pharma Copilot",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --pc-ink: #1f2933;
            --pc-muted: #5f6b76;
            --pc-line: #d8e0df;
            --pc-surface: #ffffff;
            --pc-band: #f5f8f7;
            --pc-teal: #007f6d;
            --pc-teal-soft: #e4f3f0;
            --pc-warn: #fff2d2;
        }
        .block-container {
            padding-top: 3.6rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        h1, h2, h3 {
            color: var(--pc-ink);
        }
        .pc-kicker {
            color: var(--pc-teal);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .pc-title {
            font-size: 2.3rem;
            font-weight: 800;
            line-height: 1.08;
            margin-bottom: 0.35rem;
        }
        .pc-subtitle {
            color: var(--pc-muted);
            max-width: 850px;
            font-size: 1rem;
            line-height: 1.55;
            margin-bottom: 1.2rem;
        }
        .pc-helper {
            color: var(--pc-muted);
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0.35rem 0 0.7rem;
        }
        .pc-result {
            background: var(--pc-surface);
            border: 1px solid var(--pc-line);
            border-left: 5px solid var(--pc-teal);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 0.85rem 0;
        }
        .pc-result-title {
            color: var(--pc-ink);
            font-size: 1.02rem;
            font-weight: 750;
        }
        .pc-meta {
            color: var(--pc-muted);
            font-size: 0.86rem;
            margin-top: 0.25rem;
        }
        .pc-preview {
            color: var(--pc-ink);
            line-height: 1.55;
            margin-top: 0.75rem;
        }
        .pc-chip {
            display: inline-block;
            background: var(--pc-teal-soft);
            border: 1px solid #b7deda;
            border-radius: 8px;
            color: #0a6459;
            font-size: 0.76rem;
            padding: 0.15rem 0.45rem;
            margin: 0.15rem 0.2rem 0.1rem 0;
        }
        .pc-chip-warn {
            background: var(--pc-warn);
            border-color: #e7c56b;
            color: #6f4b00;
        }
        div[data-testid="stForm"] {
            background: var(--pc-band);
            border: 1px solid var(--pc-line);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 0.75rem 0 1.2rem;
        }
        div[data-testid="stForm"] textarea {
            border-radius: 8px;
        }
        .stButton > button {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_collection(persist_path: str, collection_name: str) -> Any:
    retriever = LocalRetriever()
    return retriever.open_chroma_collection(
        persist_path=persist_path,
        collection_name=collection_name,
    )


@st.cache_resource(show_spinner=False)
def get_retriever() -> LocalRetriever:
    return LocalRetriever()


def infer_model_name(collection: Any) -> str:
    return LocalRetriever.infer_chroma_model_name(collection, DEFAULT_RETRIEVAL_MODEL)


def run_retrieval(
    *,
    query: str,
    persist_path: str,
    collection_name: str,
) -> tuple[list[dict[str, Any]], str]:
    persist_dir = Path(persist_path).expanduser()
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {persist_dir.resolve()}. "
            "Run `uv run python runners/run_chroma_store.py --reset` first."
        )

    collection = get_collection(persist_path, collection_name)
    resolved_model = infer_model_name(collection)
    retriever = get_retriever()
    results = retriever.retrieve_associated_debug_from_chroma(
        query,
        collection,
        model_name=resolved_model,
        seed_k=DEFAULT_SEED_CHUNKS,
        candidate_k=DEFAULT_CANDIDATE_CHUNKS,
        associated_window=DEFAULT_ASSOCIATED_WINDOW,
        max_context_chunks=DEFAULT_MAX_CONTEXT_CHUNKS,
    )
    return results, resolved_model


def run_answer_generation(
    *,
    query: str,
    results: list[dict[str, Any]],
    ollama_model: str,
    ollama_host: str | None,
    temperature: float,
    num_ctx: int,
) -> str:
    prompt = build_grounded_prompt(query, results)
    context = build_citation_context(results)
    adapter = local_llm_adapter(
        OllamaConfig(
            model=ollama_model,
            host=ollama_host,
            temperature=temperature,
            num_ctx=num_ctx,
        )
    )
    return adapter.generate(prompt, context=context)


def sidebar_settings() -> dict[str, Any]:
    with st.sidebar:
        st.header("Knowledge Base")
        persist_path = st.text_input("Chroma path", value=str(DEFAULT_PERSIST_PATH))
        collection_name = st.text_input("Collection", value=DEFAULT_CHROMA_COLLECTION)
        show_debug = st.toggle("Show retrieval evidence details", value=False)

        st.divider()
        st.header("Answer Generation")
        ollama_model = st.text_input(
            "Ollama model",
            value="llama3.2:3b",
            help="Use a locally available Ollama model, for example llama3.2:3b.",
        )
        ollama_host = st.text_input(
            "Ollama host",
            value="",
            placeholder="Optional, e.g. http://localhost:11434",
        )
        temperature = st.slider(
            "Answer temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
        )
        num_ctx = st.select_slider(
            "Context window",
            options=[2048, 4096, 8192, 16384],
            value=8192,
        )

    return {
        "persist_path": persist_path,
        "collection_name": collection_name,
        "show_debug": show_debug,
        "ollama_model": ollama_model.strip() or "llama3.2:3b",
        "ollama_host": ollama_host.strip() or None,
        "temperature": temperature,
        "num_ctx": num_ctx,
    }


def render_result(index: int, result: dict[str, Any], show_debug: bool) -> None:
    title = html.escape(str(result.get("section_title") or "Untitled section"))
    file_name = html.escape(str(result.get("file_name") or "Unknown file"))
    score = float(result.get("score", 0.0))
    chunk_length = result.get("chunk_length", 0)
    page_number = result.get("page_number")
    page_label = html.escape(f"Page {page_number}" if page_number else "Page unavailable")
    preview = html.escape(str(result.get("preview", "")))
    relationship = str(result.get("relationship") or "semantic_seed")
    relationship_label = "Best match" if relationship == "semantic_seed" else "Associated context"

    st.markdown(
        f"""
        <div class="pc-result">
            <div class="pc-result-title">{index}. {title}</div>
            <div class="pc-meta">{file_name} | {page_label} | {html.escape(relationship_label)} | Score {score:.3f} | {chunk_length} chars</div>
            <div class="pc-preview">{preview}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    reasons = result.get("reasons") or []
    if reasons:
        chips = []
        for reason in reasons:
            chip_class = "pc-chip-warn" if str(reason).startswith("penalty") else ""
            chips.append(f'<span class="pc-chip {chip_class}">{html.escape(str(reason))}</span>')
        st.markdown("".join(chips), unsafe_allow_html=True)

    if show_debug:
        st.json(
            {
                "metadata": result.get("metadata", {}),
                "final_score": result.get("final_score"),
                "raw_vector_score": result.get("raw_vector_score"),
                "rerank_delta": result.get("rerank_delta"),
                "chroma_distance": result.get("chroma_distance"),
                "reasons": result.get("reasons", []),
            }
        )


def render_generation(
    *,
    query: str,
    results: list[dict[str, Any]],
    settings: dict[str, Any],
) -> bool:
    status = st.empty()
    try:
        with st.spinner("Generating answer..."):
            status.info("Preparing the grounded prompt from retrieved context...")
            prompt = build_grounded_prompt(query, results)
            context = build_citation_context(results)

            status.info(f"Sending {len(context)} retrieved context chunks to Ollama...")
            adapter = local_llm_adapter(
                OllamaConfig(
                    model=settings["ollama_model"],
                    host=settings["ollama_host"],
                    temperature=settings["temperature"],
                    num_ctx=settings["num_ctx"],
                )
            )

            status.info("Ollama is reading the context and drafting the answer...")
            st.session_state.last_answer = adapter.generate(prompt, context=context)

            status.info("Finalizing the answer with citations...")
        status.empty()
        st.markdown(st.session_state.last_answer)
        st.session_state.answer_error = None
        return True
    except LocalLLMNotConfigured as exc:
        st.session_state.answer_error = str(exc)
    except Exception as exc:
        st.session_state.answer_error = (
            f"Ollama generation failed: {exc}. Confirm Ollama is running and "
            f"the model `{settings['ollama_model']}` is pulled locally."
        )
    return False


def main() -> None:
    apply_theme()
    settings = sidebar_settings()

    st.markdown('<div class="pc-kicker">Local-first RAG workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="pc-title">Pharma Copilot Retrieval Console</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pc-subtitle">Ask a question, retrieve grounded evidence from ChromaDB, '
        'then let Ollama synthesize a cited answer.</div>',
        unsafe_allow_html=True,
    )

    if "query_text" not in st.session_state:
        st.session_state.query_text = ""

    st.markdown(
        '<div class="pc-helper">Ask one focused question. The app retrieves matching context and writes one grounded answer.</div>',
        unsafe_allow_html=True,
    )
    with st.form("question_form"):
        st.text_area(
            "Question",
            key="query_text",
            height=96,
            placeholder="Ask about scope, assumptions, KPI logic, ownership, timelines, or out-of-scope items.",
        )
        submitted = st.form_submit_button("Answer question", type="primary", use_container_width=True)

    if not submitted and "last_results" not in st.session_state:
        return

    active_query = st.session_state.query_text.strip()
    if submitted and not active_query:
        st.warning("Enter a question before asking for an answer.")
        return

    if submitted:
        try:
            with st.spinner("Searching ChromaDB and gathering context..."):
                results, resolved_model = run_retrieval(
                    query=active_query,
                    persist_path=settings["persist_path"],
                    collection_name=settings["collection_name"],
                )
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.info("Confirm that the Chroma database exists. You can build it with `uv run python runners/run_chroma_store.py --reset`.")
            return

        st.session_state.last_results = results
        st.session_state.last_query = active_query
        st.session_state.resolved_model = resolved_model
        st.session_state.last_answer = None
        st.session_state.answer_error = None

    results = st.session_state.get("last_results", [])
    last_query = st.session_state.get("last_query", active_query)
    resolved_model = st.session_state.get("resolved_model", DEFAULT_RETRIEVAL_MODEL)

    st.subheader("Answer")
    generated_this_run = bool(results) and submitted and render_generation(
        query=last_query,
        results=results,
        settings=settings,
    )

    if st.session_state.get("answer_error"):
        st.error(st.session_state.answer_error)
    elif st.session_state.get("last_answer") and not generated_this_run:
        st.markdown(st.session_state.last_answer)

    if results:
        with st.expander("Retrieved evidence", expanded=False):
            st.caption(
                f"{len(results)} context chunks retrieved with embedding model "
                f"`{resolved_model.split('/')[-1]}`."
            )
            for index, result in enumerate(results, start=1):
                render_result(index, result, settings["show_debug"])

        with st.expander("Grounded prompt", expanded=False):
            st.text_area(
                "Prompt sent to Ollama",
                value=build_grounded_prompt(last_query, results),
                height=420,
            )


if __name__ == "__main__":
    main()
