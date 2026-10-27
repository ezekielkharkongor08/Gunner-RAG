"""Streamlit UI for Arsenal FC Wiki RAG.

Connected to FastAPI backend with dedicated endpoints for:
- Chat (multi-turn conversation with memory)
- Ask (single question with citation and optional grounding verification)
- Summarize (Wikipedia article deep summary)
- Ingest (Wikipedia page pipeline: fetch -> clean -> split -> embed -> pgvector)
- Diagnostics (PostgreSQL, Ollama, pgvector collection status)

Official Arsenal FC aesthetic (Arsenal Red & Navy) configured via .streamlit/config.toml.
Zero raw HTML code used -- 100% native Streamlit components.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import requests
import streamlit as st

# Configure page layout and title
st.set_page_config(
    page_title="Arsenal FC — Gunner Wiki RAG",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# API Helpers
# ---------------------------------------------------------------------------
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def check_api_health(base_url: str) -> Optional[Dict[str, Any]]:
    """Check if FastAPI backend is healthy and return status data."""
    try:
        resp = requests.get(f"{base_url}/api/status", timeout=6)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_available_articles(base_url: str) -> List[str]:
    """Fetch pre-configured articles from FastAPI backend."""
    try:
        resp = requests.get(f"{base_url}/api/articles", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("articles", [])
    except Exception:
        pass
    return []


def api_ask(base_url: str, question: str, check_grounding: bool) -> Dict[str, Any]:
    """Call /api/ask endpoint."""
    resp = requests.post(
        f"{base_url}/api/ask",
        json={"question": question, "check_grounding": check_grounding},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def api_chat(base_url: str, question: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Call /api/chat endpoint."""
    resp = requests.post(
        f"{base_url}/api/chat",
        json={"question": question, "history": history},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def api_summarize(base_url: str, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Call /api/summarize endpoint to summarize retrieved docs for query."""
    resp = requests.post(
        f"{base_url}/api/summarize",
        json={"query": query, "history": history or []},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def api_ingest(base_url: str, title: str, reset: bool) -> Dict[str, Any]:
    """Call /api/ingest endpoint."""
    resp = requests.post(
        f"{base_url}/api/ingest",
        json={"title": title, "reset": reset},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------
st.sidebar.title("🔴 Arsenal FC")
st.sidebar.caption("Gunner Wiki RAG Intelligence Hub")
st.sidebar.divider()

# Backend Connection Setting
api_url = st.sidebar.text_input("FastAPI Backend URL", value=DEFAULT_API_URL).rstrip("/")

# Check API status
api_status = check_api_health(api_url)

if api_status:
    st.sidebar.success("Backend API: Connected (Online)")
    st.sidebar.metric(
        label="Configured Wikipedia Pages",
        value=api_status.get("configured_articles_count", 0),
    )
else:
    st.sidebar.error("Backend API: Offline")
    st.sidebar.info(
        "Run the FastAPI server inside your venv:\n\n"
        "`.venv\\Scripts\\activate`\n\n"
        "`uvicorn api:app --host 127.0.0.1 --port 8000 --reload`"
    )

st.sidebar.divider()

# Navigation
page = st.sidebar.radio(
    "Navigation Mode",
    [
        "💬 Gunner Chat",
        "🎯 Ask Direct",
        "📥 Wikipedia Ingestion Hub",
        "⚙️ System Diagnostics",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.info(
    "Powered by LangChain, PostgreSQL pgvector, and Ollama local models."
)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Page 1: Gunner Chat
# ---------------------------------------------------------------------------
if page == "💬 Gunner Chat":
    st.title("🔴 Gunner Chat")
    st.caption("Multi-turn conversational assistant with context memory over Arsenal Wikipedia records")

    # Mode Selector Dropdown
    chat_mode = st.selectbox(
        "Response Mode",
        [
            "💬 Standard Q&A",
            "📜 Summarize Retrieved Documents for Query",
        ],
        index=0,
        help="Select whether to receive a conversational Q&A answer or synthesize a grounded summary of all retrieved document passages for your query.",
    )
    if chat_mode == "📜 Summarize Retrieved Documents for Query":
        st.info("💡 **Summarize Mode Active**: Your query will retrieve all matching Wikipedia document passages from pgvector and synthesize an executive summary with citations.")

    # Session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # List of dicts: {"role": "user"|"assistant", "content": str, "sources": list}

    # Suggested Prompts
    st.write("**Quick Prompts:**")
    prompt_col1, prompt_col2, prompt_col3, prompt_col4 = st.columns(4)

    selected_prompt = None
    with prompt_col1:
        if st.button("🏆 The 2003–04 Invincibles", use_container_width=True):
            selected_prompt = "Tell me about Arsenal's 2003-04 Invincibles season."
    with prompt_col2:
        if st.button("🏟️ Emirates Stadium Facts", use_container_width=True):
            selected_prompt = "When was Emirates Stadium opened and what is its capacity?"
    with prompt_col3:
        if st.button("⚽ Record Goalscorer", use_container_width=True):
            selected_prompt = "Who is Arsenal's all-time top goalscorer and how many goals did he score?"
    with prompt_col4:
        if st.button("⚔️ North London Derby", use_container_width=True):
            selected_prompt = "What is Arsenal's record in the North London derby against Tottenham?"

    # Display Conversation History
    for msg in st.session_state.chat_history:
        role = msg["role"]
        with st.chat_message(role, avatar="⚪" if role == "user" else "🔴"):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} Cited Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- **{s.get('title')}** | Section: *{s.get('section', 'Lead')}* | [Wikipedia Link]({s.get('url')})")
                        st.caption(f"Relevance Score: {s.get('score', 0):.4f}")

    # Action row (Clear chat)
    if st.session_state.chat_history:
        if st.button("Clear Conversation History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    # Chat Input
    user_query = st.chat_input("Ask a question or enter a topic to summarize about Arsenal F.C. ...")

    # If clicked quick prompt or typed question
    active_query = selected_prompt or user_query
    if active_query:
        if not api_status:
            st.error("Cannot send message: FastAPI backend is not reachable. Please start `uvicorn api:app`.")
        else:
            # Display user message
            with st.chat_message("user", avatar="⚪"):
                st.write(active_query)

            # Build prior history turns for backend
            backend_history: List[Dict[str, str]] = []
            for i in range(0, len(st.session_state.chat_history) - 1, 2):
                if i + 1 < len(st.session_state.chat_history):
                    backend_history.append({
                        "question": st.session_state.chat_history[i]["content"],
                        "answer": st.session_state.chat_history[i + 1]["content"],
                    })

            # Call API
            with st.chat_message("assistant", avatar="🔴"):
                is_summary_mode = (chat_mode == "📜 Summarize Retrieved Documents for Query")
                spinner_text = (
                    "Retrieving matching documents and synthesizing summary for query..."
                    if is_summary_mode
                    else "Searching Arsenal pgvector index and generating answer..."
                )
                with st.spinner(spinner_text):
                    try:
                        if is_summary_mode:
                            res = api_summarize(api_url, active_query, backend_history)
                            answer = res.get("summary", "No summary received.")
                        else:
                            res = api_chat(api_url, active_query, backend_history)
                            answer = res.get("answer", "No answer received.")
                        sources = res.get("sources", [])

                        st.write(answer)
                        if sources:
                            with st.expander(f"📚 {len(sources)} Cited Sources"):
                                for s in sources:
                                    st.markdown(f"- **{s.get('title')}** | Section: *{s.get('section', 'Lead')}* | [Wikipedia Link]({s.get('url')})")
                                    st.caption(f"Relevance Score: {s.get('score', 0):.4f}")

                        # Append to state
                        st.session_state.chat_history.append({"role": "user", "content": active_query})
                        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
                    except Exception as e:
                        st.error(f"Error communicating with backend: {e}")


# ---------------------------------------------------------------------------
# Page 2: Ask Direct
# ---------------------------------------------------------------------------
elif page == "🎯 Ask Direct":
    st.title("🎯 Ask Direct")
    st.caption("Ask targeted questions with strict fact verification against retrieved Wikipedia sources")

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g. Who did Arsenal lose to in the 2025-26 EFL Cup final?",
        )
    with col2:
        check_grounding = st.toggle("Verify Grounding", value=True, help="Check if numbers and entity names match the cited sources")

    if st.button("Submit Question", type="primary"):
        if not question.strip():
            st.warning("Please type a question before submitting.")
        elif not api_status:
            st.error("Backend is unreachable. Please start `uvicorn api:app`.")
        else:
            with st.spinner("Retrieving facts and computing answer..."):
                try:
                    res = api_ask(api_url, question.strip(), check_grounding=check_grounding)
                    answer = res.get("answer", "")
                    sources = res.get("sources", [])
                    grounding = res.get("grounding_report")

                    st.subheader("Answer")
                    st.info(answer)

                    # Grounding report
                    if check_grounding and grounding:
                        st.subheader("Grounding Verification")
                        unsupported_nums = grounding.get("numbers", [])
                        unsupported_names = grounding.get("names", [])

                        if not unsupported_nums and not unsupported_names:
                            st.success("All numbers and proper entities in the answer are grounded in retrieved sources.")
                        else:
                            if unsupported_nums:
                                st.warning(f"Numbers not directly matching sources: {', '.join(unsupported_nums)}")
                            if unsupported_names:
                                st.warning(f"Names/entities not directly matching sources: {', '.join(unsupported_names[:6])}")

                    # Sources
                    st.subheader("Retrieved Wikipedia Sources")
                    if sources:
                        for idx, s in enumerate(sources, 1):
                            with st.expander(f"[{idx}] {s.get('title')} — Section: {s.get('section', 'Lead')}"):
                                st.write(f"**URL:** {s.get('url')}")
                                st.write(f"**Similarity Score:** {s.get('score', 0):.4f}")
                                st.divider()
                                st.text(s.get("text", ""))
                    else:
                        st.write("No matching passages found in the indexed Wikipedia articles.")

                except Exception as e:
                    st.error(f"Error processing question: {e}")


# ---------------------------------------------------------------------------
# Page 4: Wikipedia Ingestion Hub
# ---------------------------------------------------------------------------
elif page == "📥 Wikipedia Ingestion Hub":
    st.title("📥 Wikipedia Ingestion Hub")
    st.caption("Fetch, clean, split, embed, and index Arsenal Wikipedia articles into PostgreSQL pgvector")

    st.write(
        "Ingestion executes the 4-stage pipeline:\n"
        "1. **Load**: MediaWiki Action API fetch (section-aware prose)\n"
        "2. **Clean & Assemble**: Strip boilerplate, tables, and merge fragments\n"
        "3. **Split**: Recursive text chunking with metadata headers\n"
        "4. **Embed & Store**: Qwen/Ollama embeddings stored into PostgreSQL pgvector"
    )

    ingest_col1, ingest_col2 = st.columns([3, 1])
    with ingest_col1:
        wiki_title = st.text_input(
            "Wikipedia Page Title to Ingest:",
            value="Arsenal F.C.",
            placeholder="e.g. Bukayo Saka, Mikel Arteta, Arsenal F.C.",
        )
    with ingest_col2:
        reset_collection = st.checkbox(
            "Reset Collection",
            value=False,
            help="WARNING: If checked, existing vector records in pgvector will be wiped before inserting new ones.",
        )

    if reset_collection:
        st.warning("Reset Collection is checked: All existing chunks in pgvector will be deleted before ingestion!")

    if st.button("Run Ingestion Pipeline", type="primary"):
        if not wiki_title.strip():
            st.warning("Please provide a valid Wikipedia article title.")
        elif not api_status:
            st.error("Backend is unreachable. Please start `uvicorn api:app`.")
        else:
            with st.status(f"Ingesting '{wiki_title}' via MediaWiki API...", expanded=True) as status:
                st.write("Connecting to FastAPI backend...")
                try:
                    res = api_ingest(api_url, wiki_title.strip(), reset=reset_collection)
                    summary = res.get("summary", {})
                    written = summary.get("written", 0)
                    chunks = summary.get("chunks", 0)

                    status.update(label=f"Ingestion Finished: '{wiki_title}' successfully indexed!", state="complete")
                    st.success(res.get("message", "Ingestion succeeded!"))

                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        st.metric("Total Chunks Created", chunks)
                    with stat_col2:
                        st.metric("Vector Embeddings Written", written)

                    with st.expander("Full Ingestion Report"):
                        st.json(summary)

                except Exception as e:
                    status.update(label="Ingestion Failed", state="error")
                    st.error(f"Error during ingestion: {e}")


# ---------------------------------------------------------------------------
# Page 5: System Diagnostics
# ---------------------------------------------------------------------------
elif page == "⚙️ System Diagnostics":
    st.title("⚙️ System Diagnostics & Configuration")
    st.caption("Live health check of all database, vector store, and LLM services")

    if not api_status:
        st.error("Backend API is currently OFFLINE at " + api_url)
        st.write("To start the backend, execute:")
        st.code(".venv\\Scripts\\activate\nuvicorn api:app --host 127.0.0.1 --port 8000 --reload")
    else:
        st.success("FastAPI Backend is ONLINE and responding properly.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("PostgreSQL Database", "Connected" if "OK" in api_status.get("database", "") else "Issue")
            st.caption(api_status.get("database", ""))
        with c2:
            st.metric("Ollama Service", "Connected" if "OK" in api_status.get("ollama", "") else "Issue")
            st.caption(api_status.get("ollama", ""))
        with c3:
            st.metric("pgvector Collection", "Ready" if "OK" in api_status.get("index", "") else "Empty/Check")
            st.caption(api_status.get("index", ""))

        st.divider()
        st.subheader("Configured Wikipedia Articles")
        articles = api_status.get("configured_articles", [])
        if articles:
            for idx, a in enumerate(articles, 1):
                st.write(f"{idx}. **{a}**")
        else:
            st.write("No configured articles found in settings.")
