"""NovaCart RAG Agent — Ask questions about NovaCart's business data."""

import streamlit as st

from rag_agent import ask_question
from vectorstore import get_index_stats

# --- Page Config ---
st.set_page_config(
    page_title="NovaCart RAG Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# --- Sidebar ---
st.sidebar.title("NovaCart RAG Agent")
st.sidebar.caption("Grounded Q&A over internal documents")
st.sidebar.divider()
page = st.sidebar.radio(
    "Navigation",
    ["Chat", "Sources", "About"],
    label_visibility="collapsed",
)

# --- Index Stats ---
st.sidebar.divider()
st.sidebar.subheader("Pinecone Index")
try:
    stats = get_index_stats()
    st.sidebar.metric("Vectors Indexed", stats["total_vectors"])
    st.sidebar.metric("Dimensions", stats["dimension"])
except Exception:
    st.sidebar.warning("Could not connect to Pinecone. Check your API key and index name.")

# --- Sample Questions ---
st.sidebar.divider()
st.sidebar.subheader("Try These")
sample_questions = [
    "What was NovaCart's total revenue in 2024?",
    "Which SKU had the highest conversion rate?",
    "What products does NovaCart sell?",
    "What are the key business risks mentioned in the annual report?",
]
for q in sample_questions:
    if st.sidebar.button(q, use_container_width=True):
        st.session_state.sample_question = q
        st.rerun()


# ============================================================
#  CHAT PAGE
# ============================================================
def render_chat():
    st.title("Ask NovaCart")
    st.caption(
        "Questions are answered using only retrieved evidence from the vector database. "
        "The agent never uses prior knowledge or guesses."
    )

    # Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry.get("confidence"):
                confidence_color = (
                    "green" if entry["confidence"] == "High"
                    else "orange" if entry["confidence"] == "Medium"
                    else "red"
                )
                st.markdown(f"**Confidence:** :{confidence_color}[{entry['confidence']}]")
            if entry.get("chunks_used"):
                with st.expander(f"Sources ({entry['chunks_used']} chunks)"):
                    for source in entry.get("sources", []):
                        st.markdown(
                            f"- `{source['chunk_id']}` — {source['file_type']} "
                            f"(score: {source['score']:.3f})"
                        )
                        st.caption(source["text_preview"])

    # Input — check for sample question or user input
    prefill = st.session_state.pop("sample_question", None)
    user_input = st.chat_input("Ask a question about NovaCart...")

    question = prefill or user_input

    if question:
        # Show user message
        with st.chat_message("user"):
            st.markdown(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    result = ask_question(question)
                except Exception as e:
                    st.error(f"Agent error: {e}")
                    return

            st.markdown(result["answer"])

            confidence = result.get("confidence", "Unknown")
            confidence_color = (
                "green" if confidence == "High"
                else "orange" if confidence == "Medium"
                else "red"
            )
            st.markdown(f"**Confidence:** :{confidence_color}[{confidence}]")

            sources = result.get("sources", [])
            if sources:
                with st.expander(f"Sources ({len(sources)} chunks)"):
                    for source in sources:
                        st.markdown(
                            f"- `{source['chunk_id']}` — {source['file_type']} "
                            f"(score: {source['score']:.3f})"
                        )
                        st.caption(source["text_preview"])

        # Save to history
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "confidence": result.get("confidence"),
            "chunks_used": len(sources),
            "sources": sources,
        })
        st.session_state.last_sources = sources


# ============================================================
#  SOURCES PAGE
# ============================================================
def render_sources():
    st.title("Retrieved Sources")
    st.caption("Chunks retrieved during the last query, ranked by relevance score.")

    sources = st.session_state.last_sources
    if not sources:
        st.info("No sources yet. Ask a question on the Chat page first.")
        return

    for i, source in enumerate(sources, 1):
        score = source["score"]
        score_color = "green" if score > 0.8 else "orange" if score > 0.6 else "red"

        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**#{i}** — `{source['chunk_id']}`")
            with col2:
                st.markdown(f"Type: **{source['file_type']}**")
            with col3:
                st.markdown(f"Score: :{score_color}[**{score:.3f}**]")

            st.text(source["text_preview"])

            # Show metadata if available
            metadata = source.get("metadata", {})
            if metadata:
                meta_cols = st.columns(len(metadata))
                for j, (key, value) in enumerate(metadata.items()):
                    if key not in ("chunk_id", "file_type", "chunk_text"):
                        meta_cols[j % len(meta_cols)].caption(f"{key}: {value}")


# ============================================================
#  ABOUT PAGE
# ============================================================
def render_about():
    st.title("About")

    st.markdown("""
This is a **Retrieval-Augmented Generation (RAG)** agent built for NovaCart, a fictional
e-commerce company. It answers questions using only evidence retrieved from internal
company documents — never from the LLM's prior knowledge.

### Architecture

```
User Question → Embedding (OpenAI) → Pinecone Vector Search → Top-K Chunks → GPT-4o → Grounded Response
```

### Knowledge Base

| Document | Type | Content |
|----------|------|---------|
| NovaCart Company Annual Report 2024 | DOCX | Financial results, KPIs, business strategy |
| NovaCart Company Intro | PPTX | Company overview, mission, product lines |
| NovaCart Product Catalog | PDF | SKU details, pricing, descriptions |
| SKU Weekly Sales & Conversion (3Y) | XLSX | Weekly sales, conversion rates, revenue by SKU |

### Grounding Rules

The agent follows strict grounding rules:

- Uses **only** retrieved chunks — no speculation or prior knowledge
- Every claim includes an inline citation: `(source: chunk_id)`
- Conflicting sources are surfaced, not resolved
- If evidence is insufficient, the agent says so explicitly

### Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| Vector DB | Pinecone |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI GPT-4o |
| Language | Python |
    """)

    st.divider()
    st.caption("Built as part of an AI/ML course — Week 2 RAG Agent assignment.")


# ============================================================
#  PAGE ROUTER
# ============================================================
if page == "Chat":
    render_chat()
elif page == "Sources":
    render_sources()
elif page == "About":
    render_about()
