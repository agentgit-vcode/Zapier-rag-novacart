# Technical Specification

## NovaCart RAG Agent — Grounded Q&A over Internal Documents

| Field | Details |
|-------|---------|
| **Author** | Vaibhav Lad |
| **Date** | May 2026 |
| **PRD Reference** | [PRD.md](PRD.md) |
| **Status** | Implemented |

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      Streamlit App                           │
│                                                              │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐           │
│  │  Chat    │      │ Sources  │      │  About   │           │
│  │  Page    │      │  Page    │      │  Page    │           │
│  └────┬─────┘      └────┬─────┘      └──────────┘           │
│       │                 │                                    │
│       └────────┬────────┘                                    │
│                │                                             │
│       ┌────────▼────────┐                                    │
│       │   rag_agent.py  │  Orchestrates retrieval + LLM      │
│       └───┬─────────┬───┘                                    │
│           │         │                                        │
│  ┌────────▼───┐  ┌──▼──────────────┐                         │
│  │vectorstore │  │  OpenAI API     │                         │
│  │   .py      │  │  (GPT-4o)       │                         │
│  └────┬───────┘  └─────────────────┘                         │
│       │                                                      │
│  ┌────▼──────────────────────┐                                │
│  │  Pinecone Vector Database │                                │
│  │  (novacart-rag index)     │                                │
│  └───────────────────────────┘                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘

        ┌────────────┐
        │  seed.py   │  One-time: chunk docs → embed → upsert
        └────┬───────┘
             │
     ┌───────▼────────┐
     │    Files/       │  DOCX, PPTX, PDF, XLSX
     └────────────────┘
```

**Design Principle:** Minimal-dependency RAG pipeline. No LLM frameworks (LangChain, LlamaIndex). Direct API calls to OpenAI and Pinecone. The entire retrieval and generation flow is under 150 lines of code.

---

## 2. Tech Stack

| Component | Technology | Version | Justification |
|-----------|-----------|---------|---------------|
| UI Framework | Streamlit | >= 1.38 | Rapid prototyping, built-in chat components |
| Vector DB | Pinecone | >= 5.0 | Managed vector search, no infra to maintain |
| Embeddings | OpenAI text-embedding-3-small | — | 1536 dimensions, good quality/cost ratio |
| LLM | GPT-4o | gpt-4o | Best quality for grounded generation with JSON mode |
| Language | Python | >= 3.10 | Single language for entire stack |

### Why No LLM Framework (LangChain, LlamaIndex)?

The RAG pipeline is a linear 3-step flow: embed query → search Pinecone → call GPT-4o. There are no chains, no agents with tool selection, no conversation memory, no complex routing. Using a framework would add 20+ transitive dependencies to wrap 3 API calls. See [ADR-1](#adr-1-no-llm-framework) below.

### Why Pinecone?

| Alternative | Why Not |
|------------|---------|
| ChromaDB | Good for local dev, but no managed cloud offering at scale |
| Weaviate | More complex setup, over-featured for a single-index use case |
| pgvector | Requires Postgres, adds DB ops overhead |
| FAISS | In-memory only, no persistence without custom code |

Pinecone provides a managed, serverless vector index with a simple API. Zero infrastructure management.

---

## 3. Data Pipeline

### 3.1 Document Processing (seed.py)

```
Files/                          seed.py                          Pinecone
──────                          ───────                          ────────
Annual Report (.docx)  ──►  python-docx extracts text  ──►
Company Intro (.pptx)  ──►  python-pptx extracts text  ──►  chunk_text()     ──►  embed_texts()  ──►  upsert()
Product Catalog (.pdf) ──►  PyPDF2 extracts text       ──►  500 chars/chunk       OpenAI API         Pinecone
Sales Data (.xlsx)     ──►  openpyxl extracts rows     ──►  100 char overlap
```

### 3.2 Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 500 characters | Small enough for precise retrieval, large enough for context |
| Overlap | 100 characters | Prevents splitting mid-sentence at chunk boundaries |
| XLSX handling | Row-level with headers | Each row becomes "Header1: Val1 \| Header2: Val2 \| ..." |

### 3.3 Metadata Schema

Each vector in Pinecone stores:

```json
{
    "id": "annual_report_0012",
    "values": [0.023, -0.041, ...],
    "metadata": {
        "file_name": "NovaCart Company Annual Report 2024.docx",
        "file_type": "annual_report",
        "chunk_index": 12,
        "chunk_text": "First 1000 chars of the chunk for retrieval display..."
    }
}
```

---

## 4. RAG Pipeline

### 4.1 Query Flow

```
1. User submits question via Streamlit chat input
2. vectorstore.embed_text() embeds the question (text-embedding-3-small)
3. vectorstore.search_chunks() queries Pinecone (top_k=8, cosine similarity)
4. rag_agent.ask_question() builds context from retrieved chunks
5. OpenAI GPT-4o generates a grounded answer (JSON mode)
6. Response is parsed and displayed with citations and confidence
```

### 4.2 Prompt Design

The system prompt enforces:

| Constraint | Implementation |
|------------|---------------|
| Grounding | "Use ONLY retrieved chunks. Do NOT infer, speculate, or predict." |
| Citations | "Every factual claim must include (source: chunk_id)" |
| Structured output | JSON response format with answer_summary, evidence, anomalies, confidence |
| Conflict handling | "If sources conflict, state the conflict and cite all relevant chunk_ids" |
| Insufficient data | Fixed fallback: "Insufficient data to answer confidently based on available sources." |

### 4.3 Response Schema

```json
{
    "answer_summary": "NovaCart's total revenue in 2024 was $12.4M (source: annual_report_0003)",
    "supporting_evidence": [
        "Q1 revenue was $2.8M (source: annual_report_0005)",
        "Q4 revenue was $4.1M, driven by holiday sales (source: annual_report_0012)"
    ],
    "anomalies_risks": [
        "Q2 revenue dropped 15% QoQ — report attributes this to supply chain delays (source: annual_report_0008)"
    ],
    "confidence": "High"
}
```

---

## 5. Retrieval Parameters

| Parameter | Value | Tuning Notes |
|-----------|-------|-------------|
| top_k | 8 | Balances recall vs. context window cost; 5-10 is typical |
| Embedding model | text-embedding-3-small | 1536 dims; upgrade to text-embedding-3-large (3072 dims) for better recall |
| Similarity metric | Cosine | Default for normalized embeddings |
| Chunk text in metadata | First 1000 chars | Allows displaying source text without a separate store |

---

## 6. Project Structure

```
├── app.py                  # Streamlit app (Chat, Sources, About pages)
├── rag_agent.py            # OpenAI GPT-4o integration (grounded generation)
├── vectorstore.py          # Pinecone client (embed, upsert, search, stats)
├── seed.py                 # Document chunking and Pinecone indexing
├── Agent Prompt.txt        # Original Zapier agent prompt (reference)
├── Files/
│   ├── NovaCart Company Annual Report 2024.docx
│   ├── NovaCart Company Intro.pptx
│   ├── NovaCart_Product_Catalog.pdf
│   └── SKU_Weekly_Sales_Conversion_3Y_with_Revenue.xlsx
├── docs/
│   ├── PRD.md              # Product Requirements Document
│   └── TECH_SPEC.md        # This file
├── .streamlit/
│   └── config.toml         # Streamlit UI config
├── .env.example
├── requirements.txt
└── README.md
```

---

## 7. Architectural Decision Records (ADRs)

### ADR-1: No LLM Framework

**Decision:** Use direct OpenAI and Pinecone SDK calls instead of LangChain or LlamaIndex.

**Context:** The RAG pipeline is a 3-step linear flow (embed → search → generate). There is no multi-agent orchestration, no tool selection, no conversation memory, and no complex chain routing.

**Consequences:**
- (+) Zero framework lock-in — easy to swap models or vector DBs
- (+) ~8 dependencies total instead of ~50
- (+) Full visibility into every API call — no hidden abstractions
- (-) Must manually handle embedding batching and chunk assembly
- (-) No built-in conversation memory (not needed for single-turn Q&A)

### ADR-2: Pinecone over Local Vector Store

**Decision:** Use Pinecone (managed) instead of ChromaDB or FAISS (local).

**Context:** The original Zapier-based agent used Pinecone. Keeping the same vector DB preserves the indexed data and allows direct comparison between the Zapier and Python implementations.

**Consequences:**
- (+) Zero infrastructure — serverless, managed index
- (+) Consistent with the original Zapier implementation
- (+) Scales beyond demo without architecture changes
- (-) Requires API key and network access
- (-) Cannot run fully offline

### ADR-3: Text Embedding 3 Small over Ada-002

**Decision:** Use `text-embedding-3-small` (1536 dims) for embeddings.

**Context:** OpenAI's newer embedding model offers better quality at lower cost compared to the legacy `text-embedding-ada-002`.

**Consequences:**
- (+) Better retrieval quality for the same dimensionality
- (+) Lower cost per token
- (-) Requires re-embedding if migrating from ada-002
