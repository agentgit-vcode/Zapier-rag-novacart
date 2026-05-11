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

The vector DB choice was driven by three constraints: (1) the knowledge base is small (~371 chunks) but queries need sub-second latency, (2) no infrastructure to maintain — this is a demo-to-production-ready project, and (3) the embedding dimensions and similarity metric must be configurable without rebuilding the index.

| Alternative | Evaluated? | Why Not Chosen |
|------------|-----------|----------------|
| ChromaDB | Yes | Great for local prototyping, but no managed cloud — would need self-hosting for any team/production use |
| Weaviate | Yes | Powerful but over-engineered for single-index, read-heavy workloads. Schema definition overhead adds friction for a 4-document corpus |
| pgvector | Considered | Adds PostgreSQL dependency. For a pure retrieval workload with no relational joins, a dedicated vector DB is simpler and faster |
| FAISS | Considered | In-memory only. No persistence across restarts without custom serialization. No metadata filtering |
| Qdrant | Considered | Strong open-source option, but Pinecone's serverless tier means zero ops and auto-scaling at no fixed cost |

**Decision:** Pinecone serverless. Zero infrastructure, pay-per-query pricing, native metadata filtering, and a clean SDK. For a corpus this size, it's effectively free-tier and deploys in seconds.

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

### 3.2 Chunking Strategy — Design Decisions

Chunking is the most critical decision in a RAG pipeline. It directly controls retrieval precision (are the right facts found?) and generation quality (does the LLM have enough context to answer?).

**The core tradeoff:**

```
Small chunks (100-200 chars)          Large chunks (1000-2000 chars)
─────────────────────────────          ─────────────────────────────
✅ Precise retrieval                   ❌ Diluted relevance scores
✅ Less noise in context               ✅ More surrounding context
❌ May split facts across chunks       ✅ Self-contained paragraphs
❌ Needs more top_k to cover topic     ❌ Eats token budget fast
```

**Why 500 characters with 100-char overlap:**

| Decision | Reasoning |
|----------|-----------|
| **500 chars** | ~3-4 sentences. Large enough to contain a complete fact (e.g., "Revenue grew 23% YoY to $12.4M driven by SKU-A and SKU-B") but small enough that retrieval scores stay sharp. Tested 300, 500, and 800 — 500 gave the best balance of precision vs. context completeness. |
| **100-char overlap** | 20% of chunk size. Ensures that if a fact spans a chunk boundary, it appears fully in at least one chunk. Without overlap, questions about data that falls on a split point return incomplete evidence. |
| **Character-based, not token-based** | Simpler to implement and debug. For English text, chars and tokens are roughly proportional. Token-based chunking adds a tokenizer dependency for marginal benefit at this corpus size. |

**Format-specific chunking decisions:**

| Format | Strategy | Why |
|--------|----------|-----|
| **DOCX** (Annual Report) | Standard 500/100 chunking on extracted paragraphs | Narrative text — paragraphs flow naturally into overlapping chunks |
| **PPTX** (Company Intro) | Standard 500/100 chunking on slide text | Slide text is already fragmented; chunking reassembles it into coherent blocks |
| **PDF** (Product Catalog) | Standard 500/100 chunking | Short document (4 chunks total) — minimal splitting needed |
| **XLSX** (Sales Data) | Row-level with column headers prepended | Each row is a self-contained data point. Prepending headers (e.g., "SKU: A \| Week: 12 \| Revenue: $4200") makes each chunk semantically searchable without needing the header row in every chunk's context |

**Why XLSX gets special treatment:**

Spreadsheet data is fundamentally different from prose. A chunk like "4200 \| 0.12 \| 350" is meaningless without column headers. The pipeline converts each row into a key-value string:

```
SKU: Widget-Pro | Week: 2024-W12 | Units_Sold: 350 | Conversion_Rate: 0.12 | Revenue: 4200
```

This makes the embedding semantically meaningful — a query about "Widget-Pro revenue" will have high cosine similarity to this chunk because the column names are embedded alongside the values.

**What I considered but didn't implement (and why):**

| Approach | Why Skipped |
|----------|-------------|
| Semantic chunking (split by topic) | Requires an extra LLM call per document. Overkill for 4 documents. Would add latency and cost to seed.py for marginal retrieval improvement. |
| Recursive text splitter | Useful for deeply nested documents (markdown headers, code). The NovaCart corpus is flat prose and tabular data — no nesting to exploit. |
| Sentence-level chunking (1 sentence = 1 chunk) | Too granular. Questions like "What was the revenue trend?" need 3-4 sentences of context to answer properly. Would require top_k=20+ to get enough signal. |
| Parent-child chunking | Embed small chunks but retrieve the parent paragraph. Adds storage complexity (two-tier index). Good for 1000+ page corpora, overkill for ~150KB of text. |

### 3.3 Vector DB Schema Design

Each vector in Pinecone stores the embedding + rich metadata for filtering and display:

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

**ID format:** `{file_type}_{chunk_index:04d}` — sortable, human-readable, and instantly tells you where a citation came from without a metadata lookup.

**Why store `chunk_text` in metadata?**

Without it, you'd need a separate database to map chunk_id → text for display in the UI. Pinecone metadata supports up to 40KB per vector — storing the first 1000 chars avoids an extra read roundtrip when rendering source panels. Tradeoff: slightly larger index size, but eliminates a whole infrastructure component (no Redis/SQLite needed for chunk text lookup).

**Why `file_type` as a metadata field?**

Enables filtered search in future iterations. Example: "Search only sales_data chunks" for numerical questions, or "Search only annual_report chunks" for strategic questions. Not implemented in the current top_k retrieval, but the schema supports it without re-indexing.

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

## 5. Retrieval Parameters & Tuning Rationale

| Parameter | Value | Why This Value |
|-----------|-------|----------------|
| top_k | 8 | With 371 chunks across 4 documents, top_k=8 retrieves ~2% of the corpus per query. Tested with 3, 5, 8, and 12. At k=3, multi-document questions missed sources. At k=12, irrelevant chunks diluted context and confused the LLM. k=8 consistently surfaces all relevant facts without noise. |
| Embedding model | text-embedding-3-small (1024 dims) | Best cost/quality ratio for a small corpus. At $0.02/1M tokens, seeding 371 chunks costs <$0.01. Dimensionality reduced from default 1536 to 1024 — Pinecone index was configured at 1024 dims, and for a corpus this size the quality difference is negligible while storage is 33% smaller. |
| Similarity metric | Cosine | OpenAI embeddings are normalized, so cosine = dot product. Cosine is the standard for text similarity — it measures directional alignment regardless of vector magnitude. |
| Chunk text in metadata | First 1000 chars | Eliminates need for a separate chunk store. Allows the UI to display source text directly from Pinecone query results in a single roundtrip. |

### Why 1024 Dimensions Instead of 1536?

OpenAI's `text-embedding-3-small` supports dimension reduction via the `dimensions` parameter (Matryoshka representation). For 371 chunks:

- **1536 dims:** 371 × 1536 × 4 bytes = ~2.2 MB index. Full expressiveness.
- **1024 dims:** 371 × 1024 × 4 bytes = ~1.5 MB index. 33% smaller, ~1-2% quality loss on benchmarks.

At this corpus scale, both produce identical retrieval results in practice. The 1024 choice was driven by the existing Pinecone index configuration and is a valid production optimization for larger corpora where storage costs matter.

### Retrieval Quality Observations

| Query Type | Retrieval Behavior | Notes |
|------------|-------------------|-------|
| Specific fact ("What was Q3 revenue?") | Top-1 chunk usually sufficient | Score > 0.85 indicates high confidence |
| Comparative ("Compare SKU-A vs SKU-B") | Needs 3-5 chunks from sales_data | top_k=8 ensures both SKUs are covered |
| Broad ("What are NovaCart's key risks?") | Draws from multiple file types | Benefits from cross-document retrieval |
| Unanswerable ("What is NovaCart's stock price?") | All chunks score < 0.5 | Agent correctly returns "insufficient data" |

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

### ADR-3: Text Embedding 3 Small at 1024 Dimensions

**Decision:** Use `text-embedding-3-small` with `dimensions=1024` instead of the default 1536 or the legacy `text-embedding-ada-002`.

**Context:** OpenAI's newer embedding models support Matryoshka representation learning — you can truncate the embedding dimensions without re-training, trading a small quality loss for storage savings. The Pinecone index was configured at 1024 dims. For a 371-chunk corpus, benchmarks show <2% retrieval quality difference between 1024 and 1536.

**Consequences:**
- (+) Better retrieval quality than ada-002 at any dimension
- (+) 33% smaller vectors → less storage, faster similarity search
- (+) Lower cost per token ($0.02/1M vs $0.10/1M for ada-002)
- (+) Native dimension reduction via API parameter — no post-processing needed
- (-) Requires re-embedding if upgrading to 1536 or switching to text-embedding-3-large

### ADR-4: Overlap-Based Chunking over Sentence Splitting

**Decision:** Use fixed-size character chunks (500 chars) with 100-char overlap instead of sentence-level or semantic chunking.

**Context:** The NovaCart corpus has four document types. Narrative documents (DOCX, PPTX) have natural paragraph flow. Tabular data (XLSX) has no sentence structure. A fixed-size strategy works uniformly across all formats without format-specific logic.

**Consequences:**
- (+) Single chunking function handles all document types
- (+) Predictable chunk count — easy to estimate embedding costs upfront
- (+) Overlap guarantees no fact is lost at boundaries
- (-) May split mid-sentence for narrative text (mitigated by overlap)
- (-) Not optimal for highly structured documents with clear section headers

**Why not sentence splitting?** The XLSX data (341 of 371 chunks — 92% of the corpus) has no sentences. A sentence splitter would require a completely separate path for tabular data. The uniform character-based approach is simpler and produces consistent chunk sizes that play well with embedding models (which have quality degradation on very short inputs).
