# Product Requirements Document (PRD)

## NovaCart RAG Agent — Grounded Q&A over Internal Documents

| Field | Details |
|-------|---------|
| **Author** | Vaibhav Lad |
| **Date** | May 2026 |
| **Status** | MVP |
| **Version** | 1.0 |

---

## 1. Problem Statement

Business teams at NovaCart generate knowledge across multiple formats — annual reports (DOCX), presentations (PPTX), product catalogs (PDF), and sales data (XLSX). When stakeholders need answers, they must:

- **Search manually** across 4+ documents in different formats
- **Cross-reference** data points between financial reports and sales spreadsheets
- **Interpret** raw numbers without context from other sources
- **Wait** for analysts to compile answers to ad-hoc questions

### Impact

| Metric | Current State | Problem |
|--------|--------------|---------|
| Time to answer a business question | 30-60 min | Stakeholders wait for analyst availability |
| Cross-document lookup | Manual | Data lives in DOCX, PPTX, PDF, XLSX — no unified search |
| Answer accuracy | Varies | Depends on which documents the analyst checks |
| Knowledge accessibility | Low | Only people who know where files are can find answers |

---

## 2. Proposed Solution

A RAG (Retrieval-Augmented Generation) agent that:

1. **Indexes** all NovaCart documents into a vector database (Pinecone)
2. **Retrieves** the most relevant chunks for any natural language question
3. **Generates** a grounded answer using only retrieved evidence
4. **Cites** every factual claim with the source chunk ID

The agent answers questions. The documents provide the truth. The LLM never guesses.

### What This Is NOT

- Not a general-purpose chatbot — it only answers from indexed documents
- Not a data analytics tool — it retrieves pre-existing data, it doesn't compute new metrics
- Not a replacement for analysts — it handles lookup questions, not strategic analysis

---

## 3. Target Users

| User | Role | How They Use the Agent |
|------|------|----------------------|
| **Business Stakeholder** | Asks ad-hoc questions | Types a question, gets a cited answer in seconds |
| **Sales Manager** | Checks SKU performance | Asks about conversion rates, revenue, top products |
| **Executive** | Reviews company metrics | Asks about annual report KPIs without opening the document |

---

## 4. Functional Requirements

### 4.1 Document Ingestion (seed.py)

| ID | Requirement |
|----|-------------|
| F1 | Extract text from DOCX, PPTX, PDF, and XLSX files |
| F2 | Split extracted text into overlapping chunks (500 chars, 100 overlap) |
| F3 | Generate embeddings via OpenAI text-embedding-3-small |
| F4 | Upsert chunks with metadata (file name, file type, chunk index) to Pinecone |

### 4.2 Question Answering (rag_agent.py)

| ID | Requirement |
|----|-------------|
| F5 | Embed the user's question using the same embedding model |
| F6 | Retrieve top-8 most relevant chunks from Pinecone |
| F7 | Pass retrieved chunks + question to GPT-4o with grounding prompt |
| F8 | Return structured response: answer summary, evidence, anomalies, confidence |
| F9 | Include inline citations (source: chunk_id) for every factual claim |
| F10 | If evidence is insufficient, respond with explicit "insufficient data" message |

### 4.3 User Interface (app.py)

| ID | Requirement |
|----|-------------|
| F11 | Chat interface for submitting questions and viewing answers |
| F12 | Display confidence level (High / Medium / Low) with color coding |
| F13 | Expandable source panel showing retrieved chunks and relevance scores |
| F14 | Sources page with detailed view of last retrieval |
| F15 | Sidebar with Pinecone index stats and sample questions |
| F16 | About page explaining the architecture and grounding rules |

---

## 5. Grounding & Citation Rules

| Rule | Description |
|------|-------------|
| **No hallucination** | Agent uses only retrieved chunks — never LLM prior knowledge |
| **Mandatory citations** | Every factual claim must cite (source: chunk_id) |
| **Conflict transparency** | If sources conflict, state the conflict and cite all chunk_ids |
| **Explicit uncertainty** | If evidence is insufficient, say so — never guess |
| **No inference** | Do not predict, speculate, or extrapolate beyond retrieved data |

---

## 6. Success Metrics

| Metric | Target |
|--------|--------|
| Answer grounding rate | 100% of claims cite a retrieved chunk |
| Retrieval relevance | Top-8 chunks contain answer evidence > 80% of the time |
| Response time | Under 10 seconds for end-to-end question answering |
| Confidence calibration | "High" confidence answers are factually correct > 90% of the time |
