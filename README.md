# NovaCart RAG Agent — Grounded Q&A over Internal Documents

A web-based RAG (Retrieval-Augmented Generation) agent that answers business questions about NovaCart using only evidence retrieved from a Pinecone vector database — never from the LLM's prior knowledge. Every answer is cited, grounded, and verifiable.

## How It Works

```
User Question
      │
      ▼
┌─────────────────────┐
│  OpenAI Embeddings  │  text-embedding-3-small (1536 dims)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Pinecone Search    │  Top-8 chunks by cosine similarity
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  GPT-4o (OpenAI)    │  Grounded generation with strict rules
└────────┬────────────┘
         │
    ┌────┼────────────────────────┐
    │    │                        │
Answer  Citations              Confidence
Summary (source: chunk_id)     High/Med/Low
```

## Features

- **Chat** — natural language Q&A with streaming-style responses
- **Citations** — every factual claim cites the source chunk ID
- **Sources Panel** — view retrieved chunks ranked by relevance score
- **Confidence Level** — High / Medium / Low based on evidence quality
- **Grounding** — never hallucinates, never uses LLM prior knowledge
- **Multi-format Knowledge Base** — indexes DOCX, PPTX, PDF, and XLSX documents

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Vector DB | Pinecone (serverless) |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | OpenAI GPT-4o |
| Document Parsing | python-docx, python-pptx, PyPDF2, openpyxl |
| Language | 100% Python |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/agentgit-vcode/Zapier-rag-novacart.git
cd Zapier-rag-novacart

# Install dependencies
pip install -r requirements.txt

# Set your API keys
cp .env.example .env
# Edit .env and add your OpenAI and Pinecone API keys

# Seed the vector database (one-time)
python seed.py

# Run the app
streamlit run app.py
```

## Knowledge Base

The following NovaCart documents are chunked, embedded, and indexed into Pinecone:

| Document | Format | Content |
|----------|--------|---------|
| NovaCart Company Annual Report 2024 | DOCX | Financial results, KPIs, business strategy |
| NovaCart Company Intro | PPTX | Company overview, mission, product lines |
| NovaCart Product Catalog | PDF | SKU details, pricing, product descriptions |
| SKU Weekly Sales & Conversion (3Y) | XLSX | Weekly sales, conversion rates, revenue by SKU |

## Agent Decision Logic

| Condition | Agent Behavior |
|-----------|---------------|
| Sufficient evidence found | Answers with inline citations and "High" confidence |
| Partial evidence | Answers with caveats, flags gaps, "Medium" confidence |
| Conflicting sources | States the conflict, cites all chunk_ids, does not resolve |
| No relevant chunks | Returns "Insufficient data to answer confidently" |
| Question requires speculation | Refuses to infer — states "Needs human review" |

## Grounding Rules (Enforced via System Prompt)

- Use ONLY retrieved chunks — no LLM prior knowledge
- Every factual claim must cite `(source: chunk_id)`
- Never infer, speculate, or predict beyond retrieved data
- If evidence is insufficient, say so explicitly
- Conflicting sources are surfaced, not resolved

## Project Structure

```
├── app.py                  # Streamlit app (Chat, Sources, About pages)
├── rag_agent.py            # GPT-4o integration (grounded generation)
├── vectorstore.py          # Pinecone client (embed, upsert, search)
├── seed.py                 # Document chunking and indexing pipeline
├── Agent Prompt.txt        # Original Zapier agent prompt (reference)
├── Files/
│   ├── NovaCart Company Annual Report 2024.docx
│   ├── NovaCart Company Intro.pptx
│   ├── NovaCart_Product_Catalog.pdf
│   └── SKU_Weekly_Sales_Conversion_3Y_with_Revenue.xlsx
├── docs/
│   ├── PRD.md              # Product Requirements Document
│   └── TECH_SPEC.md        # Technical Specification
├── .streamlit/
│   └── config.toml
├── .env.example
├── requirements.txt
└── README.md
```

## Documentation

- **[Product Requirements Document (PRD)](docs/PRD.md)** — problem statement, user stories, functional requirements, grounding rules, success metrics
- **[Technical Specification](docs/TECH_SPEC.md)** — architecture, data pipeline, prompt design, retrieval parameters, ADRs

## Background

This project started as a Zapier-based automation (Chatbot → Pinecone → ChatGPT) and was rebuilt into a full-stack Python application with direct API integration, a proper chunking pipeline, and a Streamlit UI for interactive exploration.
