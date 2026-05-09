# NovaCart RAG Agent (Zapier + Pinecone)

A Retrieval-Augmented Generation (RAG) agent built on Zapier that answers questions about NovaCart's business data using Pinecone as the vector database and ChatGPT (OpenAI) for response generation.

## Architecture

```
User Question → Zapier Chatbot → Pinecone Vector Search → ChatGPT (OpenAI) → Grounded Response
```

The agent follows strict grounding rules — it only answers using retrieved evidence from the vector database and never relies on the LLM's prior knowledge.

## Components

| Component | Role |
|-----------|------|
| **Zapier Chatbot** | User interface for submitting questions |
| **Pinecone** | Vector database storing embedded document chunks |
| **ChatGPT (OpenAI)** | Generates responses grounded in retrieved context |
| **Agent Prompt** | System prompt enforcing citation and grounding rules |

## Knowledge Base (Files/)

The following NovaCart documents were chunked, embedded, and indexed into Pinecone:

- **NovaCart Company Annual Report 2024.docx** — Annual financial and operational report
- **NovaCart Company Intro.pptx** — Company overview presentation
- **NovaCart_Product_Catalog.pdf** — Product catalog with SKU details
- **SKU_Weekly_Sales_Conversion_3Y_with_Revenue.xlsx** — 3 years of weekly sales, conversion, and revenue data by SKU

## Agent Behavior

The agent prompt (`Agent Prompt.txt`) enforces:

- **Grounding** — Only retrieved chunks are used; no speculation or prior knowledge
- **Citations** — Every factual claim includes `(source: chunk_id)` inline citations
- **Conflict handling** — Conflicting sources are surfaced, not resolved
- **Structured output** — Responses follow a fixed format: Answer Summary, Supporting Evidence, Anomalies/Risks, Confidence Level

## Setup

1. Create a Pinecone index and upload the embedded document chunks from `Files/`
2. Set up a Zapier Chatbot with a Pinecone search action
3. Add the ChatGPT (OpenAI) conversation action using the prompt in `Agent Prompt.txt`
4. Configure the Zapier workflow: Chatbot → Pinecone Search → ChatGPT → Reply

For a detailed walkthrough, see `Week 2-Zapier_Solution_Step-by-Step _Guide.docx`.

## Project Context

Built as part of an AI/ML course (Week 2 — RAG Agent assignment). The problem statement is in `RAG Problem Statement_Week 2.pdf`.
