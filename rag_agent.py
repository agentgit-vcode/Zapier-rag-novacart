"""RAG Agent — retrieves context from Pinecone and generates grounded answers via OpenAI."""

import json
import os
import re

from openai import OpenAI
from dotenv import load_dotenv

from vectorstore import search_chunks

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o"
TOP_K = 8

SYSTEM_PROMPT = """ROLE:

You are a Grounded Retrieval-Augmented Intelligence Agent.

PURPOSE:

Answer user questions strictly using information retrieved from a vector database.
Do not use prior knowledge or assumptions.

DATA SOURCE:

- All knowledge comes from retrieved vector database results
- Each result contains:
  - chunk_text
  - chunk_id
  - metadata (file type, KPI, department, time period, source)

GROUNDING RULES (MANDATORY):

- Use ONLY retrieved chunks
- Do NOT infer, speculate, or predict
- Do NOT answer without evidence
- If evidence is insufficient, respond exactly with:
  "Insufficient data to answer confidently based on available sources."

REASONING RULES:

- Summarize or compare only if supported by retrieved data
- Flag anomalies only when explicitly supported
- If explanations are missing, state "Needs human review"
- If sources conflict, state the conflict and cite all relevant chunk_ids
- Do not resolve conflicts

CITATION RULES:

- Every factual claim must include an inline citation
- Citation format: (source: chunk_id)
- Never cite non-retrieved sources

RESPONSE FORMAT (FIXED):

You must return a valid JSON object with these exact fields:

{
    "answer_summary": "Short, factual response with inline (source: chunk_id) citations",
    "supporting_evidence": ["Bullet point 1 with (source: chunk_id)", "Bullet point 2..."],
    "anomalies_risks": ["Only if supported by data, otherwise empty list"],
    "confidence": "High" | "Medium" | "Low"
}

TONE:

Professional, conservative, executive-ready.

FINAL CHECK:

- All claims cited
- No external knowledge used
- Uncertainty clearly stated where applicable"""


def ask_question(question: str) -> dict:
    """Search Pinecone for relevant chunks, then generate a grounded answer."""

    # Step 1: Retrieve relevant chunks from Pinecone
    chunks = search_chunks(question, top_k=TOP_K)

    if not chunks:
        return {
            "answer": "No relevant documents found in the knowledge base for this question.",
            "confidence": "Low",
            "sources": [],
        }

    # Step 2: Build context from retrieved chunks
    context_parts = []
    sources = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        chunk_text = chunk["text"]
        file_type = chunk.get("file_type", "unknown")
        score = chunk["score"]

        context_parts.append(
            f"[chunk_id: {chunk_id}] [file_type: {file_type}] [score: {score:.3f}]\n{chunk_text}"
        )

        sources.append({
            "chunk_id": chunk_id,
            "file_type": file_type,
            "score": score,
            "text_preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
            "metadata": chunk.get("metadata", {}),
        })

    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Call OpenAI with retrieved context
    user_message = f"""RETRIEVED CONTEXT:

{context}

---

QUESTION:

{question}

INSTRUCTIONS:

Answer using only the retrieved context above. If the information is insufficient or unclear, do not guess."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    # Step 4: Parse response
    raw = response.choices[0].message.content
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "answer": raw,
            "confidence": "Low",
            "sources": sources,
        }

    # Build formatted answer
    answer_parts = [result.get("answer_summary", "")]

    evidence = result.get("supporting_evidence", [])
    if evidence:
        answer_parts.append("\n**Supporting Evidence:**")
        for item in evidence:
            answer_parts.append(f"- {item}")

    anomalies = result.get("anomalies_risks", [])
    if anomalies:
        answer_parts.append("\n**Anomalies / Risks:**")
        for item in anomalies:
            answer_parts.append(f"- {item}")

    return {
        "answer": "\n".join(answer_parts),
        "confidence": result.get("confidence", "Unknown"),
        "sources": sources,
    }
