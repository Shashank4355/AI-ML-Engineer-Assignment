"""Bonus: LLM-as-judge eval. Answers come from our vLLM-backed /ask API;
Gemini is used ONLY as the judge (allowed by the assignment rules).

python eval/eval.py   (API must be running on :8081)
"""
import json
import os

import requests
from dotenv import load_dotenv
from google import genai

load_dotenv()
API = os.getenv("API_URL", "http://localhost:8081")
judge = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

QUESTIONS = [
    "How many attention heads does the base Transformer use?",
    "What optimizer and learning-rate schedule was used to train the Transformer?",
    "What are the two pre-training tasks of BERT?",
    "What is the difference between RAG-Sequence and RAG-Token?",
    "What encoder does Dense Passage Retrieval use for questions and passages?",
    "Why does LoRA introduce no additional inference latency?",
    "What problem does PagedAttention solve in LLM serving?",
    "What are reflection tokens in Self-RAG?",
    "What does the retrieval evaluator do in Corrective RAG?",
    "What three retrieval functionalities does BGE-M3 support?",
    "How does ReAct combine reasoning and acting?",
    "What is chain-of-thought prompting?",
    "Who won the 2022 FIFA World Cup?",  # should hit fallback
]

JUDGE = """You are grading a RAG system. Question: {q}
Answer: {a}
Retrieved sources: {s}

Return JSON only: {{"faithfulness": 1-5, "relevance": 1-5, "reason": "..."}}
faithfulness = answer only makes claims plausibly supported by the cited papers (an honest "not found" is 5).
relevance = answer addresses the question."""

rows = []
for q in QUESTIONS:
    r = requests.post(f"{API}/ask", json={"question": q}, timeout=300).json()
    g = judge.models.generate_content(model="gemini-2.5-flash",
                                      contents=JUDGE.format(q=q, a=r["answer"], s=r["sources"]))
    score = json.loads(g.text.strip().removeprefix("```json").removesuffix("```"))
    rows.append({"question": q, **r, **score})
    print(f"[F{score['faithfulness']} R{score['relevance']}] {q}")

n = len(rows)
print(f"\nMean faithfulness: {sum(r['faithfulness'] for r in rows) / n:.2f}/5")
print(f"Mean relevance:    {sum(r['relevance'] for r in rows) / n:.2f}/5")
json.dump(rows, open("eval/results.json", "w"), indent=2)
