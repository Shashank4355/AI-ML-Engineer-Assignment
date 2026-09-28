"""Bonus: LLM-as-judge eval. Answers come from our vLLM-backed /ask API;
Gemini is used ONLY as the judge (allowed by the assignment rules).

The two steps can run on different machines (answers on the GPU box, judging anywhere):
  python eval/eval.py --collect eval/answers.json   (API must be running on :8081)
  python eval/eval.py --judge   eval/answers.json   (needs GEMINI_API_KEY)
  python eval/eval.py                                (both in one go)
"""
import argparse
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()
API = os.getenv("API_URL", "http://localhost:8081")

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
relevance = answer addresses the question (an honest "not found" for an out-of-scope question is 5)."""


def collect(path):
    rows = []
    for q in QUESTIONS:
        r = requests.post(f"{API}/ask", json={"question": q}, timeout=900).json()
        rows.append({"question": q, **r})
        print(f"{r['agent_path'][-1]:>12} | {q}")
    json.dump(rows, open(path, "w"), indent=2)
    return rows


def judge(rows):
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")
    for row in rows:
        g = client.models.generate_content(
            model=model, contents=JUDGE.format(q=row["question"], a=row["answer"], s=row["sources"]))
        text = g.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        row.update(json.loads(text))
        print(f"[F{row['faithfulness']} R{row['relevance']}] {row['question']}")
    n = len(rows)
    print(f"\nMean faithfulness: {sum(r['faithfulness'] for r in rows) / n:.2f}/5")
    print(f"Mean relevance:    {sum(r['relevance'] for r in rows) / n:.2f}/5")
    json.dump(rows, open("eval/results.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", metavar="OUT")
    ap.add_argument("--judge", metavar="IN")
    args = ap.parse_args()
    if args.collect:
        collect(args.collect)
    elif args.judge:
        judge(json.load(open(args.judge)))
    else:
        judge(collect("eval/answers.json"))
