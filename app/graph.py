"""Agentic workflow (LangGraph):

router -> [chitchat | out_of_scope | retrieve]
retrieve -> grade -> (relevant ? answer : rewrite -> retrieve, max MAX_RETRIES) / fallback
answer -> verify -> (supported ? END : fallback)
"""
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.config import MAX_RETRIES
from app.llm import chat
from app.retriever import retrieve

NOT_FOUND = "I couldn't find this in the provided documents."


class State(TypedDict, total=False):
    question: str
    query: str
    route: str
    chunks: list[dict]
    retries: int
    answer: str
    sources: list[dict]
    path: list[str]


def _step(state: State, name: str) -> list[str]:
    return state.get("path", []) + [name]


def router(state: State) -> State:
    label = chat(
        "Classify the user message into exactly one label:\n"
        "document - a question that could be answered from a corpus of AI/ML research papers "
        "(LLMs, transformers, RAG, retrieval, fine-tuning, model serving)\n"
        "chitchat - greetings, thanks, small talk\n"
        "out_of_scope - anything else (sports, news, general trivia, etc.)\n\n"
        f"Message: {state['question']}\nLabel:", max_tokens=5).lower()
    route = next((r for r in ("document", "chitchat", "out_of_scope") if r in label), "document")
    return {"route": route, "query": state["question"], "retries": 0, "path": _step(state, f"router:{route}")}


def chitchat(state: State) -> State:
    ans = chat(state["question"], system="You are a friendly assistant for a research-paper Q&A tool. "
               "Reply briefly and invite a question about the documents.", max_tokens=100)
    return {"answer": ans, "sources": [], "path": _step(state, "chitchat")}


def fallback(state: State) -> State:
    return {"answer": NOT_FOUND, "sources": [], "path": _step(state, "fallback")}


def retrieve_node(state: State) -> State:
    return {"chunks": retrieve(state["query"]), "path": _step(state, "retrieve")}


def grade(state: State) -> State:
    relevant = []
    for c in state["chunks"]:
        verdict = chat(f"Question: {state['question']}\n\nPassage: {c['text']}\n\n"
                       "Does this passage contain information useful to answer the question? "
                       "Answer yes or no.", max_tokens=3).lower()
        if verdict.startswith("yes"):
            relevant.append(c)
    return {"chunks": relevant, "path": _step(state, f"grade:{len(relevant)}_relevant")}


def rewrite(state: State) -> State:
    q = chat(f"Rewrite this question as a better search query for research papers. "
             f"Return only the query.\n\nQuestion: {state['question']}", max_tokens=60)
    return {"query": q, "retries": state["retries"] + 1, "path": _step(state, "rewrite")}


def _context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] (source: {c['source']}, page {c['page']})\n{c['text']}"
                       for i, c in enumerate(chunks))


def answer(state: State) -> State:
    ans = chat(
        f"Context:\n{_context(state['chunks'])}\n\nQuestion: {state['question']}\n\n"
        "Answer using ONLY the context. Cite every claim as [Source: <file>, page <n>]. "
        f"If the context does not contain the answer, reply exactly: {NOT_FOUND}",
        system="You are a precise assistant that answers strictly from provided documents.")
    sources = [{"source": c["source"], "page": c["page"], "type": c["type"]} for c in state["chunks"]]
    return {"answer": ans, "sources": sources, "path": _step(state, "answer")}


def verify(state: State) -> State:
    if NOT_FOUND in state["answer"]:
        return {"route": "unsupported", "path": _step(state, "verify:not_found")}
    verdict = chat(f"Context:\n{_context(state['chunks'])}\n\nAnswer: {state['answer']}\n\n"
                   "Is every claim in the answer supported by the context? Answer yes or no.",
                   max_tokens=3).lower()
    ok = verdict.startswith("yes")
    return {"route": "supported" if ok else "unsupported", "path": _step(state, f"verify:{'pass' if ok else 'fail'}")}


def after_grade(state: State) -> str:
    if state["chunks"]:
        return "answer"
    return "rewrite" if state["retries"] < MAX_RETRIES else "fallback"


def build_graph():
    g = StateGraph(State)
    for name, fn in [("router", router), ("chitchat", chitchat), ("fallback", fallback),
                     ("retrieve", retrieve_node), ("grade", grade), ("rewrite", rewrite),
                     ("answer", answer), ("verify", verify)]:
        g.add_node(name, fn)
    g.set_entry_point("router")
    g.add_conditional_edges("router", lambda s: s["route"],
                            {"document": "retrieve", "chitchat": "chitchat", "out_of_scope": "fallback"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade)
    g.add_edge("rewrite", "retrieve")
    g.add_edge("answer", "verify")
    g.add_conditional_edges("verify", lambda s: END if s["route"] == "supported" else "fallback")
    g.add_edge("chitchat", END)
    g.add_edge("fallback", END)
    return g.compile()


agent = build_graph()
