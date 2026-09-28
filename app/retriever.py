"""Hybrid retrieval: dense (bge + Qdrant) + sparse (BM25), fused with Reciprocal Rank Fusion."""
import pickle
from functools import lru_cache

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.config import BM25_PATH, COLLECTION, EMBED_MODEL, QDRANT_URL, TOP_K

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _resources():
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    return SentenceTransformer(EMBED_MODEL), QdrantClient(url=QDRANT_URL), bm25


def _key(r: dict) -> str:
    return f"{r['source']}|{r['page']}|{r['text'][:60]}"


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    model, qdrant, bm = _resources()

    vec = model.encode(BGE_QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    dense = [p.payload for p in qdrant.query_points(COLLECTION, query=vec, limit=k * 3).points]

    scores = bm["bm25"].get_scores(query.lower().split())
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k * 3]
    sparse = [bm["records"][i] for i in top]

    fused, docs = {}, {}
    for ranking in (dense, sparse):
        for rank, r in enumerate(ranking):
            key = _key(r)
            fused[key] = fused.get(key, 0) + 1 / (60 + rank)
            docs[key] = r
    return [docs[key] for key in sorted(fused, key=fused.get, reverse=True)[:k]]
