"""Download the knowledge base: 12 public arXiv papers on LLMs / RAG / serving."""
from pathlib import Path
import urllib.request

PAPERS = {
    "attention_is_all_you_need": "1706.03762",
    "bert": "1810.04805",
    "rag_lewis": "2005.11401",
    "dense_passage_retrieval": "2004.04906",
    "lora": "2106.09685",
    "chain_of_thought": "2201.11903",
    "react": "2210.03629",
    "vllm_pagedattention": "2309.06180",
    "self_rag": "2310.11511",
    "corrective_rag": "2401.15884",
    "bge_m3": "2402.03216",
    "qwen2_5_vl": "2502.13923",
}

out = Path("data/pdfs")
out.mkdir(parents=True, exist_ok=True)
for name, arxiv_id in PAPERS.items():
    dest = out / f"{name}.pdf"
    if dest.exists():
        continue
    print(f"Downloading {name}")
    req = urllib.request.Request(f"https://arxiv.org/pdf/{arxiv_id}", headers={"User-Agent": "Mozilla/5.0"})
    dest.write_bytes(urllib.request.urlopen(req).read())
print("Done.")
