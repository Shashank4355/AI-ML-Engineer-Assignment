"""Offline pipeline: parse PDFs -> chunk -> (VLM figure captions) -> embed -> Qdrant + BM25.

Run:  python -m app.ingest            (text + figures)
      python -m app.ingest --no-images (text only, faster)
"""
import argparse
import pickle
import uuid
from pathlib import Path

import pymupdf as fitz
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import (BM25_PATH, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION,
                        EMBED_MODEL, PDF_DIR, QDRANT_URL)
from app.llm import chat_image

FIGURE_PROMPT = (
    "Describe this figure from a research paper in detail: its type (chart, diagram, table), "
    "every label, component, axis, value and relationship shown. Be factual and specific."
)


def chunk_text(text: str) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]


def extract(pdf_path: Path, with_images: bool) -> list[dict]:
    doc = fitz.open(pdf_path)
    records = []
    for page_no, page in enumerate(doc, start=1):
        for chunk in chunk_text(page.get_text()):
            records.append({"text": chunk, "source": pdf_path.name, "page": page_no, "type": "text"})

        if not with_images:
            continue
        for img in page.get_images(full=True):
            pix = fitz.Pixmap(doc, img[0])
            if pix.width < 200 or pix.height < 150:  # skip icons/logos
                continue
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            try:
                desc = chat_image(pix.tobytes("png"), FIGURE_PROMPT)
            except Exception as e:
                print(f"  ! figure caption failed on {pdf_path.name} p{page_no}: {e}")
                continue
            records.append({"text": f"[Figure] {desc}", "source": pdf_path.name,
                            "page": page_no, "type": "figure"})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    records = []
    for pdf in sorted(Path(PDF_DIR).glob("*.pdf")):
        print(f"Parsing {pdf.name}")
        records += extract(pdf, with_images=not args.no_images)
    print(f"{len(records)} chunks ({sum(r['type'] == 'figure' for r in records)} figures)")

    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([r["text"] for r in records], normalize_embeddings=True,
                           batch_size=32, show_progress_bar=True)

    qdrant = QdrantClient(url=QDRANT_URL)
    qdrant.recreate_collection(COLLECTION, vectors_config=VectorParams(size=vectors.shape[1],
                                                                       distance=Distance.COSINE))
    points = [PointStruct(id=str(uuid.uuid4()), vector=v.tolist(), payload=r)
              for v, r in zip(vectors, records)]
    for i in range(0, len(points), 256):
        qdrant.upsert(COLLECTION, points[i:i + 256])

    bm25 = BM25Okapi([r["text"].lower().split() for r in records])
    Path(BM25_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "records": records}, f)
    print("Indexing done.")


if __name__ == "__main__":
    main()
