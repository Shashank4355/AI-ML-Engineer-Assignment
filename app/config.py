import os
from dotenv import load_dotenv

load_dotenv()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
COLLECTION = "docs"
PDF_DIR = os.getenv("PDF_DIR", "data/pdfs")
BM25_PATH = os.getenv("BM25_PATH", "data/bm25.pkl")

CHUNK_SIZE = 800      # characters (~200 tokens)
CHUNK_OVERLAP = 150
TOP_K = 5
MAX_RETRIES = 2
