"""End-to-end check runner for a GPU box (Kaggle/Colab): starts vLLM, Qdrant and the API,
builds the index, then exercises every agent path, /ask-image, /health and the benchmark.

python scripts/run_checks.py            (from the project root, after pip install)
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import requests

MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
API = "http://localhost:8081"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def start(cmd, log):
    # own session + no inherited stdout, so the notebook cell can finish while servers keep running
    return subprocess.Popen(f"exec {cmd} > {log} 2>&1", shell=True, start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for(url, minutes):
    for _ in range(minutes * 6):
        try:
            if requests.get(url, timeout=3).ok:
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


def ask(q):
    t = time.time()
    r = requests.post(f"{API}/ask", json={"question": q}, timeout=900).json()
    print(f"\nQ: {q} ({time.time() - t:.1f}s)\n{json.dumps(r, indent=2)}", flush=True)
    return r


# 1. vLLM
start(f"vllm serve {MODEL} --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.80 "
      "--quantization awq --dtype float16 --limit-mm-per-prompt '{\"image\":1}' --enforce-eager", "vllm.log")
check("vLLM server up", wait_for("http://localhost:8000/v1/models", 20))

# 2. Qdrant
if not Path("qdrant").exists():
    url = "https://github.com/qdrant/qdrant/releases/download/v1.12.4/qdrant-x86_64-unknown-linux-gnu.tar.gz"
    urllib.request.urlretrieve(url, "qdrant.tgz")
    subprocess.run("tar -xzf qdrant.tgz", shell=True, check=True)
start("./qdrant", "qdrant.log")
check("Qdrant up", wait_for("http://localhost:6333", 2))

# 3. Ingestion
subprocess.run([sys.executable, "scripts/download_docs.py"], check=True)
ing = subprocess.run([sys.executable, "-m", "app.ingest"], capture_output=True, text=True)
summary = [l for l in ing.stdout.splitlines() if "chunks" in l or "failed" in l]
print("\n".join(summary))
check("Ingestion", ing.returncode == 0 and Path("data/bm25.pkl").exists(), summary[0] if summary else ing.stderr[-500:])

# 4. API
start("uvicorn app.main:app --port 8081", "api.log")
ok = wait_for(f"{API}/health", 3)
health = requests.get(f"{API}/health").json() if ok else {}
check("/health", health.get("status") == "ok", str(health))

# 5. Agent paths
r = ask("What problem does PagedAttention solve in LLM serving?")
check("Document path (grade -> answer -> verify)", "verify:pass" in r["agent_path"] and "[Source:" in r["answer"], str(r["agent_path"]))
r = ask("What does Figure 1 of the Transformer paper show?")
check("Figure question answered from a figure chunk", any(s["type"] == "figure" for s in r["sources"]), str(r["sources"]))
r = ask("What are the two pre-training tasks of BERT?")
check("BERT question (known weak case)", "verify:pass" in r["agent_path"], r["answer"][:120])
r = ask("Hi there!")
check("Chit-chat route", r["agent_path"][0] == "router:chitchat", str(r["agent_path"]))
r = ask("Who won the 2022 FIFA World Cup?")
check("Out-of-scope -> fallback", r["agent_path"][-1] == "fallback" and "couldn't find" in r["answer"], str(r["agent_path"]))
r = ask("What learning rate did the LoRA paper use to fine-tune GPT-5?")
check("Plausible-but-unanswerable -> verify fail -> fallback", "couldn't find" in r["answer"], str(r["agent_path"]))
r = ask("What quantum error-correction code does the ReAct paper propose?")
check("Nothing relevant -> rewrite x2 -> fallback", r["agent_path"].count("rewrite") == 2 and r["agent_path"][-1] == "fallback", str(r["agent_path"]))
r = ask("")
# empty question should be rejected by the API (400) -- ask() returns the error body
check("Empty question rejected", "detail" in r, str(r))

# 6. Visual Q&A
import pymupdf
pymupdf.open("data/pdfs/attention_is_all_you_need.pdf")[2].get_pixmap(dpi=100).save("page.png")
with open("page.png", "rb") as f:
    r = requests.post(f"{API}/ask-image", data={"question": "Which components are in this diagram?"},
                      files={"image": ("page.png", f, "image/png")}, timeout=600).json()
print("\n/ask-image:", r.get("answer", r)[:400])
check("/ask-image", "answer" in r and len(r["answer"]) > 50)

# 7. Benchmark
b = subprocess.run([sys.executable, "scripts/benchmark.py", "--concurrency", "32"], capture_output=True, text=True)
print("\n" + b.stdout)
check("Benchmark", b.returncode == 0, b.stdout.strip().replace("\n", " | ") or b.stderr[-300:])

print("\n===== SUMMARY =====")
for name, ok, _ in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
print(f"{sum(ok for _, ok, _ in results)}/{len(results)} passed")
