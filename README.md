# Document Intelligence Assistant: RAG + Agentic Q&A on vLLM

**Author:** Shashank Pandey

A self-hosted assistant that answers questions about a set of AI/ML research papers. Every answer cites its source (file and page), and when the documents don't contain the answer it says so instead of guessing. One vision-language model, served by **vLLM**, handles both text generation and understanding figures.

## Architecture

```
                 ┌───────────── offline: python -m app.ingest ─────────────┐
 12 arXiv PDFs ─►│ PyMuPDF text → 800-char chunks (150 overlap)             │
                 │ PyMuPDF images → Qwen2.5-VL caption → "[Figure]" chunks  │──► Qdrant (dense, bge-base)
                 │ bge-base-en-v1.5 embeddings                              │──► BM25 index (pickle)
                 └──────────────────────────────────────────────────────────┘

 POST /ask ─► LangGraph agent
     router ──chitchat──► chitchat ─────────────────────────────► END
       │  └─out_of_scope───────────────────────► fallback ──────► END
       └─document─► retrieve (hybrid, RRF) ─► grade ─ none relevant ─► rewrite ─┐ (max 2)
                        ▲                        │                               │
                        └────────────────────────┼───────────────────────────────┘
                                                 └ relevant ─► answer ─► verify ─ pass ─► END
                                                                           └ fail ─► fallback

 POST /ask-image ─► Qwen2.5-VL (same vLLM server) visual Q&A
 All LLM calls ─► vLLM OpenAI-compatible API (:8000) serving Qwen/Qwen2.5-VL-7B-Instruct-AWQ
```

## Setup

You need an NVIDIA GPU with at least 16 GB of VRAM (a T4, L4, A10 or better), Docker, and the NVIDIA container toolkit.

```bash
cp .env.example .env
python scripts/download_docs.py          # fetch the 12 PDFs into data/pdfs
docker compose up -d                     # starts vLLM, Qdrant and the API
pip install -r requirements.txt
python -m app.ingest                     # build the index (use --no-images to skip figure captions)
curl localhost:8081/health
curl -X POST localhost:8081/ask -H "Content-Type: application/json" \
     -d '{"question":"What problem does PagedAttention solve?"}'
curl -X POST localhost:8081/ask-image -F question="What does this chart show?" -F image=@chart.png
```

**No local GPU?** Open `colab_run.ipynb` in Google Colab with a T4 runtime and choose Run all. It writes all the source files, starts vLLM and Qdrant without Docker, builds the index, and runs the demo and the benchmark. The Colab results below came from this notebook. The Kaggle results came from `scripts/run_checks.py`.

Ingestion runs on the host and connects to vLLM on port 8000 and Qdrant on port 6333. The API reads the `data/bm25.pkl` file that ingestion writes, through a mounted volume.

## Design decisions

| Area | Choice | Why |
|---|---|---|
| Model | `Qwen2.5-VL-7B-Instruct-AWQ` for both text and vision | One model serves both jobs, so everything fits on a single 16–24 GB GPU. AWQ 4-bit cuts the weights to about 5 GB. |
| `--max-model-len 8192` | | The largest prompt is 5 chunks of about 200 tokens plus instructions (roughly 2k tokens), so 8k leaves headroom and keeps the KV cache small enough for more concurrent requests. |
| `--gpu-memory-utilization 0.90` | | Gives most of the VRAM to the KV cache while leaving room for the CUDA context. |
| `--dtype float16 --quantization awq` | | AWQ kernels need fp16. |
| `--tensor-parallel-size 1` | | A quantized 7B model fits on one GPU, so tensor parallelism would only add communication overhead. |
| Chunking | 800 chars (about 200 tokens), 150 overlap, per page | Small enough for precise retrieval and citations, and the overlap keeps sentences that cross a boundary. Chunking per page keeps page citations exact. |
| Embeddings | `BAAI/bge-base-en-v1.5` with its query instruction prefix | Strong MTEB retrieval scores at a CPU-friendly size (768 dimensions). |
| Retrieval | Dense (Qdrant) + BM25, fused with Reciprocal Rank Fusion (k=60) | BM25 catches exact terms (e.g. "PagedAttention", "RRF"), while dense search catches paraphrases. RRF needs no score calibration. |
| Agent | LangGraph state machine | Easy to read, and `agent_path` is returned from `/ask` for debugging. |
| Grounding | LLM relevance grading per chunk, a rewrite loop, and a yes/no hallucination check | Unsupported answers fall back to "I couldn't find this in the provided documents." |
| Multimodal | **Both options.** Option 1: figures are captioned by the VLM at ingest time and indexed with source and page. Option 2: `/ask-image` | Option 1 is the main one, because it makes questions about diagrams answerable through the normal `/ask` flow with citations. Option 2 took very little extra code since it uses the same server. |

## Benchmark

Measured on a **Tesla T4 (16 GB, Turing)**, on Google Colab and on Kaggle, with `python scripts/benchmark.py --concurrency 32` (32 requests at once, 256 max output tokens, streaming).

| GPU | Concurrency | Output tok/s | TTFT mean / P95 | Latency mean / P95 |
|---|---|---|---|---|
| Tesla T4 (Colab, first run on a cold server) | 32 | **231.8** | 9.56 s / 9.63 s | 21.01 s / 22.27 s |
| Tesla T4 (Kaggle, second run on a warm server) | 32 | **580.5** | 0.365 s / 0.377 s | 8.78 s / 8.83 s |

These settings were used on the T4: `--max-model-len 4096 --gpu-memory-utilization 0.80 --enforce-eager`. The T4 has less memory and the bge embedding model shares the GPU, which is why they are lower than the docker-compose defaults, which target a 24 GB GPU. `--enforce-eager` turns off CUDA graphs, which lowers throughput but saves memory and startup time on the T4. In the first row, the TTFT (time to first token) is high because all 32 prompts are prefilled at once on a GPU without FlashAttention-2, which needs compute capability 8.0 or higher. On an L4 or A10 with CUDA graphs enabled, both numbers should improve a lot.

The second row is faster mainly because the server was already warm: its kernels were compiled and cached. Also, all 32 requests share the same prompt, so vLLM's automatic prefix caching skips most of the prefill work. Read the first row as the pessimistic number and the second as the best case.

## Verification

`scripts/run_checks.py` starts vLLM, Qdrant and the API, builds the index, and then checks every agent path. Its first full run on Kaggle (2× T4) passed 12 of 14 checks. Both failures came from the test questions: one figure question (see the note below), and a rewrite-loop question that never triggered the loop. The script now uses questions that reliably reach each path, and each of those checks was run again against the same server:

| Check | Result | Agent path observed |
|---|---|---|
| vLLM server / Qdrant / ingestion / `/health` | PASS | |
| Document question with citation | PASS | `router:document → retrieve → grade:4_relevant → answer → verify:pass` |
| Figure question answered from a VLM figure caption | PASS* | sources include `attention_is_all_you_need.pdf p.3 (figure)` |
| BERT question (known weak case, see below) | PASS | `verify:pass` |
| Small talk | PASS | `router:chitchat → chitchat` |
| Out of scope | PASS | `router:out_of_scope → fallback` |
| Plausible but unanswerable (*"LoRA learning rate for GPT-5"*) | PASS | `… grade:1_relevant → answer → verify:fail → fallback` |
| Nothing relevant (*"quantum error-correction code in ReAct"*) | PASS | `grade:0 → rewrite → retrieve → grade:0 → rewrite → retrieve → grade:0 → fallback` |
| Empty question | PASS | HTTP 400 |
| `/ask-image` | PASS | `visual_qa` |
| Benchmark | PASS | see table above |

\* The first figure question, *"what sits on top of the decoder stack?"*, passed on Colab but returned "not found" on Kaggle. Figure captions are generated again on every ingest, so a different caption can drop a detail such as "Softmax". This is listed under limitations.

## Evaluation (bonus)

`eval/eval.py` sends 13 questions through `/ask`: 12 about the papers and 1 out of scope. **Gemini** (`gemini-3.8-flash`, key in `.env`) acts only as the judge and scores faithfulness and relevance from 1 to 5. Gemini never generates the answers; the assignment rules allow closed APIs only for judging. The two steps can run on different machines:

```bash
python eval/eval.py --collect eval/answers.json   # on the GPU box (Kaggle): answers from vLLM
python eval/eval.py --judge   eval/answers.json   # anywhere with GEMINI_API_KEY
```

**Results** (answers collected on Kaggle 2× T4, judged locally): see `eval/answers.json` and `eval/results.json`.

| Metric | Score |
|---|---|
| Mean faithfulness | **4.62 / 5** |
| Mean relevance | **4.69 / 5** |

Most answers scored 5/5. The misses, analysed honestly:

| Question | Score | What went wrong |
|---|---|---|
| BERT pre-training tasks | F2 | Answered "LTR + NSP" from the ablation section, when the correct answer is MLM + NSP. This is a retrieval-ranking error: the verifier checks support, not correctness. |
| LoRA: no inference latency | F3 | The core reason is correct (W = W0 + BA is merged), but the model added an unsupported claim about gradients. |
| DPR encoders | R2 | Described the passage encoder only, without saying that DPR uses two independent BERT encoders. |
| Transformer attention heads | (judged 5/5) | Returned "not found", but the answer (8 heads) *is* in the paper. The judge rewards honesty, yet this is a **retrieval miss**, so real recall is 11/12 answerable questions. |

A cross-encoder reranker and chunking that is aware of tables and sections would target most of these.

## Sample responses (real outputs from the Colab run)

These are the actual responses from `colab_run.ipynb`. The index has 1,628 chunks, 29 of which are figure descriptions written by the VLM.

```jsonc
// Document question: grade → answer → verify pass (10.4 s)
{"answer": "PagedAttention solves the problem of memory challenges in serving large language models ... by allowing continuous keys and values to be stored in non-contiguous memory space, thereby optimizing memory usage and improving the throughput of LLM serving engines like vLLM. [Source: vllm_pagedattention.pdf, page 2].",
 "agent_path": ["router:document", "retrieve", "grade:4_relevant", "answer", "verify:pass"]}

// Question about a figure, answered from a VLM figure caption (image-aware ingestion, Option 1)
{"answer": "The Softmax layer sits on top of the decoder stack. [Source: attention_is_all_you_need.pdf, page 3].",
 "sources": [{"source": "attention_is_all_you_need.pdf", "page": 3, "type": "figure"}],
 "agent_path": ["router:document", "retrieve", "grade:1_relevant", "answer", "verify:pass"]}

// Small talk
{"answer": "Hello! How can I assist you with your research paper today?",
 "agent_path": ["router:chitchat", "chitchat"]}

// Out of scope → honest fallback
{"answer": "I couldn't find this in the provided documents.", "sources": [],
 "agent_path": ["router:out_of_scope", "fallback"]}

// POST /ask-image with page 3 of the Transformer paper → the VLM describes the encoder/decoder diagram
{"answer": "... **Decoder**: The decoder is also composed of a stack of N identical layers ... The third sub-layer performs multi-head attention over the output of the encoder ...",
 "agent_path": ["visual_qa"]}
```

**A failure case, kept on purpose.** For *"What are the two pre-training tasks of BERT?"* the system answered "left-to-right LM and NSP". It cited `bert.pdf` p.14, which is the ablation section, when the correct answer is Masked LM and NSP. The retrieved chunk really does mention LTR and NSP, so the verifier passed the answer. This shows the verifier's limit: it checks that an answer is *supported by the retrieved text*, not that the right text was retrieved. A reranker would help here (see below).

## Known limitations

- Grading each chunk means up to 5 extra LLM calls per retrieval. This is simple but adds latency (about 5–10 s per question on a T4).
- The verifier checks that the answer is supported by the retrieved context, so it cannot catch wrong-but-supported answers such as the BERT case above.
- Figure captions come from the VLM and differ from one ingest to the next, so figure-detail questions can succeed in one run and get "not found" in another (seen between the Colab and Kaggle runs). Caching the captions and asking for exhaustive label lists would make them stable.
- Figures are only found when they are embedded raster images. Vector-drawn diagrams are missed.
- Chunking by character count can split tables, and there's no layout-aware parsing.
- The BM25 index is a pickle file that is rebuilt in full on each ingest, with no incremental updates.
- The yes/no checks from a 7B model are sometimes too strict or too lenient.

## Improvements with more time

- Grade all chunks in one batched call, or use a cross-encoder reranker (`bge-reranker-v2-m3`) instead of the per-chunk LLM grading. A reranker would also have fixed the BERT failure case.
- Use layout-aware parsing (docling) and render pages to find vector-drawn figures.
- Add SSE streaming, a Gradio UI, and a LoRA fine-tune for citation formatting.
- Benchmark with varied prompts, so that prefix caching doesn't make the throughput numbers look better than real traffic would.
