# Document Intelligence Assistant

RAG + agentic Q&A over a set of research papers, with the model self-hosted on vLLM.

Shashank Pandey

The knowledge base is 12 arXiv papers on LLMs, RAG and serving (Transformer, BERT, RAG, DPR, LoRA, CoT, ReAct, vLLM, Self-RAG, CRAG, BGE-M3, Qwen2.5-VL). You ask a question and get an answer with file/page citations. If the papers don't have the answer, it says "I couldn't find this in the provided documents." instead of making something up.

I used a single model, Qwen2.5-VL-7B-Instruct (AWQ), for both text and images. That way only one vLLM server is needed and everything fits on one 16 GB GPU.

## How it works

```
Ingestion (python -m app.ingest)
  PDFs -> PyMuPDF text -> 800-char chunks, 150 overlap, split per page
       -> embedded images -> Qwen2.5-VL writes a description -> "[Figure]" chunk
  all chunks -> bge-base-en-v1.5 -> Qdrant
             -> BM25 index (pickle)

POST /ask  (LangGraph)
  router --chitchat--> reply directly
         --out_of_scope--> fallback
         --document--> retrieve (dense + BM25, RRF)
                        -> grade each chunk (yes/no)
                           none relevant -> rewrite query -> retrieve again (max 2 times) -> fallback
                           some relevant -> answer with citations -> verify against context
                                                                     pass -> return
                                                                     fail -> fallback

POST /ask-image -> same vLLM server, question + image
```

`/ask` also returns `agent_path`, the list of nodes the request went through. This made debugging much easier.

## Running it

It needs an NVIDIA GPU with 16 GB+ (T4 or better).

```bash
cp .env.example .env
pip install -r requirements.txt
python scripts/download_docs.py
docker compose up -d            # vLLM + Qdrant + API
python -m app.ingest            # --no-images skips figure captioning (much faster)

curl localhost:8081/health
curl -X POST localhost:8081/ask -H "Content-Type: application/json" \
     -d '{"question":"What problem does PagedAttention solve?"}'
curl -X POST localhost:8081/ask-image -F question="What does this chart show?" -F image=@chart.png
```

I don't have a GPU locally, so I did all the testing on Colab and Kaggle:

- `colab_run.ipynb` is for Colab (T4).
- `kaggle_notebook.ipynb` is for Kaggle (T4 x2). This is the main one, and a run with all outputs is public here: https://www.kaggle.com/code/ssp499849/document-intelligence-assistant-rag-agent-on-vll
- `scripts/run_checks.py` starts everything and tests each agent path. The Kaggle notebook uses it.

## Choices and why

**Model / vLLM settings.** A 7B VL model in AWQ is about 5 GB of weights, so it fits on a T4 with room left for the KV cache. The main settings:

- `--max-model-len`: 8192 in docker-compose, 4096 on the T4. The biggest prompt I send is 5 chunks plus instructions, around 2k tokens, so 4096 is enough. A shorter max length leaves more KV cache for concurrent requests.
- `--gpu-memory-utilization`: 0.90, lowered to 0.80 on the T4 because the embedding model shares the GPU.
- `--quantization awq --dtype float16`: the AWQ kernels want fp16. The T4 doesn't support bf16 anyway.
- `--tensor-parallel-size 1`: the model fits on one GPU, and splitting it would only add communication overhead.
- `--enforce-eager`: T4 only. It skips CUDA graph capture, which saves memory and startup time at the cost of some speed.

**Chunking.** 800 characters (about 200 tokens) with 150 overlap, chunked per page, so every chunk maps to exactly one page and the citations stay correct. Larger chunks made the citations vaguer.

**Retrieval.** Dense search alone missed exact terms like "PagedAttention" or "RRF", and BM25 alone missed paraphrased questions, so I use both and merge them with Reciprocal Rank Fusion (k=60). RRF works on ranks, so I don't have to normalise the two score scales. For bge I add its query instruction prefix.

**Agent.** Built as a small LangGraph graph. Each chunk gets a yes/no relevance check. If nothing passes, the question is rewritten and retrieval runs again, at most twice. After the answer is generated, another yes/no call checks that it's supported by the context. If that fails, the user gets the not-found message.

**Images.** I did both options. The main one is captioning figures at ingest time (Option 1), because then diagram questions go through the normal `/ask` flow and get page citations. `/ask-image` (Option 2) was only a few lines on top, since it uses the same model. Large images are downscaled to 768 px first; before that, some figures went over the 4096 context.

## Benchmark

`python scripts/benchmark.py --concurrency 32`: 32 parallel requests, 256 max output tokens, streaming.

| Run | Output tok/s | TTFT mean / P95 | Latency mean / P95 |
|---|---|---|---|
| Colab T4, cold server | 231.8 | 9.56 / 9.63 s | 21.0 / 22.3 s |
| Kaggle T4, cold server | 271.5 | 8.71 / 8.75 s | 18.1 / 19.2 s |
| Kaggle T4, warm server (second run) | 580.5 | 0.37 / 0.38 s | 8.8 / 8.8 s |

The cold numbers are the realistic ones. TTFT is high there because all 32 prompts get prefilled together, and the T4 can't use FlashAttention-2 (that needs compute capability 8.0+). The warm run is much faster partly because the kernels were already compiled, and partly because every request uses the same prompt, so prefix caching kicks in. I'd expect an L4/A10 with CUDA graphs to do a lot better.

## Testing

`scripts/run_checks.py` checks each path and prints PASS/FAIL. The latest Kaggle run passed all 14 checks:

- vLLM up, Qdrant up, ingestion (1628 chunks, 29 of them figure descriptions), `/health`
- normal question: `router:document -> retrieve -> grade:4_relevant -> answer -> verify:pass`
- figure question: sources include the figure chunk from `attention_is_all_you_need.pdf` p.3
- "Hi there!": `router:chitchat -> chitchat`
- "Who won the 2022 World Cup?": `router:out_of_scope -> fallback`
- "What learning rate did the LoRA paper use to fine-tune GPT-5?": gets as far as answering, then `verify:fail -> fallback`
- "What quantum error-correction code does the ReAct paper propose?": nothing relevant, so `rewrite` twice and then `fallback`
- empty question is rejected with a 400, `/ask-image` works, and the benchmark runs

## Evaluation

`eval/eval.py` sends 13 questions (12 about the papers, 1 off-topic) through `/ask`, and Gemini scores the answers for faithfulness and relevance (1-5). Gemini is only the judge. The answers themselves come from the vLLM model, as the assignment requires. I split it into two steps so the answers can be collected on the GPU machine and judged on my laptop, which is where the API key lives:

```bash
python eval/eval.py --collect eval/answers.json   # needs the API running
python eval/eval.py --judge   eval/answers.json   # needs GEMINI_API_KEY
```

The result was a mean faithfulness of **4.62/5** and a mean relevance of **4.69/5**. The raw answers and scores are in `eval/answers.json` and `eval/results.json`.

Most answers got 5/5. The weak ones:

- **BERT pre-training tasks** (faithfulness 2): it answered "left-to-right LM + NSP", taken from the ablation section on p.14. The correct answer is masked LM + NSP. The retrieved chunk really does say LTR + NSP, so the verifier passed it. This is the main weakness of my verify step: it checks that the answer matches what was retrieved, not that the right chunk was retrieved.
- **LoRA latency** (faithfulness 3): the main point is right (the merged W = W0 + BA adds no extra step at inference), but it added a sentence about gradients that isn't in the paper.
- **DPR encoders** (relevance 2): it only described the passage encoder and didn't say that DPR uses two separate BERT encoders.
- **Transformer attention heads**: it said "not found", but the answer (8 heads) is in the paper. The judge scored it 5 because it didn't hallucinate, but really this is a retrieval miss.

A cross-encoder reranker would probably fix the BERT and attention-heads cases.

## Example output

From the Colab run:

```jsonc
// normal question
{"answer": "PagedAttention solves the problem of memory challenges in serving large language models ... by allowing continuous keys and values to be stored in non-contiguous memory space ... [Source: vllm_pagedattention.pdf, page 2].",
 "agent_path": ["router:document", "retrieve", "grade:4_relevant", "answer", "verify:pass"]}

// figure question, answered from the figure description
{"answer": "The Softmax layer sits on top of the decoder stack. [Source: attention_is_all_you_need.pdf, page 3].",
 "sources": [{"source": "attention_is_all_you_need.pdf", "page": 3, "type": "figure"}],
 "agent_path": ["router:document", "retrieve", "grade:1_relevant", "answer", "verify:pass"]}

// off-topic
{"answer": "I couldn't find this in the provided documents.", "sources": [],
 "agent_path": ["router:out_of_scope", "fallback"]}
```

## Known issues

- The per-chunk grading costs up to 5 extra LLM calls per question, which adds about 5-10 s on a T4.
- The verify step can't catch an answer that is wrong but supported by the retrieved text (the BERT case).
- Figure descriptions are regenerated on every ingest and vary between runs. The "top of the decoder stack" question worked on Colab and got "not found" on Kaggle. Caching the captions would fix this.
- Only embedded raster images are captioned. Vector diagrams are missed.
- Chunking by character count can cut tables in half.
- The BM25 index is a pickle that gets rebuilt from scratch each time.
- Yes/no judgements from a 7B model are sometimes too strict and sometimes too loose.

## What I'd do next

- Replace the per-chunk LLM grading with a reranker (bge-reranker-v2-m3), or at least grade all chunks in one call.
- Use layout-aware parsing (docling) and render full pages, so vector figures and tables are handled.
- Cache the figure captions so ingestion is deterministic.
- Add streaming (SSE) and a small Gradio UI.
- Benchmark with varied prompts, so that prefix caching doesn't inflate the numbers.
