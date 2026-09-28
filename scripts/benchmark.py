"""Concurrent load test against vLLM: reports TTFT, P95 latency and output tokens/sec.

python scripts/benchmark.py --concurrency 32
"""
import argparse
import asyncio
import os
import statistics
import time

from openai import AsyncOpenAI

BASE = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
PROMPT = "Explain retrieval-augmented generation and its benefits in about 150 words."


async def one(client):
    start = time.perf_counter()
    ttft, tokens = None, 0
    stream = await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": PROMPT}], max_tokens=256,
        stream=True, stream_options={"include_usage": True})
    async for ev in stream:
        if ev.choices and ev.choices[0].delta.content and ttft is None:
            ttft = time.perf_counter() - start
        if ev.usage:
            tokens = ev.usage.completion_tokens
    return ttft, time.perf_counter() - start, tokens


async def main(n):
    client = AsyncOpenAI(base_url=BASE, api_key="x")
    t0 = time.perf_counter()
    results = await asyncio.gather(*[one(client) for _ in range(n)])
    wall = time.perf_counter() - t0
    ttfts = sorted(r[0] for r in results)
    lats = sorted(r[1] for r in results)
    toks = sum(r[2] for r in results)
    p95 = lambda xs: xs[int(0.95 * (len(xs) - 1))]
    print(f"Requests: {n}  | wall: {wall:.2f}s")
    print(f"Output throughput: {toks / wall:.1f} tokens/s")
    print(f"TTFT mean/P95: {statistics.mean(ttfts):.3f}s / {p95(ttfts):.3f}s")
    print(f"Latency mean/P95: {statistics.mean(lats):.2f}s / {p95(lats):.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=32)
    asyncio.run(main(ap.parse_args().concurrency))
