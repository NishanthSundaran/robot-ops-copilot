"""Hybrid retrieval: BM25 + dense (bge-m3, multilingual, via Ollama), fused with reciprocal rank fusion,
optionally reranked by a cross-encoder (bge-reranker-v2-m3).

  python -m copilot.retrieve --build          # embed every chunk once (data/index/dense.npy)
  python -m copilot.retrieve "C153A1 shoulder" # try a query
"""

from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import requests
from rank_bm25 import BM25Okapi

from copilot.config import EMBED_MODEL, INDEX, OLLAMA, RERANK_MODEL

TOKEN = re.compile(r"\w+", re.UNICODE)
MODES = ("bm25", "dense", "hybrid", "hybrid+rerank")


def tokenize(s: str) -> list[str]:
    return TOKEN.findall(s.lower())


def embed(texts: list[str], batch: int = 32, gpu: bool = True) -> np.ndarray:
    """gpu=False for queries: one short text embeds fast on the CPU and leaves the 6 GB GPU to the LLM."""
    out = []
    opts = {} if gpu else {"num_gpu": 0}
    for i in range(0, len(texts), batch):
        r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts[i:i + batch],
                                                       "keep_alive": "30m", "options": opts}, timeout=600)
        r.raise_for_status()
        out.append(np.asarray(r.json()["embeddings"], dtype=np.float32))
    v = np.concatenate(out)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def chunk_text(c: dict) -> str:
    """What gets indexed: the section title carries the context a 200-word window loses."""
    return f"{c['title']}\n{c['text']}"


class Retriever:
    def __init__(self, rerank: bool = True):
        self.chunks = [json.loads(ln) for ln in open(INDEX / "chunks.jsonl")]
        self.bm25 = BM25Okapi([tokenize(chunk_text(c)) for c in self.chunks])
        self.dense = np.load(INDEX / "dense.npy")
        assert len(self.dense) == len(self.chunks), "dense index is stale: python -m copilot.retrieve --build"
        self._reranker = None
        self.rerank_model = RERANK_MODEL
        self.want_rerank = rerank

    def set_reranker(self, name: str) -> None:
        self.rerank_model, self._reranker = name, None

    @property
    def reranker(self):
        if self._reranker is None:
            import os
            import torch
            from sentence_transformers import CrossEncoder
            # CPU by default: next to a 7B model the 6 GB GPU has no room for it
            dev = os.environ.get("COPILOT_RERANK_DEVICE", "cpu")
            self._reranker = CrossEncoder(self.rerank_model, device=dev, max_length=512,
                                          model_kwargs={"torch_dtype": torch.float16} if dev == "cuda" else {})
        return self._reranker

    def _filter(self, sources):
        if not sources:
            return np.ones(len(self.chunks), bool)
        return np.array([c["source"] in sources for c in self.chunks])

    def search(self, query: str, k: int = 6, mode: str = "hybrid+rerank", sources=None, pool: int = 30):
        """-> list of (chunk, score). Scores are only comparable within one mode."""
        mask = self._filter(sources)
        ranks = []
        if mode in ("bm25", "hybrid", "hybrid+rerank"):
            s = np.where(mask, self.bm25.get_scores(tokenize(query)), -np.inf)
            ranks.append(np.argsort(-s)[:pool * 2])
        if mode in ("dense", "hybrid", "hybrid+rerank"):
            s = np.where(mask, self.dense @ embed([query], gpu=False)[0], -np.inf)
            ranks.append(np.argsort(-s)[:pool * 2])
        if len(ranks) == 1:
            order = list(ranks[0])
            scores = {i: 1.0 / (60 + r) for r, i in enumerate(order)}
        else:  # reciprocal rank fusion, k=60
            scores = {}
            for rk in ranks:
                for r, i in enumerate(rk):
                    scores[i] = scores.get(i, 0.0) + 1.0 / (60 + r)
            order = sorted(scores, key=scores.get, reverse=True)
        order = [i for i in order if mask[i]]
        if mode == "hybrid+rerank":
            cand = order[:pool]
            rs = self.reranker.predict([(query, chunk_text(self.chunks[i])) for i in cand], batch_size=16)
            best = np.argsort(-np.asarray(rs))[:k]
            return [(self.chunks[cand[j]], float(rs[j])) for j in best]
        return [(self.chunks[i], float(scores[i])) for i in order[:k]]


def build() -> None:
    chunks = [json.loads(ln) for ln in open(INDEX / "chunks.jsonl")]
    t = time.time()
    v = embed([chunk_text(c) for c in chunks])
    np.save(INDEX / "dense.npy", v)
    print(f"embedded {len(v)} chunks, dim {v.shape[1]}, {time.time() - t:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--mode", default="hybrid+rerank", choices=MODES)
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    if a.build:
        build()
    if a.query:
        r = Retriever()
        t = time.time()
        hits = r.search(a.query, a.k, a.mode)
        print(f"{(time.time() - t) * 1000:.0f} ms")
        for c, s in hits:
            print(f"{s:8.3f}  {c['id']:55.55s} {c['title'][:40]:40s} {c['text'][:90]}")


if __name__ == "__main__":
    main()
