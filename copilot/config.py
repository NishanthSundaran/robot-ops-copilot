"""Paths and model names. Everything large lives under DATA (a symlink to the data drive)."""

import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(os.environ.get("COPILOT_DATA", REPO / "data"))
CORPUS = DATA / "corpus"
INDEX = DATA / "index"
os.environ.setdefault("HF_HOME", str(DATA / "cache" / "hf"))  # the root disk is full; models go to DATA
THESIS_WS = pathlib.Path(os.environ.get("THESIS_WS", pathlib.Path.home() / "ros2_ws"))

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11435")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA
EMBED_MODEL = os.environ.get("COPILOT_EMBED", "bge-m3")
CHAT_MODEL = os.environ.get("COPILOT_CHAT", "qwen2.5:7b")
RERANK_MODEL = os.environ.get("COPILOT_RERANK", "BAAI/bge-reranker-v2-m3")
