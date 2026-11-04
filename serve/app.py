"""Robot Ops Copilot service.

  OLLAMA_HOST=127.0.0.1:11435 uvicorn serve.app:app --port 8100

POST /ask        {"question": "..."} -> answer, cited sources (document and page), tool calls, guard flags
GET  /audit      the last N audit records
GET  /health
GET  /           a small page to ask questions

Every answer is appended to an audit log (data/logs/audit.jsonl): time, request id, question, model and retrieval
settings, every tool call, the ids of the passages the model saw, the answer, guard flags and latency. That record
is what makes an answer traceable after the fact: which page of which manual it came from, and which model wrote it.
Everything runs locally; no question or document leaves the machine.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from copilot.config import CHAT_MODEL, DATA

AUDIT = DATA / "logs" / "audit.jsonl"
RERANK = os.environ.get("COPILOT_RERANK_SIZE", "small")
RERANKERS = {"small": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "large": "BAAI/bge-reranker-v2-m3"}

app = FastAPI(title="Robot Ops Copilot")
_lock = threading.Lock()
_copilot = None


def copilot():
    global _copilot
    if _copilot is None:
        from copilot.agent import Copilot
        from copilot.retrieve import Retriever
        r = Retriever()
        mode = "hybrid"
        if RERANK in RERANKERS:
            r.set_reranker(RERANKERS[RERANK])
            mode = "hybrid+rerank"
        _copilot = Copilot(r, os.environ.get("COPILOT_CHAT", CHAT_MODEL), mode=mode,
                           pipeline=os.environ.get("COPILOT_PIPELINE", "agent"))
    return _copilot


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok", "model": os.environ.get("COPILOT_CHAT", CHAT_MODEL), "reranker": RERANK}


@app.post("/ask")
def ask(req: Ask):
    rid = uuid.uuid4().hex[:12]
    with _lock:  # one GPU, one model: answer one question at a time
        try:
            res = copilot().ask(req.question)
        except Exception as e:  # noqa: BLE001
            _audit({"request_id": rid, "question": req.question, "error": str(e)})
            raise HTTPException(503, f"copilot unavailable: {e}")
    rec = {"request_id": rid, **res}
    _audit(rec)
    cited = [s for s in res["sources"] if s["n"] in res["cited"]]
    return {"request_id": rid, "answer": res["answer"], "refused": res["refused"], "sources": cited,
            "tool_calls": res["tool_calls"], "flags": res["flags"], "latency_s": res["latency_s"]}


@app.get("/audit")
def audit(n: int = 20):
    if not AUDIT.exists():
        return []
    return [json.loads(ln) for ln in AUDIT.read_text().splitlines()[-n:]]


def _audit(rec: dict) -> None:
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **rec}
    with open(AUDIT, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>Robot Ops Copilot</title><style>
:root{--bg:#f6f7f9;--fg:#1c2026;--muted:#5d6672;--line:#d5dae1;--accent:#1f5fbf;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--fg:#e6e8eb;--muted:#9aa3ad;--line:#2c3138;--accent:#7fb0ff;--card:#1a1d21}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 16px}
form{display:flex;gap:8px}input{flex:1;min-width:0;padding:10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);font:inherit}
button{padding:10px 16px;border:0;border-radius:6px;background:var(--accent);color:#fff;font:inherit;cursor:pointer}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#out{margin-top:20px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;white-space:pre-wrap;display:none}
#src{color:var(--muted);font-size:14px;margin-top:12px}</style></head><body><main>
<h1>Robot Ops Copilot</h1><p class="sub">UR3e manuals, UR error codes, ROS 2 / MoveIt docs and thesis logs. Local model; answers cite their pages.</p>
<form id="f"><input id="q" placeholder="e.g. What does C153A1 mean? / Wie aktiviere ich Freedrive?" autocomplete="off"><button>Ask</button></form>
<div id="out"><div id="a"></div><div id="src"></div></div></main><script>
const $=id=>document.getElementById(id),f=$('f'),q=$('q'),out=$('out'),a=$('a'),src=$('src');
f.onsubmit=async e=>{e.preventDefault();out.style.display='block';a.textContent='Thinking...';src.textContent='';
const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q.value})});
const j=await r.json();a.textContent=j.answer||j.detail;src.textContent=(j.sources||[]).map(s=>`[${s.n}] ${s.doc}${s.doc.endsWith('.pdf')?' p.'+s.page:''}: ${s.title}`).join('\\n')+`\\n${j.latency_s||''} s`;};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def page():
    return PAGE
