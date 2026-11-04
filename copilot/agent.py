"""The copilot: a local LLM (Ollama) that answers only from what its tools return, with citations.

Two pipelines, same retriever and prompt rules, so the evaluation can compare them:
  rag    retrieve top-k passages for the question, one LLM call
  agent  the LLM calls tools (search, error code lookup, SQL over trial logs, parameters) for up to
         MAX_STEPS rounds, then answers

Guardrails, applied to both:
  - every factual sentence must cite a source [n] that was actually returned; citations to numbers that
    were never returned are removed and flagged
  - an answer with no valid citation is retried once, then replaced by a refusal
  - if the model answers without calling any tool, the question is searched for it (retrieval first)

  python -m copilot.agent "What does C153A1 mean?"
"""

from __future__ import annotations

import argparse
import json
import re
import time

import requests

from copilot.config import CHAT_MODEL, OLLAMA
from copilot.tools import SPECS, Tools

MAX_STEPS = 4
REFUSAL_EN = "I could not find this in the documentation I have."
REFUSAL_DE = "Das konnte ich in der vorliegenden Dokumentation nicht finden."
REFUSAL_RE = re.compile(r"(could not find|couldn't find|not (?:in|covered by) the (?:documentation|sources)|"
                        r"no information|nicht finden|keine informationen|nicht in der dokumentation)", re.I)
CITE = re.compile(r"\[(\d+)\]")

SYSTEM = """You are a support copilot for a Universal Robots UR3e cell running ROS 2 Humble and MoveIt 2.
Answer only from the numbered sources your tools return. Rules:
- Use the tools first. For an error code like C153A1 use lookup_error_code. For numbers about the logged thesis
  trials use query_trials with SQL. For a parameter of the thesis stack use get_parameter. Otherwise search_docs.
- After each fact, cite its source number in square brackets, like [2]. Cite only numbers you were given.
- If the sources do not contain the answer, say exactly: "{refusal}" Do not guess and do not use outside knowledge.
- The question is in {lang}. Write the whole answer in {lang}. Be short and concrete: steps as a list.
"""


def normalize_citations(a: str) -> str:
    """[1, 2] and [1,2] -> [1][2], so every citation is one bracket the guard can check."""
    return re.sub(r"\[(\d+(?:\s*,\s*\d+)+)\]", lambda m: "".join(f"[{x.strip()}]" for x in m.group(1).split(",")), a)


def is_german(q: str) -> bool:
    words = set(re.findall(r"\w+", q.lower()))
    return len(words & {"wie", "was", "der", "die", "das", "ich", "ist", "welche", "warum", "kann", "nicht", "und",
                        "beim", "wird", "mit", "eine", "einen", "bedeutet", "wo", "wann", "soll"}) >= 2


def chat(messages, model, tools=None, num_ctx=8192):
    body = {"model": model, "messages": messages, "stream": False, "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx}}
    if tools:
        body["tools"] = tools
    r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    return r.json()


class Copilot:
    def __init__(self, retriever, model: str = CHAT_MODEL, mode: str = "hybrid+rerank", pipeline: str = "agent",
                 k: int = 5):
        self.tools = Tools(retriever, mode, k)
        self.model, self.pipeline, self.k = model, pipeline, k

    def _format(self, note, srcs, sources):
        lines = [note] if note else []
        for s in srcs:
            sources.append(s)
            where = s["doc"] + (f", page {s['page']}" if s["doc"].endswith(".pdf") else "")
            lines.append(f"[{len(sources)}] ({where}; {s['title']}) {s['text']}")
        return "\n".join(lines) or "No results."

    def ask(self, question: str) -> dict:
        t0 = time.time()
        de = is_german(question)
        refusal = REFUSAL_DE if de else REFUSAL_EN
        msgs = [{"role": "system", "content": SYSTEM.format(refusal=refusal, lang="German" if de else "English")}]
        sources, calls, flags = [], [], []
        if self.pipeline == "rag":
            note, srcs = self.tools.search_docs(question)
            ctx = self._format(note, srcs, sources)
            calls.append({"tool": "search_docs", "args": {"query": question}})
            msgs.append({"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {question}"})
            answer = chat(msgs, self.model)["message"]["content"]
        else:
            msgs.append({"role": "user", "content": question})
            answer = ""
            for step in range(MAX_STEPS + 1):
                m = chat(msgs, self.model, SPECS if step < MAX_STEPS else None)["message"]
                tcs = m.get("tool_calls") or []
                if not tcs:
                    answer = m.get("content", "")
                    if not sources and step == 0:  # answered from memory: search for it and ask again
                        flags.append("forced_retrieval")
                        note, srcs = self.tools.search_docs(question)
                        calls.append({"tool": "search_docs", "args": {"query": question}, "forced": True})
                        msgs.append({"role": "assistant", "content": answer})
                        msgs.append({"role": "user", "content": "Answer only from these sources, with citations:\n"
                                     + self._format(note, srcs, sources)})
                        continue
                    break
                msgs.append({"role": "assistant", "content": m.get("content", ""), "tool_calls": tcs})
                for tc in tcs:
                    fn = tc["function"]
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    note, srcs = self.tools.call(fn["name"], args)
                    calls.append({"tool": fn["name"], "args": args, "n_sources": len(srcs)})
                    msgs.append({"role": "tool", "content": self._format(note, srcs, sources)})
        answer, flags = self._guard(answer, sources, msgs, refusal, flags)
        cited = sorted({int(n) for n in CITE.findall(answer)})
        return {"question": question, "answer": answer, "model": self.model, "pipeline": self.pipeline,
                "retrieval": self.tools.mode, "tool_calls": calls, "flags": flags,
                "sources": [dict(n=i + 1, **{k: s[k] for k in ("doc", "page", "title", "id")})
                            for i, s in enumerate(sources)],
                "cited": cited, "refused": bool(REFUSAL_RE.search(answer)),
                "latency_s": round(time.time() - t0, 2)}

    def _guard(self, answer, sources, msgs, refusal, flags):
        n = len(sources)
        answer = normalize_citations(answer)
        bad = [int(x) for x in CITE.findall(answer) if not 1 <= int(x) <= n]
        if bad:
            flags.append(f"invalid_citations:{sorted(set(bad))}")
            answer = CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n else "", answer)
        if CITE.search(answer) or REFUSAL_RE.search(answer):
            return answer.strip(), flags
        flags.append("uncited_retry")
        msgs.append({"role": "assistant", "content": answer})
        msgs.append({"role": "user", "content": "Rewrite your answer so every fact cites a source number like [1]. "
                     f"If the sources do not answer the question, reply exactly: {refusal}"})
        answer = normalize_citations(chat(msgs, self.model)["message"]["content"])
        answer = CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n else "", answer)
        if not (CITE.search(answer) or REFUSAL_RE.search(answer)):
            flags.append("uncited_refused")
            answer = refusal
        return answer.strip(), flags


def main() -> None:
    from copilot.retrieve import Retriever
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--pipeline", default="agent", choices=["agent", "rag"])
    ap.add_argument("--model", default=CHAT_MODEL)
    a = ap.parse_args()
    r = Copilot(Retriever(), a.model, pipeline=a.pipeline).ask(a.question)
    print(r["answer"], "\n")
    for s in r["sources"]:
        print(f"  [{s['n']}] {s['doc']} p{s['page']} {s['title'][:50]}{'  (cited)' if s['n'] in r['cited'] else ''}")
    print(" ", r["tool_calls"], r["flags"], r["latency_s"], "s")


if __name__ == "__main__":
    main()
