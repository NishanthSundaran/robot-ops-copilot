"""The copilot: a local LLM (Ollama) that answers only from what its tools return, with citations.

Two pipelines, same retriever and prompt rules, so the evaluation can compare them:
  rag    retrieve top-k passages for the question, one LLM call
  agent  the same first retrieval, then the LLM may call tools (search, error code lookup, SQL over the trial
         logs, parameters) for up to MAX_STEPS rounds before it answers

What the first evaluation round taught, and what is enforced here:
  - retrieval first: the model's own search queries were often worse than the question ("voltage range"), so
    the question itself is always searched and its passages come with it
  - tool gating: lookup_error_code runs only for a code that is in the question (the model used the prompt's
    example code for symptom questions); a parameter miss falls back to searching
  - relevance gate: if the best passage for the question scores below RELEVANCE and no tool returned data, the
    answer is a refusal (most wrong answers to unanswerable questions came from irrelevant passages)
  - citations: every fact cites a source [n] that was returned; invented numbers are removed and flagged, an
    answer with no valid citation is retried once, then replaced by a refusal
  - a refusal is an answer that says so, not any answer that mentions "no information" in passing

  python -m copilot.agent "What does C153A1 mean?"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import requests

from copilot.config import CHAT_MODEL, OLLAMA
from copilot.tools import SPECS, Tools

MAX_STEPS = 3
PRODUCT_RULE = os.environ.get("COPILOT_PRODUCT_RULE", "1") == "1"
# optional relevance gate: refuse when the best passage's reranker logit is below this. Chosen on the dev split by
# replaying a gate-off run (eval/gate_offline.py); from cycle 4 on dev picks "off", so it is off by default
RELEVANCE = float(os.environ.get("COPILOT_RELEVANCE", "-99"))
REFUSAL_EN = "I could not find this in the documentation I have."
REFUSAL_DE = "Das konnte ich in der vorliegenden Dokumentation nicht finden."
REFUSAL_RE = re.compile(r"^(i could not find|i couldn't find|i can't find|i cannot find|the (documentation|sources?) "
                        r"(do(es)? not|don't)|das (konnte|kann) ich|ich (konnte|kann) .{0,40}nicht finden|in der .{0,30}dokumentation "
                        r".{0,20}nicht)", re.I)
CITE = re.compile(r"\[(\d+)\]")
CODE = re.compile(r"\bC\d{1,3}(?:A\d{1,4})?\b", re.I)
# cycle 5: asked about another robot, the 7B answered with the UR3e's value ("the UR3e is IP54") or the UR3e trial
# count. The copilot covers one UR3e cell, so a question naming another robot (and not the UR3e) is out of scope.
OTHER_ROBOT = re.compile(r"\b(ur(?:5|10|16|20|30)e?|cb3|kuka|iiwa|franka|panda|abb|yumi|fanuc|yaskawa|kinova|doosan|"
                         r"techman)\b", re.I)
BRACKET = re.compile(r"\[([^\]\n]{0,80})\]")
# intent router: questions about the logged thesis trials are answered by SQL; documentation passages only
# distracted the model there (cycle 3: "after GUIDING the controller goes to RUNNING", from a ROS tutorial)
TRIAL_SCHEMA = (
    "Tables: trials(trial_id, trial_date 'YYYY-MM-DD', total_time_s, grasp_success 1/0, num_guide_cycles, "
    "peak_force_N, mean_force_N, double_tap_time_ms, displacement_m, phase_INIT_s, phase_PERCEIVE_s, "
    "phase_PLAN_GRASP_s, phase_MOVE_TO_PREGRASP_s, phase_MOVE_TO_GRASP_s, phase_GRASP_s, phase_LIFT_s, "
    "phase_ENABLE_SERVO_s, phase_AWAIT_INTERACTION_s, phase_PREP_PLACE_s, phase_DESCEND_TO_PLACE_s, phase_PLACE_s, "
    "phase_RETRACT_s) -- one row per trial, every phase_*_s column is in this table; "
    "guide_cycles(trial_id, cycle, duration_s, peak_force_N, mean_force_N, peak_velocity_mps, latency_ms); "
    "state_transitions(trial_id, time_s, from_state, to_state) -- controller states IDLE, NUDGE, GUIDING.\n"
    "Examples: how many succeeded = SELECT COUNT(*) FROM trials WHERE grasp_success=1; success rate = "
    "SELECT AVG(grasp_success) FROM trials.\n"
    "The logs record only these columns, all from one UR3e setup. If the question asks about something they do "
    "not record (another robot, a gripper setting, a trial id that returns no rows), refuse.")
TRIALS = re.compile(r"\b(trials?|versuch\w*)\b", re.I)
STATES = re.compile(r"\b(guiding|nudge|idle)\b", re.I)
STATE_Q = re.compile(r"\b(state|states|transition\w*|zustand\w*|go(es)? to|after|nach)\b", re.I)

SYSTEM = """You are a support copilot for a Universal Robots UR3e cell running ROS 2 Humble and MoveIt 2.
You answer only from numbered sources. Passages found for the question are given with it.
Rules:
- If those passages answer the question, answer from them. If they do not, call a tool before giving up:
  * lookup_error_code only for an error code written in the question,
  * query_trials (SQL) for anything about the logged thesis trials; do the arithmetic in SQL (COUNT, AVG,
    MAX, ORDER BY), never by hand,
  * get_parameter for a value in the thesis's own ROS 2 config files,
  * search_docs with a better query or a source filter for anything else.
- End each sentence with the number of the source it comes from in square brackets, for example "... [3]."
  Cite only numbers you were given. Do not put citations at the start of the answer.
- For an error code, first say what the code means, then give the steps the source lists.
- Give procedures step by step with the exact actions the source describes; do not invent steps.
- If the sources do not answer the question{product_rule}, reply exactly: "{refusal}"
  Do not guess and do not use outside knowledge.
- The question is in {lang}. Write the whole answer in {lang}. Be short and concrete."""


def normalize_citations(a: str) -> str:
    """[1, 2], [1,2] and [1-3] -> [1][2]([3]), so every citation is one bracket the guard can check."""
    def rng(m):
        a_, b_ = int(m.group(1)), int(m.group(2))
        return "".join(f"[{i}]" for i in range(a_, b_ + 1)) if 0 < b_ - a_ < 10 else m.group(0)
    a = re.sub(r"\[(\d+)\s*[-–]\s*(\d+)\]", rng, a)
    return re.sub(r"\[(\d+(?:\s*,\s*\d+)+)\]", lambda m: "".join(f"[{x.strip()}]" for x in m.group(1).split(",")), a)


def is_german(q: str) -> bool:
    words = set(re.findall(r"\w+", q.lower()))
    return len(words & {"wie", "was", "der", "die", "das", "ich", "ist", "welche", "welcher", "warum", "kann", "nicht",
                        "und", "beim", "wird", "mit", "eine", "einen", "bedeutet", "wo", "wann", "soll", "für", "meine",
                        "viele", "lange", "darf", "zu", "im", "den", "dem"}) >= 2


def is_refusal(answer: str) -> bool:
    body = CITE.sub("", answer).strip().strip("*").strip()
    return bool(REFUSAL_RE.match(body))


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
                 k: int = 5, gate: bool = True):
        self.tools = Tools(retriever, mode, k)
        self.model, self.pipeline, self.k = model, pipeline, k
        self.gate = gate and mode == "hybrid+rerank"  # the threshold is on the reranker's scale

    def _format(self, note, srcs, sources):
        lines = [note] if note else []
        for s in srcs:
            sources.append(s)
            where = s["doc"] + (f", page {s['page']}" if s["doc"].endswith(".pdf") else "")
            # cycle 5: "Range: [100 - 3000]" in a passage was read as a citation marker; brackets in source text
            # become parentheses so [n] only ever means a source
            text = BRACKET.sub(r"(\1)", s["text"])
            lines.append(f"[{len(sources)}] ({where}; {s['title']}) {text}")
        return "\n".join(lines) or "No results."

    def _call(self, name, args, question, calls):
        """Run one tool call, with the gates the first evaluation round showed are needed."""
        if name == "lookup_error_code":
            code = str(args.get("code", "")).upper().replace(" ", "")
            in_q = {c.upper() for c in CODE.findall(question)}
            if code not in in_q:
                calls.append({"tool": name, "args": args, "gated": "code not in question"})
                note, srcs = self.tools.search_docs(question, "ur_errors")
                return ("The question contains no such error code; searched the error directory for the "
                        "question instead:\n" + note).strip(), srcs, "search_docs"
        note, srcs = self.tools.call(name, args)
        if name == "get_parameter" and not srcs:
            more_note, more = self.tools.search_docs(question, "thesis")
            return note + " Passages from the thesis config files:", more, "search_docs"
        return note, srcs, name

    def ask(self, question: str) -> dict:
        t0 = time.time()
        de = is_german(question)
        refusal = REFUSAL_DE if de else REFUSAL_EN
        rule = (", or they are about a different robot, product or version than the one asked about"
                if PRODUCT_RULE else "")
        msgs = [{"role": "system", "content": SYSTEM.format(refusal=refusal, lang="German" if de else "English",
                                                           product_rule=rule)}]
        sources, calls, flags, data_tools = [], [], [], 0
        if PRODUCT_RULE and OTHER_ROBOT.search(question) and not re.search(r"\bur3e?\b", question, re.I):
            return {"question": question, "answer": refusal, "model": self.model, "pipeline": self.pipeline,
                    "retrieval": self.tools.mode, "tool_calls": [], "flags": ["other_robot"], "sources": [],
                    "cited": [], "refused": True, "top_score": -99.0, "data_tools": 0,
                    "latency_s": round(time.time() - t0, 2)}
        trials = self.pipeline == "agent" and bool(TRIALS.search(question) or
                                                   (STATES.search(question) and STATE_Q.search(question)))
        if trials:
            flags.append("routed_to_sql")
            top = -99.0
            msgs.append({"role": "user", "content": f"Question: {question}\n\nThis question is about the logged thesis "
                         "trials. Call query_trials with one SQL query that computes the answer, then answer from its "
                         "result.\n" + TRIAL_SCHEMA})
        else:
            note, first = self.tools.search_docs(question)
            top = max((s["score"] for s in first), default=-99.0)
            calls.append({"tool": "search_docs", "args": {"query": question}, "first": True,
                          "top_score": round(top, 3)})
            ctx = self._format(note, first, sources)
            msgs.append({"role": "user", "content": f"Question: {question}\n\nPassages found for the question:\n{ctx}"})
        if self.pipeline == "rag":
            answer = chat(msgs, self.model)["message"]["content"]
        else:
            answer = ""
            nudged = False
            for step in range(MAX_STEPS + 1):
                # staged: the first answer is written without tool specs in the prompt (cycle 3: offering tools made
                # the 7B model refuse questions its passages answered); tools come after a refusal, or at once
                # for questions routed to the trial logs
                offer = (step > 0 or trials) and step < MAX_STEPS
                m = chat(msgs, self.model, SPECS if offer else None)["message"]
                tcs = m.get("tool_calls") or []
                if not tcs:
                    answer = m.get("content", "")
                    # cycle 2: with passages in hand the model refused trial-log questions instead of running SQL
                    if is_refusal(answer) and not nudged and step < MAX_STEPS:
                        nudged = True
                        flags.append("nudged_to_tools")
                        msgs.append({"role": "assistant", "content": answer})
                        msgs.append({"role": "user", "content": "Before refusing, try a tool: query_trials for the "
                                     "logged thesis trials, get_parameter for a thesis config value, lookup_error_code "
                                     "for an error code in the question, or search_docs with a different query. If a tool "
                                     "returned an error, correct the call and run it again."})
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
                    note, srcs, used = self._call(fn["name"], args, question, calls)
                    if used in ("query_trials", "lookup_error_code", "get_parameter") and srcs:
                        data_tools += 1
                    calls.append({"tool": fn["name"], "args": args, "n_sources": len(srcs), "ran": used})
                    msgs.append({"role": "tool", "content": self._format(note, srcs, sources)})
        if self.gate and not trials and top < RELEVANCE and not data_tools and not CODE.search(question):
            flags.append(f"relevance_gate:{top:.2f}")
            answer = refusal
        answer, flags = self._guard(answer, sources, msgs, refusal, flags)
        cited = sorted({int(n) for n in CITE.findall(answer)})
        return {"question": question, "answer": answer, "model": self.model, "pipeline": self.pipeline,
                "retrieval": self.tools.mode, "tool_calls": calls, "flags": flags,
                "sources": [dict(n=i + 1, **{k: s[k] for k in ("doc", "page", "title", "id")})
                            for i, s in enumerate(sources)],
                "cited": cited, "refused": is_refusal(answer), "top_score": round(top, 3), "data_tools": data_tools,
                "latency_s": round(time.time() - t0, 2)}

    def _guard(self, answer, sources, msgs, refusal, flags):
        n = len(sources)
        answer = normalize_citations(answer)
        bad = [int(x) for x in CITE.findall(answer) if not 1 <= int(x) <= n]
        if bad:
            flags.append(f"invalid_citations:{sorted(set(bad))}")
            answer = CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n else "", answer)
        if is_refusal(answer):
            return refusal, flags
        empty = len(re.findall(r"\w", CITE.sub("", answer))) < 3
        if CITE.search(answer) and not empty:
            return answer.strip(), flags
        msgs.append({"role": "assistant", "content": answer})
        if empty:
            # cycle 5: the 3B model often answered with a bare "[1]"
            flags.append("empty_retry")
            msgs.append({"role": "user", "content": "A citation alone is not an answer. Answer the question in "
                         "sentences with the facts from the sources, and add the source number after each fact."})
        else:
            flags.append("uncited_retry")
            # cycle 2: offering the refusal here made the model drop correct answers; ask for citations only
            msgs.append({"role": "user", "content": "Rewrite the same answer and add, after each fact, the number "
                         "of the source it comes from, like [1]."})
        answer = normalize_citations(chat(msgs, self.model)["message"]["content"])
        answer = CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n else "", answer)
        if is_refusal(answer):
            return refusal, flags
        if not CITE.search(answer) or len(re.findall(r"\w", CITE.sub("", answer))) < 3:
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
    r = Retriever()
    r.set_reranker("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    res = Copilot(r, a.model, pipeline=a.pipeline).ask(a.question)
    print(res["answer"], "\n")
    for s in res["sources"]:
        print(f"  [{s['n']}] {s['doc']} p{s['page']} {s['title'][:50]}{'  (cited)' if s['n'] in res['cited'] else ''}")
    print(" ", res["tool_calls"], res["flags"], res["latency_s"], "s")


if __name__ == "__main__":
    main()
