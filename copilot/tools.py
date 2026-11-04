"""The copilot's tools. Each returns text for the model plus the sources it may cite.

search_docs        hybrid retrieval over the manuals and docs
lookup_error_code  exact lookup in the parsed UR Error Codes Directory (C153A1, C72, ...)
query_trials       read-only SQL over the 38 logged thesis trials
get_parameter      the thesis stack's ROS 2 parameters (YAML), with their comments
"""

from __future__ import annotations

import difflib
import json
import re
import sqlite3

from copilot.config import INDEX

SOURCES = {"ur_manual": ["ur_manual_en", "ur_manual_de"], "ur_errors": ["ur_errors"],
           "ros2": ["ros2_docs"], "moveit": ["moveit_docs"], "ur_driver": ["ur_driver_docs"],
           "thesis": ["thesis_config"]}

SPECS = [
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search the UR3e user manual (English and German), the UR error code directory, the ROS 2 "
                       "Humble docs, the MoveIt 2 tutorials, the UR ROS 2 driver docs and the thesis config files. "
                       "Returns numbered passages you can cite.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what to look for, in English or German"},
            "source": {"type": "string", "enum": ["all"] + list(SOURCES),
                       "description": "restrict to one collection, or all"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "lookup_error_code",
        "description": "Exact lookup of a Universal Robots error code such as C153A1 or C72: its title, explanation "
                       "and suggested fix from the official Error Codes Directory.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "query_trials",
        "description": "Run one read-only SQLite SELECT over the logged UR3e thesis trials. Tables: "
                       "trials(trial_id, total_time_s, grasp_success ('True'/'False'), num_guide_cycles, "
                       "peak_force_N, mean_force_N, double_tap_time_ms, displacement_m, phase_<NAME>_s ...); "
                       "guide_cycles(trial_id, cycle, duration_s, peak_force_N, mean_force_N, peak_velocity_mps, "
                       "latency_ms); state_transitions(trial_id, time_s, from_state, to_state). trial_id is the "
                       "start time YYYYMMDD_HHMMSS. Values are stored as text: CAST(x AS REAL) for maths.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}},
    {"type": "function", "function": {
        "name": "get_parameter",
        "description": "Look up a ROS 2 parameter of the thesis UR3e stack (safety monitor, hybrid controller, servo, "
                       "proximity monitor, MoveIt limits) by name, e.g. max_guide_force or near_dist.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
]


SYN = {"velocity": "speed", "vel": "speed", "speed": "velocity", "distance": "dist", "dist": "distance",
       "frequency": "rate", "hz": "rate", "limit": "max", "maximum": "max", "minimum": "min", "acc": "acceleration",
       "accel": "acceleration", "time": "t"}


class Tools:
    def __init__(self, retriever, mode: str = "hybrid+rerank", k: int = 5):
        self.r, self.mode, self.k = retriever, mode, k
        e = json.loads((INDEX / "errors.json").read_text())
        self.codes = {c["code"]: c for c in e["codes"]}
        self.groups = {g["code"]: g for g in e["groups"]}
        self.params = json.loads((INDEX / "params.json").read_text())
        self.db = INDEX / "trials.sqlite"

    # each returns (text for the model, [source dicts])
    def search_docs(self, query: str, source: str = "all"):
        hits = self.r.search(query, self.k, self.mode, None if source in (None, "all") else SOURCES.get(source))
        return "", [{"doc": c["doc"], "page": c["page"], "title": c["title"], "id": c["id"], "text": c["text"],
                     "score": round(s, 3)} for c, s in hits]

    def lookup_error_code(self, code: str):
        code = code.strip().upper().replace(" ", "")
        m = re.match(r"^(C\d+)(A\d+)?$", code)
        if not m:
            return f"'{code}' is not a UR error code (format C<number> or C<number>A<number>).", []
        out = []
        g = self.groups.get(m.group(1))
        rec = self.codes.get(code) if m.group(2) else None
        if rec:
            t = f"{rec['code']} {rec['title']} (group {rec['group']} {rec['group_title']})."
            if rec["explanation"]:
                t += f" Explanation: {rec['explanation']}."
            if rec["suggestion"]:
                t += f" Suggestion: {rec['suggestion']}."
            out.append({"doc": "ErrorCodes.pdf", "page": rec["page"], "title": rec["code"], "id": f"err:{code}",
                        "text": t})
        if g:
            subs = [c for c in self.codes.values() if c["group"] == g["code"]]
            t = f"{g['code']} {g['title']}. {' '.join(g['notes'])}".strip()
            if not rec and subs:
                t += " Sub-codes: " + "; ".join(f"{c['code']} {c['title']}" for c in subs[:25])
            out.append({"doc": "ErrorCodes.pdf", "page": g["page"], "title": g["code"], "id": f"err:{g['code']}",
                        "text": t})
        if not out:
            return f"{code} is not in the UR Error Codes Directory.", []
        if m.group(2) and not rec:
            return f"{code} itself is not listed; its group is:", out
        return "", out

    def query_trials(self, sql: str):
        s = sql.strip().rstrip(";")
        if not re.match(r"(?is)^\s*(select|with)\b", s) or re.search(r"(?i)\b(insert|update|delete|drop|alter|"
                                                                     r"attach|pragma|create)\b", s):
            return "Only a single read-only SELECT is allowed.", []
        con = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True)
        try:
            cur = con.execute(s)
            rows = cur.fetchmany(51)
            cols = [d[0] for d in cur.description]
        except sqlite3.Error as e:
            return f"SQL error: {e}", []
        txt = " | ".join(cols) + "\n" + "\n".join(" | ".join(str(v) for v in r) for r in rows[:50])
        if len(rows) > 50:
            txt += "\n(more rows cut off; aggregate in SQL)"
        return "", [{"doc": "trials.sqlite", "page": 0, "title": "thesis trial logs", "id": "sql:" + s[:80],
                     "text": f"SQL: {s}\nResult:\n{txt}"}]

    def get_parameter(self, name: str):
        key = name.strip().split(".")[-1].lower()
        exact = [p for p in self.params if p["name"].split(".")[-1].lower() == key]
        hits = exact or [p for p in self.params if key in p["name"].lower()]
        note = ""
        if not hits:  # the model often guesses a near name (max_guide_velocity for max_guide_speed)
            want = set(re.split(r"[_\W]+", key)) - {""}
            want |= {SYN[w] for w in want if w in SYN}
            def score(p):
                have = set(re.split(r"[_\W]+", p["name"].lower()))
                return len(want & have) + difflib.SequenceMatcher(None, key, p["name"].split(".")[-1].lower()).ratio()
            ranked = sorted(self.params, key=score, reverse=True)
            hits = [p for p in ranked[:6] if score(p) >= 1.4]
            if not hits:
                return f"No parameter '{name}'. Try search_docs with source 'thesis'.", []
            note = f"No parameter named exactly '{name}'. The closest names are:"
        return note, [{"doc": p["file"], "page": 0, "title": p["name"], "id": f"param:{p['file']}:{p['name']}",
                     "text": f"{p['name']} = {p['value']}" + (f"  # {p['comment']}" if p["comment"] else "")
                     + f" (file {p['file']})"} for p in hits[:8]]

    def call(self, name: str, args: dict):
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_") or name == "call":
            return f"Unknown tool {name}.", []
        try:
            return fn(**args)
        except TypeError as e:
            return f"Bad arguments for {name}: {e}", []
