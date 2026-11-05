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
        "description": "Exact lookup of a Universal Robots error code that the user wrote in the question (format "
                       "C<number> or C<number>A<number>): title, explanation and suggested fix from the official Error "
                       "Codes Directory. If the user describes a fault without a code, use search_docs with source "
                       "ur_errors instead.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "query_trials",
        "description": "Run one read-only SQLite SELECT over the 38 logged UR3e thesis trials. Tables and columns: "
                       "trials(trial_id TEXT 'YYYYMMDD_HHMMSS', trial_date TEXT 'YYYY-MM-DD', total_time_s REAL, "
                       "grasp_success INTEGER 1/0, num_guide_cycles REAL, peak_force_N REAL (the highest force in "
                       "that trial), mean_force_N REAL (the average force in that trial), double_tap_time_ms REAL, "
                       "displacement_m REAL, phase_<STATE>_s REAL time spent in each state, e.g. "
                       "phase_AWAIT_INTERACTION_s); guide_cycles(trial_id, cycle, duration_s, peak_force_N, "
                       "mean_force_N, peak_velocity_mps, latency_ms); state_transitions(trial_id, time_s, from_state, "
                       "to_state) with states such as IDLE, NUDGE, GUIDING. Examples: success rate = "
                       "SELECT AVG(grasp_success) FROM trials; trials on a day = SELECT COUNT(*) FROM trials WHERE "
                       "trial_date='2026-04-21'; transitions = SELECT from_state, to_state, COUNT(*) FROM "
                       "state_transitions GROUP BY 1, 2 ORDER BY 3 DESC.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}},
    {"type": "function", "function": {
        "name": "get_parameter",
        "description": "Look up a parameter in the thesis's own ROS 2 YAML files (safety_monitor, hybrid_control_node, "
                       "proximity_monitor, servo, joint_limits) by its exact name, e.g. max_guide_force or near_dist. "
                       "Only for questions about the thesis stack's configuration. UR robot safety limits, Recovery "
                       "Mode, ROS 2, MoveIt and UR driver settings are in the documentation: use search_docs.",
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
        return "", [{"doc": c["doc"], "page": c["page"], "title": c["title"], "id": c["id"],
                     "text": self._expand(c), "score": round(s, 3)} for c, s in hits]

    def _expand(self, c: dict) -> str:
        """A hit brings the next window of the same page: a 220-word window often holds the start of a table
        and its next window the row asked about (cycle 2: the mains voltage row of the spec table)."""
        by_id = getattr(self.r, "by_id", None)
        if not by_id:
            return c["text"]
        head, _, k = c["id"].rpartition(":")
        nxt = by_id.get(f"{head}:{int(k) + 1}") if k.isdigit() else None
        if nxt is None:
            return c["text"]
        tail = nxt["text"].split()[40:]  # windows overlap by 40 words
        return c["text"] + " " + " ".join(tail)

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
        except sqlite3.Error as e:  # give the schema back so the model can correct the query
            schema = "; ".join(f"{t}({', '.join(r[1] for r in con.execute(f'PRAGMA table_info({t})'))})"
                               for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"))
            return f"SQL error: {e}. Tables: {schema}", []
        if not rows:
            return f"The query returned 0 rows: SQL: {s}", []
        txt = " | ".join(cols) + "\n" + "\n".join(" | ".join(str(v) for v in r) for r in rows[:50])
        if len(rows) > 50:
            txt += "\n(more rows cut off; aggregate in SQL)"
        return "", [{"doc": "trials.sqlite", "page": 0, "title": "thesis trial logs", "id": "sql:" + s[:80],
                     "text": f"SQL: {s}\nResult:\n{txt}"}]

    def get_parameter(self, name: str):
        """Exact name, or a name that contains it. No fuzzy guessing: in the first evaluation round a near match
        (max_guide_force for any 'force limit') was reported as the answer to questions about the UR manual."""
        key = name.strip().split(".")[-1].lower()
        hits = [p for p in self.params if p["name"].split(".")[-1].lower() == key] or \
               [p for p in self.params if key in p["name"].lower() and len(key) >= 4]
        if not hits:
            want = set(re.split(r"[_\W]+", key)) - {""}
            want |= {SYN[w] for w in want if w in SYN}
            ov = {p["name"]: len(want & set(re.split(r"[_\W]+", p["name"].split(".")[-1].lower()))) for p in self.params}
            near = sorted((n for n, k in ov.items() if k >= 2), key=lambda n: -ov[n])
            return (f"No thesis parameter named '{name}'." + (f" Similar names: {', '.join(near[:5])}." if near else "")
                    + " If the question is not about the thesis configuration, use search_docs."), []
        return "", [{"doc": p["file"], "page": 0, "title": p["name"], "id": f"param:{p['file']}:{p['name']}",
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
