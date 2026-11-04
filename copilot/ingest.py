"""Corpus -> data/index/chunks.jsonl, errors.json, params.json, trials.sqlite.

Sources (see README for where each comes from; none of them is redistributed in this repo):
  ur_manual_en / ur_manual_de  UR3e user manual, PolyScope 5.24, English and German PDF
  ur_errors                    UR Error Codes Directory PDF, parsed into one record per code
  ros2_docs, moveit_docs,      ROS 2 Humble documentation, MoveIt 2 tutorials (Humble),
  ur_driver_docs               Universal_Robots_ROS2_Driver docs (rst / md)
  thesis_config                YAML parameter files of my UR3e thesis stack
  trials                       38 logged thesis trials (CSV), loaded into SQLite for the query tool

A chunk keeps where it came from (doc, page or section), so every answer can cite a page.

  python -m copilot.ingest
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3

import pymupdf as fitz

from copilot.config import CORPUS, INDEX, THESIS_WS

WORDS, OVERLAP = 220, 40

NOISE = re.compile(r"^(Copyright ©.*|UR3e|User Manual|Benutzerhandbuch|Error Codes Directory|All Robots|\d{1,3}|"
                   r"710-94\d-00|Original instructions.*|Originalbetriebsanleitung.*)$")
HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.\s+([^\d.].{2,70})$")


def is_heading(ln: str):
    """'7.7. Freedrive' or '18. Maintenance and Repair', not a numbered step like '4. Move the robot arm.'"""
    m = HEADING.match(ln)
    if not m:
        return None
    title = m.group(2)
    if "." not in m.group(1):  # top level: short, title-like, no sentence punctuation, chapters only go to ~20
        if int(m.group(1)) > 25 or len(title.split()) > 6 or any(ch in title for ch in ".:;,?!") \
                or not title[0].isupper():
            return None
    elif title.endswith(".") or len(title.split()) > 10:
        return None
    return m


def windows(text: str, n: int = WORDS, overlap: int = OVERLAP):
    w = text.split()
    if len(w) <= n:
        if w:
            yield " ".join(w)
        return
    for i in range(0, len(w), n - overlap):
        yield " ".join(w[i:i + n])
        if i + n >= len(w):
            break


def clean_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not NOISE.match(ln.strip())]


def pdf_chunks(path, source: str, lang: str):
    doc = fitz.open(path)
    section = ""
    for pno, page in enumerate(doc, start=1):
        lines = clean_lines(page.get_text())
        if not lines:
            continue
        # a table-of-contents page is all headings; index it but do not let it set the section
        heads = [ln for ln in lines if is_heading(ln)]
        toc = len(heads) > 0.5 * len(lines)
        body = []
        title_here = section
        for ln in lines:
            m = is_heading(ln)
            if m and not toc:
                section = f"{m.group(1)}. {m.group(2)}"
                title_here = title_here or section
            body.append(ln)
        for k, t in enumerate(windows(" ".join(body))):
            yield {"id": f"{source}:p{pno}:{k}", "source": source, "doc": path.name, "page": pno,
                   "title": section if not toc else "Contents", "lang": lang, "text": t}


ENTRY = re.compile(r"^(C\d+A\d+)\s+(.*)$")
GROUP = re.compile(r"^\d+\.\d+\.\s+(C\d+)\s+(.+)$")


def parse_error_codes(path):
    """One record per code (C72A1 ...) and per group (C72 ...), with explanation and suggestion."""
    doc = fitz.open(path)
    records, groups = [], {}
    cur, field, group = None, None, None
    start = None
    for pno, page in enumerate(doc, start=1):
        lines = clean_lines(page.get_text())
        if start is None:  # skip the table of contents: the directory starts at the first group heading
            if any(GROUP.match(ln) for ln in lines) and not sum(bool(GROUP.match(ln)) for ln in lines) > 8:
                start = pno
            else:
                continue
        for ln in lines:
            g, e = GROUP.match(ln), ENTRY.match(ln)
            if g:
                group = {"code": g.group(1), "title": g.group(2).strip(), "page": pno, "notes": []}
                groups[group["code"]] = group
                cur, field = None, None
            elif e:
                cur = {"code": e.group(1), "title": e.group(2).strip(), "explanation": "", "suggestion": "",
                       "group": group["code"] if group else "", "group_title": group["title"] if group else "",
                       "page": pno}
                records.append(cur)
                field = "title"
            elif ln in ("EXPLANATION", "SUGGESTION"):
                field = ln.lower()
            elif cur is not None and field:
                cur[field] = (cur[field] + " " + ln).strip()
            elif group is not None:
                group["notes"].append(ln)
    return records, list(groups.values())


def error_chunks(records, groups, docname):
    by_group = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)
    for g in groups:
        parts = [f"{g['code']} {g['title']}."] + [" ".join(g["notes"])]
        for r in by_group.get(g["code"], []):
            s = f"{r['code']} {r['title']}."
            if r["explanation"]:
                s += f" Explanation: {r['explanation']}."
            if r["suggestion"]:
                s += f" Suggestion: {r['suggestion']}."
            parts.append(s)
        for k, t in enumerate(windows(" ".join(parts), 260, 40)):
            yield {"id": f"ur_errors:{g['code']}:{k}", "source": "ur_errors", "doc": docname, "page": g["page"],
                   "title": f"{g['code']} {g['title']}", "lang": "en", "text": t}


RST_HEAD = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")
ROLE = re.compile(r":[a-z:]+:`([^`<]*?)\s*(?:<[^>]*>)?`")


def text_sections(path, md: bool):
    lines = path.read_text(errors="ignore").splitlines()
    title, buf = path.stem, []
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if md and ln.startswith("#"):
            out.append((title, buf)); title, buf = ln.lstrip("#").strip(), []
        elif not md and ln.strip() and RST_HEAD.match(nxt) and len(nxt.strip()) >= len(ln.strip()) - 2 \
                and not RST_HEAD.match(ln):
            out.append((title, buf)); title, buf = ln.strip(), []
            i += 1
        elif not (RST_HEAD.match(ln) or ln.startswith(".. ") and "::" not in ln[3:] or ln.strip().startswith(":")):
            buf.append(ln)
        i += 1
    out.append((title, buf))
    for t, b in out:
        txt = ROLE.sub(r"\1", " ".join(x.strip() for x in b if x.strip()))
        txt = txt.replace("``", "")
        if len(txt.split()) >= 15:
            yield t, txt


def doc_chunks(root, source: str, base):
    for p in sorted(root.rglob("*")):
        if p.suffix not in (".rst", ".md") or any(s in p.parts for s in (".git", "_build", "build")):
            continue
        if "Changelog" in p.name or p.name.startswith("CHANGELOG"):  # release notes of every package: noise here
            continue
        rel = str(p.relative_to(base))
        for si, (t, txt) in enumerate(text_sections(p, p.suffix == ".md")):
            for k, w in enumerate(windows(txt)):
                yield {"id": f"{source}:{rel}#{si}:{k}", "source": source, "doc": rel, "page": si, "title": t,
                       "lang": "en", "text": w}


def yaml_params(path):
    """Flatten a ROS 2 params YAML into node.param = value records, keeping the trailing comment."""
    out, stack = [], []
    for ln in path.read_text().splitlines():
        if not ln.strip() or ln.strip().startswith("#"):
            continue
        ind = len(ln) - len(ln.lstrip())
        body, _, comment = ln.strip().partition(" #")
        if ":" not in body:
            continue
        key, _, val = body.partition(":")
        while stack and stack[-1][0] >= ind:
            stack.pop()
        path_keys = [k for _, k in stack if k != "ros__parameters"]
        if val.strip():
            out.append({"file": path.name, "name": ".".join(path_keys + [key.strip()]), "value": val.strip(),
                        "comment": comment.strip()})
        else:
            stack.append((ind, key.strip()))
    return out


def thesis_chunks(cfg_dir):
    for p in sorted(cfg_dir.glob("*.yaml")):
        txt = p.read_text()
        for k, w in enumerate(windows(re.sub(r"\s+", " ", txt), 200, 40)):
            yield {"id": f"thesis_config:{p.name}:{k}", "source": "thesis_config", "doc": p.name, "page": k,
                   "title": p.name, "lang": "en", "text": w}


def load_trials(exp_dir, db):
    con = sqlite3.connect(db)
    for table, prefix in (("trials", "task_summary"), ("guide_cycles", "guide_cycles"),
                          ("state_transitions", "state_transitions")):
        rows, cols = [], None
        for f in sorted(exp_dir.glob(f"{prefix}_*.csv")):
            with open(f) as fh:
                r = list(csv.DictReader(fh))
            if not r:
                continue
            cols = cols or list(r[0].keys())
            rows += [[x.get(c) for c in cols] for x in r]
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(f"CREATE TABLE {table} ({', '.join(cols)})")
        con.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * len(cols))})", rows)
    con.commit()
    return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("trials", "guide_cycles", "state_transitions")}


def main() -> None:
    INDEX.mkdir(parents=True, exist_ok=True)
    ur = CORPUS / "ur"
    chunks = []
    chunks += pdf_chunks(ur / "UR3e_User_Manual_en_SW5.24.pdf", "ur_manual_en", "en")
    chunks += pdf_chunks(ur / "UR3e_User_Manual_de_SW5.24.pdf", "ur_manual_de", "de")
    recs, groups = parse_error_codes(ur / "ErrorCodes.pdf")
    chunks += error_chunks(recs, groups, "ErrorCodes.pdf")
    chunks += doc_chunks(CORPUS / "ros2_documentation" / "source", "ros2_docs", CORPUS / "ros2_documentation")
    chunks += doc_chunks(CORPUS / "moveit2_tutorials" / "doc", "moveit_docs", CORPUS / "moveit2_tutorials")
    chunks += doc_chunks(CORPUS / "Universal_Robots_ROS2_Driver", "ur_driver_docs",
                         CORPUS / "Universal_Robots_ROS2_Driver")
    cfg = THESIS_WS / "src" / "ur3e_moveit_config" / "config"
    chunks += thesis_chunks(cfg)
    with open(INDEX / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (INDEX / "errors.json").write_text(json.dumps({"codes": recs, "groups": groups}, indent=0))
    params = [p for f in sorted(cfg.glob("*.yaml")) for p in yaml_params(f)]
    (INDEX / "params.json").write_text(json.dumps(params, indent=0))
    n = load_trials(THESIS_WS / "experiment_data", INDEX / "trials.sqlite")
    by = {}
    for c in chunks:
        by[c["source"]] = by.get(c["source"], 0) + 1
    print("chunks", len(chunks), by)
    print("error codes", len(recs), "groups", len(groups), "params", len(params), "tables", n)


if __name__ == "__main__":
    main()
