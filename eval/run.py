"""Evaluation.

  python eval/run.py retrieval                         # recall@k / MRR per retriever, per question type
  python eval/run.py answers --pipeline agent --model qwen2.5:7b --rerank small --tag agent_7b
  python eval/run.py summary                           # tables from everything in results/

Answer grading is deterministic: an answerable question is correct when every required fact group appears in the
answer (any accepted spelling; numbers compared without spaces, decimal commas as points); an unanswerable
question is correct when the copilot refuses. No LLM judge decides correctness.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RES = REPO / "results"

RERANKERS = {"small": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "large": "BAAI/bge-reranker-v2-m3"}


def questions(split=None):
    qs = [json.loads(ln) for ln in open(REPO / "eval" / "questions.jsonl")]
    return [q for q in qs if split in (None, "all", q["split"])]


def norm(s: str) -> str:
    s = s.lower().replace("−", "-").replace("–", "-")
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    return re.sub(r"\s+", "", s)


def facts_ok(answer: str, facts) -> tuple[bool, list]:
    a = norm(answer)
    missing = [g for g in facts if not any(norm(x) in a for x in g)]
    return not missing, missing


def hit_rank(hits, gold) -> int | None:
    gold = {(d, p) for d, p in gold}
    for i, (c, _) in enumerate(hits):
        if (c["doc"], c["page"]) in gold:
            return i + 1
    return None


def run_retrieval(split):
    from copilot.retrieve import Retriever
    qs = [q for q in questions(split) if q["gold"]]
    r = Retriever()
    configs = [("bm25", "bm25", None), ("dense", "dense", None), ("hybrid", "hybrid", None),
               ("hybrid+rerank small", "hybrid+rerank", "small"), ("hybrid+rerank large", "hybrid+rerank", "large")]
    out = {}
    for name, mode, rr in configs:
        if rr:
            r.set_reranker(RERANKERS[rr])
        rows, lat = [], []
        for q in qs:
            t = time.time()
            hits = r.search(q["question"], 10, mode)
            lat.append(time.time() - t)
            rk = hit_rank(hits, q["gold"])
            rows.append({"id": q["id"], "type": q["type"], "rank": rk})
        out[name] = {"rows": rows, "latency_ms_p50": round(1000 * statistics.median(lat), 1),
                     "latency_ms_p95": round(1000 * sorted(lat)[int(0.95 * len(lat)) - 1], 1)}
        m = metrics_retrieval(rows)
        print(f"{name:22s} R@1 {m['r1']:.3f}  R@5 {m['r5']:.3f}  MRR {m['mrr']:.3f}  "
              f"p50 {out[name]['latency_ms_p50']} ms", flush=True)
    RES.mkdir(exist_ok=True)
    (RES / f"retrieval_{split}.json").write_text(json.dumps(out, indent=1))


def metrics_retrieval(rows):
    n = len(rows)
    return {"n": n, "r1": sum(1 for x in rows if x["rank"] == 1) / n,
            "r5": sum(1 for x in rows if x["rank"] and x["rank"] <= 5) / n,
            "mrr": sum(1 / x["rank"] for x in rows if x["rank"]) / n}


def run_answers(a):
    from copilot.agent import Copilot
    from copilot.retrieve import Retriever
    r = Retriever()
    if a.rerank != "none":
        r.set_reranker(RERANKERS[a.rerank])
    mode = "hybrid" if a.rerank == "none" else "hybrid+rerank"
    cp = Copilot(r, a.model, mode=mode, pipeline=a.pipeline, k=a.k)
    out = RES / f"answers_{a.tag}.jsonl"
    RES.mkdir(exist_ok=True)
    done = {json.loads(ln)["id"] for ln in open(out)} if out.exists() else set()
    qs = [q for q in questions(a.split) if q["id"] not in done]
    with open(out, "a") as f:
        for q in qs:
            try:
                res = cp.ask(q["question"])
            except Exception as e:  # a crash is a wrong answer, recorded, not a skipped question
                res = {"answer": f"<error: {e}>", "refused": False, "flags": ["crash"], "tool_calls": [],
                       "sources": [], "cited": [], "latency_s": None}
            res.update(grade(q, res))
            res.update({"id": q["id"], "type": q["type"], "split": q["split"]})
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{q['id']} {q['type']:13s} {'OK ' if res['correct'] else 'BAD'} {res['latency_s']}s "
                  f"{res['answer'][:90]!r}", flush=True)


def grade(q, res):
    if q["type"] == "unanswerable":
        return {"correct": bool(res["refused"]), "missing": []}
    if res["refused"]:
        return {"correct": False, "missing": q["facts"], "false_refusal": True}
    ok, missing = facts_ok(res["answer"], q["facts"])
    return {"correct": ok, "missing": missing}


def summary(split):
    print(f"## retrieval ({split})")
    f = RES / f"retrieval_{split}.json"
    if f.exists():
        d = json.loads(f.read_text())
        types = sorted({x["type"] for v in d.values() for x in v["rows"]})
        print("| retriever | R@1 | R@5 | MRR | p50 ms | " + " | ".join(f"R@5 {t}" for t in types) + " |")
        print("|---" * (5 + len(types)) + "|")
        for name, v in d.items():
            m = metrics_retrieval(v["rows"])
            per = [metrics_retrieval([x for x in v["rows"] if x["type"] == t])["r5"] for t in types]
            print(f"| {name} | {m['r1']:.2f} | {m['r5']:.2f} | {m['mrr']:.2f} | {v['latency_ms_p50']} | "
                  + " | ".join(f"{p:.2f}" for p in per) + " |")
    print(f"\n## answers ({split})")
    print("| config | n | accuracy | answerable | unanswerable refused | false refusals | invalid citations | "
          "uncited answers | p50 s | p95 s |")
    print("|---" * 10 + "|")
    for f in sorted(RES.glob("answers_*.jsonl")):
        rows = [json.loads(ln) for ln in open(f)]
        rows = [x for x in rows if split in ("all", x["split"])]
        if not rows:
            continue
        ans = [x for x in rows if x["type"] != "unanswerable"]
        un = [x for x in rows if x["type"] == "unanswerable"]
        lat = sorted(x["latency_s"] for x in rows if x["latency_s"] is not None)
        inv = sum(any(fl.startswith("invalid_citations") for fl in x["flags"]) for x in rows)
        unc = sum("uncited_refused" in x["flags"] for x in rows)
        print(f"| {f.stem[8:]} | {len(rows)} | {sum(x['correct'] for x in rows) / len(rows):.2f} | "
              f"{sum(x['correct'] for x in ans)}/{len(ans)} | {sum(x['correct'] for x in un)}/{len(un)} | "
              f"{sum(bool(x.get('false_refusal')) for x in ans)} | {inv} | {unc} | "
              f"{statistics.median(lat):.1f} | {lat[int(0.95 * len(lat)) - 1]:.1f} |")
    types = ["manual", "manual_de", "error", "error_symptom", "ros2", "moveit", "driver", "param", "trials",
             "unanswerable"]
    print("\n| config | " + " | ".join(types) + " |")
    print("|---" * (1 + len(types)) + "|")
    for f in sorted(RES.glob("answers_*.jsonl")):
        rows = [json.loads(ln) for ln in open(f)]
        rows = [x for x in rows if split in ("all", x["split"])]
        if not rows:
            continue
        cells = []
        for t in types:
            xs = [x for x in rows if x["type"] == t]
            cells.append(f"{sum(x['correct'] for x in xs)}/{len(xs)}" if xs else "")
        print(f"| {f.stem[8:]} | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["retrieval", "answers", "summary"])
    ap.add_argument("--split", default="all")
    ap.add_argument("--pipeline", default="agent", choices=["agent", "rag"])
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--rerank", default="small", choices=["none", "small", "large"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--tag")
    a = ap.parse_args()
    if a.what == "retrieval":
        run_retrieval(a.split)
    elif a.what == "answers":
        run_answers(a)
    else:
        summary(a.split)


if __name__ == "__main__":
    main()
