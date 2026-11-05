"""Choose the relevance-gate threshold on the dev split, from a run made with the gate off.

The gate is applied after the agent loop (copilot/agent.py): if the best passage for the question scores below T,
no tool returned data, the question has no error code and was not routed to SQL, the answer becomes a refusal.
So replaying it over a gate-off run gives exactly the answers a gated run would give. T is chosen to maximise dev
accuracy (ties toward the lower T, i.e. fewer refusals); the test split is only reported.

  python eval/gate_offline.py --run agent_7b_small_nogate --out agent_7b_small_gated
"""

import argparse
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from copilot.agent import REFUSAL_DE, REFUSAL_EN, is_german  # noqa: E402
RES = REPO / "results"
CODE = re.compile(r"\bC\d{1,3}(?:A\d{1,4})?\b", re.I)


def gated(r, t):
    return ("routed_to_sql" not in r["flags"] and r.get("data_tools", 0) == 0 and not CODE.search(r["question"])
            and r["top_score"] < t)


def score(rows, t):
    ok = 0
    for r in rows:
        if gated(r, t):
            ok += r["type"] == "unanswerable"
        else:
            ok += r["correct"]
    return ok / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, help="apply a threshold chosen elsewhere (held-out set)")
    a = ap.parse_args()
    rows = [json.loads(ln) for ln in open(RES / f"answers_{a.run}.jsonl")]
    if a.threshold is not None:
        best = a.threshold
        print(f"no gate {score(rows, -99):.3f}   gate T={best:.2f} (fixed) {score(rows, best):.3f}")
    else:
        dev = [r for r in rows if r["split"] == "dev"]
        cands = sorted({r["top_score"] for r in dev if r["top_score"] > -50})
        ts = [-99.0] + [(x + y) / 2 for x, y in zip(cands, cands[1:])] + [cands[-1] + 0.01]
        best = max(ts, key=lambda t: (round(score(dev, t), 6), -t))
        test = [r for r in rows if r["split"] == "test"]
        print(f"no gate:  dev {score(dev, -99):.3f}  test {score(test, -99):.3f}")
        print(f"gate T={best:.2f}: dev {score(dev, best):.3f}  test {score(test, best):.3f}")
    with open(RES / f"answers_{a.out}.jsonl", "w") as f:
        for r in rows:
            r = dict(r)
            if gated(r, best):
                r["flags"] = r["flags"] + [f"relevance_gate:{r['top_score']:.2f}"]
                was = r["correct"]
                r["refused"] = True
                r["correct"] = r["type"] == "unanswerable"
                r["false_refusal"] = r["type"] != "unanswerable"
                r["gate_changed"] = was != r["correct"]
                r["answer"] = REFUSAL_DE if is_german(r["question"]) else REFUSAL_EN
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (RES / f"gate_{a.out}.json").write_text(json.dumps({"threshold": best, "chosen_on": "dev"}))


if __name__ == "__main__":
    main()
