"""Choose the relevance gate threshold on the dev split only.

For each dev question: the reranker score of the best passage for the question itself. A threshold T refuses a
question when that score is below T (and no tool returned data). Trial-log questions are left out: they are
answered by SQL, which the gate does not block. The chosen T maximises dev accuracy of the decision
"answerable vs not", ties broken toward the lower T (fewer false refusals).

  python eval/calibrate_gate.py
"""

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.run import RERANKERS, questions  # noqa: E402


def main() -> None:
    from copilot.retrieve import Retriever
    r = Retriever()
    r.set_reranker(RERANKERS["small"])
    rows = []
    for split in ("dev", "test"):
        for q in questions(split):
            if q["type"] == "trials":
                continue
            top = r.search(q["question"], 5, "hybrid+rerank")[0][1]
            rows.append({"id": q["id"], "split": split, "type": q["type"], "answerable": q["type"] != "unanswerable",
                         "top": round(top, 3)})
    dev = [x for x in rows if x["split"] == "dev"]
    cands = sorted({x["top"] for x in dev})
    best = None
    for i in range(len(cands) + 1):
        t = -99 if i == 0 else (cands[i - 1] + (cands[i] if i < len(cands) else cands[i - 1] + 1)) / 2
        acc = sum((x["top"] >= t) == x["answerable"] for x in dev) / len(dev)
        if best is None or acc > best[1] + 1e-9:
            best = (t, acc)
    t = round(best[0], 2)
    print(f"dev: {len(dev)} questions, threshold {t}, dev decision accuracy {best[1]:.3f}")
    for split in ("dev", "test"):
        xs = [x for x in rows if x["split"] == split]
        fr = [x for x in xs if x["answerable"] and x["top"] < t]
        cr = [x for x in xs if not x["answerable"] and x["top"] < t]
        print(f"{split}: answerable below threshold {len(fr)}/{sum(x['answerable'] for x in xs)}  "
              f"unanswerable below threshold {len(cr)}/{sum(not x['answerable'] for x in xs)}")
    print("unanswerable scores:", sorted(x["top"] for x in rows if not x["answerable"]))
    print("lowest answerable:", sorted((x["top"], x["id"], x["type"]) for x in rows if x["answerable"])[:8])
    (REPO / "results" / "gate_calibration.json").write_text(json.dumps({"threshold": t, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
