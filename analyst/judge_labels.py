"""Calibrate the verifier (LLM-as-judge) against your own labels.

  python -m analyst.judge_labels export results/eval_ollama_verify.jsonl   -> results/judge_sample.csv (50 claims)
  (open the CSV, fill the human_supported column with 1 or 0 by reading the evidence)
  python -m analyst.judge_labels score                                      -> agreement + Cohen's kappa
"""
import csv
import json
import random
import sys

from . import config as C
from .evals import cohen_kappa

SAMPLE = C.RESULTS / "judge_sample.csv"


def export(claims_path, n=50, seed=7):
    rows = [json.loads(l) for l in open(claims_path) if l.strip()]
    claims = [c for r in rows for c in r.get("verdict_rows", [])]
    random.Random(seed).shuffle(claims)
    pos = [c for c in claims if c["supported"]][: n // 2]
    neg = [c for c in claims if not c["supported"]][: n - len(pos)]
    with SAMPLE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "claim", "evidence", "judge_supported", "human_supported"])
        w.writeheader()
        for c in pos + neg:
            w.writerow({"case": c["case"], "claim": c["text"], "evidence": c["evidence"][:3000],
                        "judge_supported": int(c["supported"]), "human_supported": ""})
    print(f"wrote {SAMPLE} with {len(pos) + len(neg)} claims; fill human_supported with 1/0")


def score():
    rows = [r for r in csv.DictReader(SAMPLE.open()) if r["human_supported"].strip() in ("0", "1")]
    if not rows:
        sys.exit("No labels yet: fill the human_supported column first.")
    j = [r["judge_supported"] == "1" for r in rows]
    h = [r["human_supported"].strip() == "1" for r in rows]
    out = {"labelled": len(rows), "agreement": round(sum(a == b for a, b in zip(j, h)) / len(rows), 3),
           "cohen_kappa": cohen_kappa(j, h),
           "judge_false_accept": sum(a and not b for a, b in zip(j, h)), "judge_false_reject": sum(b and not a for a, b in zip(j, h))}
    (C.RESULTS / "judge_calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    export(sys.argv[2]) if sys.argv[1] == "export" else score()
