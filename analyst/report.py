"""Render results/summary_*.json into the README between the RESULTS markers."""
import json
import re
from pathlib import Path

from . import config as C

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _f(v, pct=True):
    return "n/a" if v is None else (f"{v * 100:.1f}%" if pct else f"{v:,}")


def render() -> str:
    out = []
    for p in sorted(C.RESULTS.glob("summary_*.json")):
        r = json.loads(p.read_text())
        out.append(f"**Backend: {r['backend']}** ({r['n_cases']} cases, run {r.get('date', '')})\n")
        out.append("| Metric | " + " | ".join(r["variants"]) + " |")
        out.append("|---|" + "---|" * len(r["variants"]))
        vs = list(r["variants"].values())
        rows = [("Model", [v["model"] for v in vs], False),
                ("Overall accuracy", [v["overall_accuracy"] for v in vs], True)]
        for cat in ("numeric", "qualitative", "unanswerable", "adversarial"):
            rows.append((f"{cat.capitalize()} accuracy", [v["by_category"].get(cat, {}).get("accuracy") for v in vs], True))
        rows += [("Retrieval hit@6 (numeric+qualitative)",
                  [_avg([v["by_category"].get(c, {}) for c in ("numeric", "qualitative")]) for v in vs], True),
                 ("Claims the verifier judged supported", [v["verifier_supported_share"] for v in vs], True),
                 ("False abstain rate (answerable questions)", [v["false_abstain_rate"] for v in vs], True),
                 ("Latency p50 / p95 (s)", [f"{v['latency_p50_ms'] / 1000:.1f} / {v['latency_p95_ms'] / 1000:.1f}" for v in vs], False),
                 ("LLM calls", [v["llm_calls"] for v in vs], False)]
        for name, vals, pct in rows:
            out.append(f"| {name} | " + " | ".join(_f(x) if pct else str(x) for x in vals) + " |")
        out.append("")
    cal = C.RESULTS / "judge_calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        out.append(f"**Verifier vs human labels:** {c['labelled']} claims, agreement {c['agreement'] * 100:.0f}%, "
                   f"Cohen's kappa {c['cohen_kappa']}.\n")
    else:
        out.append("_Verifier-vs-human calibration not run yet (see `analyst/judge_labels.py`)._\n")
    return "\n".join(out) if out else "_No results yet._"


def _avg(ms):
    xs = [(m["retrieval_hit@k"], m["n"]) for m in ms if m.get("retrieval_hit@k") is not None]
    return round(sum(a * n for a, n in xs) / sum(n for _, n in xs), 3) if xs else None


def fill(readme: Path = None):
    readme = readme or C.ROOT / "README.md"
    s = readme.read_text()
    s = re.sub(re.escape(START) + ".*?" + re.escape(END), START + "\n" + render() + "\n" + END, s, flags=re.S)
    readme.write_text(s)


if __name__ == "__main__":
    fill()
