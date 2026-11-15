"""Run the evaluation on one backend, with and without the verifier (ablation), and write results.

  python run_evals.py --backend ollama            # local, free
  python run_evals.py --backend azure             # Azure OpenAI (student credit)
  python run_evals.py --backend databricks        # Databricks foundation-model endpoints
  python run_evals.py --backend ollama --limit 20 # quick trial
"""
import argparse
import json
import time

from analyst import config as C
from analyst.evals import build_cases, run, summarise
from analyst.pipeline import build_corpus, make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--chat-model")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-ablation", action="store_true")
    a = ap.parse_args()
    C.RESULTS.mkdir(exist_ok=True)
    if not (C.DATA / "chunks.jsonl").exists():
        build_corpus()
    cases = build_cases()
    if a.limit:                                   # keep every category in a trial run
        per = max(1, a.limit // 4)
        cases = [c for cat in ("numeric", "qualitative", "unanswerable", "adversarial")
                 for c in [x for x in cases if x["category"] == cat][:per]]
    (C.RESULTS / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    report = {"backend": a.backend, "n_cases": len(cases), "date": time.strftime("%Y-%m-%d"), "variants": {}}
    for verify in ([True] if a.no_ablation else [True, False]):
        name = "verifier" if verify else "no_verifier"
        index, llm, app = make(a.backend, verify=verify, chat_model=a.chat_model)
        log = C.RESULTS / f"eval_{a.backend}_{name}.jsonl"
        log.unlink(missing_ok=True)
        t = time.time()
        rows = run(app, index, cases, log)
        s = summarise(rows)
        s.update(model=llm.chat_model, minutes=round((time.time() - t) / 60, 1), llm_calls=llm.usage.calls,
                 prompt_tokens=llm.usage.prompt, completion_tokens=llm.usage.completion)
        report["variants"][name] = s
        print(name, json.dumps(s, indent=2))
    (C.RESULTS / f"summary_{a.backend}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
