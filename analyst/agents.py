"""LangGraph multi-agent flow: guard -> retrieve -> researcher -> verifier -> (revise once | finalize | abstain)."""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from . import config as C

ABSTAIN = "INSUFFICIENT_EVIDENCE"
INJECTION = re.compile(r"ignore (all |any |the )?(previous|prior|above) (instructions|rules)|system prompt|"
                       r"you are now|developer mode|reveal (your|the) (prompt|instructions)", re.I)

RESEARCHER = """You are a RESEARCHER analysing SEC 10-K filings. Answer ONLY from the numbered sources.
Return JSON: {"answer": "<2-4 sentences>", "claims": [{"text": "<one factual sentence from your answer>", "sources": [<source numbers>]}]}
Every claim must cite the source(s) that state it. Copy figures exactly as written in the source, with units.
If the sources do not contain the answer, return {"answer": "INSUFFICIENT_EVIDENCE", "claims": []}. Never guess."""

VERIFIER = """You are a strict VERIFIER. Decide whether the CLAIM is fully supported by the EVIDENCE alone.
Numbers must match the evidence. Anything not stated in the evidence is unsupported.
Return JSON: {"supported": true|false, "reason": "<short>"}"""


class State(TypedDict, total=False):
    question: str
    ticker: Optional[str]
    docs: List[Any]
    draft: Dict
    verdicts: List[Dict]
    feedback: str
    attempts: int
    answer: str
    citations: List[Dict]
    status: str            # answered | abstained | blocked
    trace: List[Dict]


def parse_json(text: str) -> Dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _sources(docs) -> str:
    return "\n".join(f"[{i + 1}] {d.metadata['company']} 10-K FY{d.metadata['fiscal_year']} Item {d.metadata['item']}\n"
                     f"{d.page_content}" for i, d in enumerate(docs))


def build(index, llm, verify: bool = True, judge=None):
    """judge defaults to the same model; pass a different one to separate generator and verifier."""
    judge = judge or llm

    def timed(name):
        def deco(fn):
            def run(s: State):
                t = time.perf_counter()
                out = fn(s)
                out["trace"] = s.get("trace", []) + [{"node": name, "ms": round((time.perf_counter() - t) * 1000, 1)}]
                return out
            return run
        return deco

    @timed("guard")
    def guard(s):
        if INJECTION.search(s["question"]):
            return {"status": "blocked", "answer": "Request blocked: it looks like a prompt-injection attempt.", "citations": []}
        return {"attempts": 0, "feedback": ""}

    @timed("retrieve")
    def retrieve(s):
        return {"docs": index.search(s["question"], ticker=s.get("ticker"))}

    @timed("research")
    def research(s):
        user = _sources(s["docs"]) + f"\nQUESTION: {s['question']}"
        if s.get("feedback"):
            user += f"\nA verifier rejected these claims, remove or fix them using the sources: {s['feedback']}"
        d = parse_json(llm.chat([{"role": "system", "content": RESEARCHER}, {"role": "user", "content": user}]))
        d.setdefault("answer", ABSTAIN)
        d["claims"] = [c for c in d.get("claims", []) if isinstance(c, dict) and c.get("text")]
        return {"draft": d, "attempts": s.get("attempts", 0) + 1}

    @timed("verify")
    def verifier(s):
        out = []
        for c in s["draft"]["claims"]:
            ids = [int(i) for i in c.get("sources", []) if str(i).isdigit() and 1 <= int(i) <= len(s["docs"])]
            if not ids:
                out.append({**c, "supported": False, "reason": "no valid citation"})
                continue
            ev = "\n\n".join(s["docs"][i - 1].page_content for i in ids)
            v = parse_json(judge.chat([{"role": "system", "content": VERIFIER},
                                       {"role": "user", "content": f"EVIDENCE:\n{ev}\n\nCLAIM: {c['text']}"}],
                                      max_tokens=120))
            out.append({**c, "sources": ids, "supported": bool(v.get("supported")), "reason": v.get("reason", ""), "evidence": ev[:3000]})
        bad = [v["text"] for v in out if not v["supported"]]
        return {"verdicts": out, "feedback": "; ".join(bad)}

    def route(s):
        if s["draft"]["answer"].strip() == ABSTAIN or not s["draft"]["claims"]:
            return "finalize"
        if not verify:
            return "finalize"
        return "verify"

    def after_verify(s):
        if s["feedback"] and s["attempts"] < 2:
            return "research"
        return "finalize"

    @timed("finalize")
    def finalize(s):
        d = s["draft"]
        if d["answer"].strip() == ABSTAIN or not d["claims"]:
            return {"status": "abstained", "answer": ABSTAIN, "citations": []}
        claims = s.get("verdicts") if verify else [{**c, "supported": None} for c in d["claims"]]
        kept = [c for c in claims if c["supported"] is not False]
        if not kept:
            return {"status": "abstained", "answer": ABSTAIN, "citations": []}
        text = d["answer"] if len(kept) == len(claims) else " ".join(c["text"] for c in kept)
        cites = [{"n": n, "chunk_id": s["docs"][n - 1].metadata["chunk_id"], "company": s["docs"][n - 1].metadata["company"],
                  "item": s["docs"][n - 1].metadata["item"]}
                 for n in sorted({n for c in kept for n in c.get("sources", []) if isinstance(n, int) and 1 <= n <= len(s["docs"])})]
        return {"status": "answered", "answer": text, "citations": cites}

    g = StateGraph(State)
    for name, fn in [("guard", guard), ("retrieve", retrieve), ("research", research), ("verify", verifier), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.set_entry_point("guard")
    g.add_conditional_edges("guard", lambda s: END if s.get("status") == "blocked" else "retrieve")
    g.add_edge("retrieve", "research")
    g.add_conditional_edges("research", route, {"verify": "verify", "finalize": "finalize"})
    g.add_conditional_edges("verify", after_verify, {"research": "research", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def ask(app, question: str, ticker: Optional[str] = None) -> Dict:
    s = app.invoke({"question": question, "ticker": ticker, "trace": []})
    return {k: s.get(k) for k in ("answer", "status", "citations", "verdicts", "trace", "attempts")}
