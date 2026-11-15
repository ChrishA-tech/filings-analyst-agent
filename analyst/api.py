"""FastAPI service: POST /ask {"question": ..., "ticker": "AAPL"} -> cited, verified answer."""
import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .agents import ask
from .pipeline import make

app = FastAPI(title="filings-analyst")
_STATE = {}


class Q(BaseModel):
    question: str
    ticker: Optional[str] = None


def _app():
    if "app" not in _STATE:
        _STATE["index"], _STATE["llm"], _STATE["app"] = make(os.environ.get("ANALYST_BACKEND", "ollama"))
    return _STATE["app"]


@app.get("/health")
def health():
    return {"ok": True, "backend": os.environ.get("ANALYST_BACKEND", "ollama")}


@app.post("/ask")
def ask_route(q: Q):
    out = ask(_app(), q.question, q.ticker)
    out.pop("verdicts", None)
    return out
