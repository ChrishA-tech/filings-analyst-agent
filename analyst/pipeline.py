"""Corpus build + factory for (index, llm, app) on any backend."""
from __future__ import annotations

import json
from pathlib import Path

from . import config as C
from .agents import build
from .llm import APIEmbeddings, FakeChat, HashEmbeddings, OpenAICompat
from .retrieval import HybridIndex, load_rows, to_documents
from .text import documents


def build_corpus(sec_dir: Path = None, out: Path = None) -> Path:
    sec_dir, out = sec_dir or C.DATA / "sec", out or C.DATA / "chunks.jsonl"
    meta = json.loads((sec_dir / "meta.json").read_text())
    problems = []
    with out.open("w") as f:
        for t, m in meta.items():
            raw = (sec_dir / f"{t}_10k.html").read_text(errors="ignore")
            docs = documents(t, m["name"], m["report_date"][:4], raw)
            items = sorted({d["meta"]["item"] for d in docs})
            size = {i: sum(len(d["text"]) for d in docs if d["meta"]["item"] == i) // 1000 for i in ("1", "1A", "7", "8")}
            print(f"{t}: {len(docs)} chunks, items {items}, thousand chars in Items 1/1A/7/8: {size}")
            problems += [f"{t} missing Item {i}" for i in ("1", "1A", "7", "8") if i not in items]
            for d in docs:
                f.write(json.dumps(d) + "\n")
    if problems:                      # section split failed: evals by Item would be meaningless
        print("WARNING: " + "; ".join(problems))
    return out


def make(backend: str = "ollama", verify: bool = True, chunks: Path = None,
         persist: bool = True, chat_model: str = None):
    chunks = chunks or C.DATA / "chunks.jsonl"
    rows = load_rows(chunks)
    if backend == "fake":
        llm, emb, pdir = FakeChat(), HashEmbeddings(), None
    else:
        llm = OpenAICompat(backend, chat_model=chat_model)
        emb = APIEmbeddings(llm)
        import hashlib
        digest = hashlib.md5(chunks.read_bytes()).hexdigest()[:10]               # new corpus -> new index
        tag = f"{backend}-{llm.embed_model.replace(':', '_').replace('/', '_')}-{digest}"
        pdir = (C.INDEX_DIR / tag) if persist else None
    index = HybridIndex(to_documents(rows), emb, persist_dir=pdir)
    return index, llm, build(index, llm, verify=verify)
