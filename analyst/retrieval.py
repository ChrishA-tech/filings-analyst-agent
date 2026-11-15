"""Hybrid retrieval: BM25 (exact terms, numbers, tickers) + dense vectors in Chroma, fused with reciprocal rank fusion."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from . import config as C


def to_documents(rows: List[Dict]) -> List[Document]:
    return [Document(page_content=r["text"], metadata={**r["meta"], "chunk_id": r["id"]}, id=r["id"]) for r in rows]


class HybridIndex:
    def __init__(self, docs: List[Document], embeddings: Embeddings, persist_dir: Optional[Path] = None,
                 k: int = C.TOP_K, collection: str = "filings"):
        import chromadb
        settings = chromadb.config.Settings(anonymized_telemetry=False)
        kw = {"persist_directory": str(persist_dir)} if persist_dir else {}
        self.k, self.docs = k, docs
        self.vs = Chroma(collection_name=collection, embedding_function=embeddings, client_settings=settings, **kw)
        if not self.vs.get(limit=1)["ids"]:
            for i in range(0, len(docs), 256):
                batch = docs[i:i + 256]
                self.vs.add_documents(batch, ids=[d.id for d in batch])
        self.bm25 = BM25Retriever.from_documents(docs, k=k * 2)

    def search(self, query: str, ticker: Optional[str] = None, k: Optional[int] = None, mode: str = "hybrid") -> List[Document]:
        k = k or self.k
        dense = self.vs.as_retriever(search_kwargs={"k": k * 2, **({"filter": {"ticker": ticker}} if ticker else {})})
        if mode == "dense":
            hits = dense.invoke(query)
        else:
            bm = self.bm25
            if ticker:                                   # BM25 has no metadata filter: over-fetch, then filter
                bm = BM25Retriever.from_documents([d for d in self.docs if d.metadata["ticker"] == ticker], k=k * 2)
            if mode == "bm25":
                hits = bm.invoke(query)
            else:
                hits = EnsembleRetriever(retrievers=[bm, dense], weights=[0.5, 0.5], c=C.RRF_K, id_key="chunk_id").invoke(query)
        return hits[:k]


def load_rows(path: Path) -> List[Dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
