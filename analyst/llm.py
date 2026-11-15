"""One small OpenAI-compatible client for Ollama, Azure OpenAI and Databricks serving endpoints, plus offline fakes."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.request
from typing import Dict, List, Optional

from langchain_core.embeddings import Embeddings

from . import config as C


class Usage:
    def __init__(self):
        self.prompt = self.completion = self.calls = 0

    def add(self, u: Optional[Dict]):
        self.calls += 1
        if u:
            self.prompt += int(u.get("prompt_tokens", 0))
            self.completion += int(u.get("completion_tokens", 0))


class OpenAICompat:
    """POST {base}/chat/completions and {base}/embeddings. Standard library only, so it runs anywhere."""

    def __init__(self, backend: str = "ollama", chat_model: Optional[str] = None, embed_model: Optional[str] = None):
        b = C.BACKENDS[backend]
        self.backend, self.base = backend, b["base_url"]
        self.chat_model, self.embed_model = chat_model or b["chat"], embed_model or b["embed"]
        self.key = os.environ.get(b["key_env"], "") if b["key_env"] else "ollama"
        if b["key_env"] and not self.key:
            raise RuntimeError(f"Set {b['key_env']} for backend '{backend}'")
        self.usage = Usage()

    def _post(self, path: str, body: Dict, retries: int = 3) -> Dict:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}", "api-key": self.key}
        for i in range(retries):
            try:
                req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers=headers)
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.loads(r.read())
            except Exception:
                if i == retries - 1:
                    raise
                time.sleep(3 * (i + 1))
        raise RuntimeError("unreachable")

    def chat(self, messages: List[Dict], temperature: float = 0.0, max_tokens: int = 700) -> str:
        if self.backend == "ollama":        # native API: the OpenAI-style endpoint ignores num_ctx and may truncate
            r = self._post_native("/api/chat", {"model": self.chat_model, "messages": messages, "stream": False,
                                                "options": {"temperature": temperature, "num_predict": max_tokens,
                                                            "num_ctx": 8192, "seed": 7}})
            self.usage.add({"prompt_tokens": r.get("prompt_eval_count", 0), "completion_tokens": r.get("eval_count", 0)})
            return r["message"]["content"] or ""
        r = self._post("/chat/completions", {"model": self.chat_model, "messages": messages,
                                             "temperature": temperature, "max_tokens": max_tokens})
        self.usage.add(r.get("usage"))
        return r["choices"][0]["message"]["content"] or ""

    def _post_native(self, path: str, body: Dict) -> Dict:
        base = self.base[:-3] if self.base.endswith("/v1") else self.base
        saved, self.base = self.base, base
        try:
            return self._post(path, body)
        finally:
            self.base = saved

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for i in range(0, len(texts), 64):
            r = self._post("/embeddings", {"model": self.embed_model, "input": texts[i:i + 64]})
            out += [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
        return out


class APIEmbeddings(Embeddings):
    def __init__(self, client: OpenAICompat):
        self.client = client

    def embed_documents(self, texts):
        return self.client.embed(list(texts))

    def embed_query(self, text):
        return self.client.embed([text])[0]


# ---------------------------------------------------------------- offline fakes (tests, CI)
_WORD = re.compile(r"[a-z0-9]+")


class HashEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words vectors: good enough to test the plumbing, not a real model."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _v(self, t: str):
        v = [0.0] * self.dim
        for w in _WORD.findall(t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)


class FakeChat:
    """Rule-based stand-in: answers with the sentence of the best-overlapping source, cites it; judges by word overlap."""

    chat_model = "fake"

    def __init__(self, lie: bool = False):
        self.lie, self.usage = lie, Usage()

    def chat(self, messages, temperature=0.0, max_tokens=700):
        self.usage.add({"prompt_tokens": sum(len(m["content"]) // 4 for m in messages), "completion_tokens": 40})
        sys_, user = messages[0]["content"], messages[-1]["content"]
        if "VERIFIER" in sys_:
            return self._verify(user)
        return self._answer(user)

    def _answer(self, user: str) -> str:
        q = set(_WORD.findall(user.split("QUESTION:")[-1].lower()))
        best, src = None, None
        for m in re.finditer(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\nQUESTION:|\Z)", user, re.S):
            for s in re.split(r"(?<=[.!?])\s+", m.group(2)):
                score = len(q & set(_WORD.findall(s.lower())))
                if score >= 2 and (best is None or score > best[0]):
                    best, src = (score, s.strip()), m.group(1)
        if best is None:
            return json.dumps({"answer": "INSUFFICIENT_EVIDENCE", "claims": []})
        claim = best[1] + (" Revenue also tripled on Mars." if self.lie else "")
        claims = [{"text": best[1], "sources": [int(src)]}]
        if self.lie:
            claims.append({"text": "Revenue also tripled on Mars.", "sources": [int(src)]})
        return json.dumps({"answer": claim, "claims": claims})

    def _verify(self, user: str) -> str:
        claim = user.split("CLAIM:")[-1].strip().lower()
        evidence = user.split("CLAIM:")[0].lower()
        words = [w for w in _WORD.findall(claim) if len(w) > 3]
        ok = bool(words) and sum(w in evidence for w in words) / len(words) >= 0.8
        return json.dumps({"supported": ok, "reason": "overlap"})
