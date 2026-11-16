#!/usr/bin/env python3
"""One command on the Mac (Python 3.9 is fine: this script only uses the standard library).

  python3 run_and_push.py --limit 20     # quick trial: ~20 questions, does NOT push
  python3 run_and_push.py                # full run (all questions, with/without verifier), then commit + push
  python3 run_and_push.py --push-only    # just commit + push what is in results/

Needs: Ollama app running (https://ollama.com). Everything is free and local.
"""
import argparse
import base64
import getpass
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv312"
PY = VENV / "bin" / "python"
OWNER, REPO = "ChrishA-tech", "filings-analyst-agent"
AUTHOR = "Chrish Pep Felix A <chrishpepfelixa@gmail.com>"
TRAILER = ""
MODELS = [os.environ.get("OLLAMA_CHAT", "qwen2.5:7b"), os.environ.get("OLLAMA_EMBED", "nomic-embed-text")]


def sh(cmd, **kw):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, cwd=ROOT, **kw)


def setup():
    if PY.exists():
        return
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    if not Path(uv).exists():
        sh(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
    sh([uv, "venv", "--python", "3.12", VENV])
    sh([uv, "pip", "install", "--python", PY, "-r", "requirements-dev.txt"])


def ollama():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5).read()
    except Exception:
        sys.exit("Ollama is not running. Open the Ollama app (or run `ollama serve`) and try again.")
    for m in MODELS:
        sh(["ollama", "pull", m])


def api(path, token, method="GET", body=None):
    req = urllib.request.Request("https://api.github.com" + path, method=method,
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {}


def push():
    token = os.environ.get("GITHUB_TOKEN") or getpass.getpass("GitHub token (classic, repo scope; input hidden): ").strip()
    st, me = api("/user", token)
    if st != 200 or me.get("login") != OWNER:
        sys.exit(f"Token belongs to {me.get('login')!r}, expected {OWNER}. Stopping.")
    if api(f"/repos/{OWNER}/{REPO}", token)[0] == 404:
        api("/user/repos", token, "POST", {"name": REPO, "private": False,
            "description": "Multi-agent (researcher + verifier) RAG over SEC 10-K filings: LangGraph, hybrid BM25+vector search, "
                           "measured evals, local Ollama / Azure OpenAI / Databricks backends"})
        api(f"/repos/{OWNER}/{REPO}/topics", token, "PUT", {"names": ["rag", "langgraph", "multi-agent", "llm-evaluation",
            "azure-openai", "databricks", "ollama", "sec-filings", "mcp", "fastapi"]})
    if not (ROOT / ".git").exists():
        sh(["git", "init", "-b", "main"])
    sh(["git", "add", "-A"])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        name, email = AUTHOR[:-1].split(" <")
        sh(["git", "-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-m",
            "Filings analyst: results from local run" + TRAILER])
    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    r = subprocess.run(["git", "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {auth}", "push",
                        f"https://github.com/{OWNER}/{REPO}.git", "HEAD:main"], cwd=ROOT, capture_output=True, text=True)
    print((r.stdout + r.stderr).replace(token, "***").replace(auth, "***"))
    if r.returncode:
        sys.exit("Push failed (see above).")
    print(f"Done: https://github.com/{OWNER}/{REPO}\nNow delete the token at https://github.com/settings/tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--push-only", action="store_true")
    a = ap.parse_args()
    if not a.push_only:
        setup()
        sh([PY, "-m", "pytest", "-q"])
        ollama()
        if not (ROOT / "data/sec/meta.json").exists():
            sh([PY, "-m", "analyst.sec"])
        sh([PY, "-c", "from analyst.pipeline import build_corpus; print(build_corpus())"])
        sh([PY, "run_evals.py", "--backend", "ollama"] + (["--limit", a.limit] if a.limit else []))
        sh([PY, "-m", "analyst.report"])
        if a.limit:
            print("\nTrial finished. Check results/summary_ollama.json, then run without --limit to do the full run and push.")
            return
    push()


if __name__ == "__main__":
    main()
