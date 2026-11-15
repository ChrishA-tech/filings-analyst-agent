# filings-analyst-agent

A two-agent analyst for SEC 10-K filings. A **researcher** agent answers from retrieved passages and cites them. A **verifier** agent checks every claim against the cited text before the answer goes out. Unsupported claims are sent back once for revision, then dropped. If nothing survives, the system abstains instead of guessing.

The same code runs on three backends with one switch: **local Ollama** (free), **Azure OpenAI** (deployed to Azure Container Apps) and **Databricks** foundation-model endpoints (with MLflow tracing). Every number in the results table comes from a scripted run, not a demo.

```mermaid
flowchart LR
    Q[Question + ticker] --> G{Guard<br/>prompt injection?}
    G -- blocked --> X[Refuse]
    G -- ok --> R[Hybrid retrieval<br/>BM25 + Chroma vectors<br/>reciprocal rank fusion]
    R --> A[Researcher agent<br/>answer + per-claim citations]
    A -- no evidence --> AB[Abstain]
    A --> V[Verifier agent<br/>judges each claim vs cited text]
    V -- unsupported claims, 1st pass --> A
    V --> F[Finalize<br/>keep supported claims only]
```

## What it shows

| Skill | Where |
|---|---|
| Multi-agent orchestration (LangGraph state graph, conditional routing, bounded revision loop) | `analyst/agents.py` |
| RAG: 10-K HTML parsing, Item-level sectioning, chunking with overlap, metadata filters | `analyst/text.py`, `analyst/sec.py` |
| Hybrid search: BM25 + dense vectors (Chroma) fused with RRF via LangChain `EnsembleRetriever` | `analyst/retrieval.py` |
| Evaluation with ground truth: numeric answers checked against the XBRL values tagged in the same filing | `analyst/evals.py` |
| Faithfulness: LLM-as-judge verifier, plus a calibration step against human labels (Cohen's kappa) | `analyst/agents.py`, `analyst/judge_labels.py` |
| Ablation: identical questions with and without the verifier | `run_evals.py` |
| Guardrails: prompt-injection blocking, abstention on unanswerable questions | `analyst/agents.py` |
| Serving: FastAPI + Docker; MCP server so any MCP client can call the analyst | `analyst/api.py`, `analyst/mcp_server.py` |
| Cloud: Azure OpenAI + Azure Container Apps (script), Databricks notebook with MLflow runs and tracing | `deploy/`, `databricks/` |

## Evaluation design

Questions are generated from the five filings (Apple, Microsoft, NVIDIA, Amazon, Alphabet, latest 10-K each):

- **Numeric** (revenue, net income, total assets): correct if the answer contains the XBRL value for that fiscal year within 1%.
- **Qualitative** (risk factors, business, workforce): scored on answering with valid citations, and on whether retrieval found the gold Item section (hit@6).
- **Unanswerable** (forecasts, trivia not in a 10-K): correct only if the system abstains.
- **Adversarial** (prompt injection): correct only if blocked.

## What the first full run showed (v1) and what changed (v2)

The v1 ablation said the verifier **hurt**: 91.2% accuracy with it, 95.0% without, and 13% of answerable
questions wrongly abstained instead of 4%. Reading the records showed why:

1. **Keyword search split text on spaces only**, so "Income" never matched "income." Of 15 numeric questions,
   BM25 retrieved a passage containing the true figure for 5; with word-level tokenisation, 14.
2. **The verifier only read the cited passage.** In 3 cases the number was right but cited to the wrong passage,
   so a correct answer was rejected. It now checks every retrieved passage and repairs the citation.
3. **The verifier matched numbers, not metrics:** it accepted Alphabet's "total long-lived assets" as total
   assets. It now requires the evidence to name the same line item.
4. **The retrieval metric was too loose.** "Right section retrieved" scored 100% while the figure itself was often
   missing. Numeric questions are now scored on whether the retrieved passages contain the true figure.

These fixes were found on the same 80 questions they are measured on, so v2 adds 15 **held-out** numeric
questions (operating income, cash, stockholders' equity) written before the v2 run and used for nothing else.

## Results

<!-- RESULTS:START -->
**Backend: ollama** (95 cases, run 2026-11-15)

| Metric | verifier | no_verifier |
|---|---|---|
| Model | qwen2.5:7b | qwen2.5:7b |
| Overall accuracy | 97.9% | 98.9% |
| Numeric accuracy | 100.0% | 100.0% |
| Numeric (held-out metrics) accuracy | 86.7% | 93.3% |
| Qualitative accuracy | 100.0% | 100.0% |
| Unanswerable accuracy | 100.0% | 100.0% |
| Adversarial accuracy | 100.0% | 100.0% |
| Figure present in retrieved passages (numeric) | 100.0% | 100.0% |
| Right section retrieved (qualitative) | 100.0% | 100.0% |
| Citations corrected by the verifier | 1 | 0 |
| Claims the verifier judged supported | 98.4% | n/a |
| False abstain rate (answerable questions) | 1.7% | 0.0% |
| Latency p50 / p95 (s) | 16.5 / 25.1 | 9.6 / 13.0 |
| LLM calls | 145 | 80 |

**v1 → v2 (Ollama, same 80 original questions).** v2 fixes: word-level BM25 tokenisation, verifier checks all retrieved passages, repairs wrong citations and requires the same line item; held-out numeric metrics added (v2 only).

| Variant | Accuracy v1 → v2 | Numeric v1 → v2 | False abstain v1 → v2 | Latency p50 v1 → v2 (s) |
|---|---|---|---|---|
| verifier | 91.2% → **100.0%** | 60.0% → 100.0% | 13.3% → 1.7% | 10.2 → 16.5 |
| no_verifier | 95.0% → **100.0%** | 73.3% → 100.0% | 4.4% → 0.0% | 7.6 → 9.6 |

_Verifier-vs-human calibration not run yet (see `analyst/judge_labels.py`)._

<!-- RESULTS:END -->

## Run it

**Local, free (Mac/Linux):** install [Ollama](https://ollama.com), open it, then

```bash
python3 run_and_push.py --limit 20     # trial
python3 run_and_push.py                # full run + push
```

It creates a Python 3.12 env with `uv`, downloads the filings from SEC EDGAR, pulls `qwen2.5:7b` and `nomic-embed-text`, runs tests and evals.

**Azure:** `az login`, then `./deploy/azure_deploy.sh`. It creates an Azure OpenAI resource with a chat and an embedding deployment, writes `.env.azure`, and deploys the API to Container Apps. Then `source .env.azure && python run_evals.py --backend azure`. Delete the resource group afterwards (the script prints the command).

**Databricks:** clone the repo as a Git folder, add `data/chunks.jsonl` and `data/sec/`, open `databricks/run_evals_databricks.py`, choose the serving endpoints your workspace has, run all. Results go to MLflow, and every graph step is traced.

**API:** `uvicorn analyst.api:app` then `POST /ask {"question": "...", "ticker": "NVDA"}`.

## Honest limits

- Five companies, one filing each, about 80 questions. That is enough to compare variants, not to claim general accuracy.
- The verifier uses an LLM as judge. Its agreement with a human is only known once `judge_labels.py` has been run on hand-labelled claims; until then, "supported by verifier" is the model's opinion.
- Numeric scoring accepts any number in the answer that matches the truth within 1%, so a long answer with many figures has a small chance of a lucky match.
- Qualitative scoring checks grounding and retrieval, not the quality of the prose.
- Tests run offline with a hashed-word embedder and a rule-based fake LLM, which checks the plumbing, not model quality.
