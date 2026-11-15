# Databricks notebook source
# MAGIC %md
# MAGIC # filings-analyst on Databricks
# MAGIC Runs the same researcher + verifier evaluation with a Databricks foundation-model endpoint, logs the run to MLflow
# MAGIC and traces every LangGraph step with MLflow Tracing.
# MAGIC
# MAGIC 1. Repos / Git folder: clone `https://github.com/ChrishA-tech/filings-analyst-agent`
# MAGIC 2. Nothing to upload: the notebook uses the committed `data/chunks.jsonl` and `results/cases.jsonl`.
# MAGIC 3. Pick endpoints that exist in **Serving** in your workspace (Free Edition availability varies) and run all.

# COMMAND ----------
# MAGIC %pip install -q -r ../requirements.txt mlflow
# MAGIC %restart_python

# COMMAND ----------
import json, os, sys
sys.path.insert(0, os.path.abspath(".."))
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DATABRICKS_TOKEN"] = ctx.apiToken().get()
dbutils.widgets.text("chat", "databricks-meta-llama-3-3-70b-instruct")
dbutils.widgets.text("embed", "databricks-gte-large-en")
dbutils.widgets.text("limit", "0")
os.environ["DATABRICKS_CHAT"], os.environ["DATABRICKS_EMBED"] = dbutils.widgets.get("chat"), dbutils.widgets.get("embed")

# COMMAND ----------
import importlib, mlflow
from analyst import config as C
importlib.reload(C)
from analyst.evals import run, summarise
from analyst.pipeline import make

mlflow.langchain.autolog()          # traces every LangGraph node and LLM call
mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/filings-analyst")
cases = [json.loads(l) for l in open("../results/cases.jsonl") if l.strip()]   # same questions as the local run
lim = int(dbutils.widgets.get("limit"))
if lim:
    per = max(1, lim // 4)
    cases = [c for cat in ("numeric", "numeric_new", "qualitative", "unanswerable", "adversarial") for c in [x for x in cases if x["category"] == cat][:per]]

report = {"backend": "databricks", "n_cases": len(cases), "variants": {}}
for verify in (True, False):
    name = "verifier" if verify else "no_verifier"
    with mlflow.start_run(run_name=name):
        index, llm, app = make("databricks", verify=verify, persist=False)
        rows = run(app, index, cases)
        s = summarise(rows)
        s.update(model=llm.chat_model, llm_calls=llm.usage.calls, prompt_tokens=llm.usage.prompt, completion_tokens=llm.usage.completion)
        mlflow.log_params({"chat": llm.chat_model, "embed": llm.embed_model, "verifier": verify, "n_cases": len(cases)})
        mlflow.log_metrics({k: v for k, v in s.items() if isinstance(v, (int, float))})
        for cat, m in s["by_category"].items():
            mlflow.log_metric(f"{cat}_accuracy", m["accuracy"])
        mlflow.log_dict(rows, f"rows_{name}.json")
        report["variants"][name] = s
print(json.dumps(report, indent=2))

# COMMAND ----------
# Copy the printed JSON into results/summary_databricks.json on your Mac (Git folders are read-only for results).
report["date"] = __import__("time").strftime("%Y-%m-%d")
print("COPY EVERYTHING BELOW into results/summary_databricks.json on your Mac")
print(json.dumps(report))
