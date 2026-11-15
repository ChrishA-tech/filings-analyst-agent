"""Corpus, retrieval and backend settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INDEX_DIR = DATA / "chroma"
RESULTS = ROOT / "results"

# Latest 10-K of each company (SEC EDGAR). CIKs from https://www.sec.gov/cgi-bin/browse-edgar
COMPANIES = {"AAPL": ("Apple", 320193), "MSFT": ("Microsoft", 789019), "NVDA": ("NVIDIA", 1045810),
             "AMZN": ("Amazon", 1018724), "GOOGL": ("Alphabet", 1652044)}
# Not JPMorgan: its 10-K body is a cross-reference shell, the content sits in the annual-report exhibit.
# SEC requires a descriptive User-Agent with contact details for automated access.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "Chrish Pep Felix A research project chrishpepfelixa@gmail.com")

CHUNK_CHARS, CHUNK_OVERLAP = 1500, 200
TOP_K = 6
RRF_K = 60

# Backends: all speak the OpenAI-compatible chat/embeddings API.
BACKENDS = {
    "ollama": {"base_url": os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/v1", "key_env": None,
               "chat": os.environ.get("OLLAMA_CHAT", "qwen2.5:7b"), "embed": os.environ.get("OLLAMA_EMBED", "nomic-embed-text")},
    "azure": {"base_url": os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/") + "/openai/v1", "key_env": "AZURE_OPENAI_API_KEY",
              "chat": os.environ.get("AZURE_OPENAI_CHAT", "gpt-4.1-mini"), "embed": os.environ.get("AZURE_OPENAI_EMBED", "text-embedding-3-small")},
    "databricks": {"base_url": os.environ.get("DATABRICKS_HOST", "").rstrip("/") + "/serving-endpoints", "key_env": "DATABRICKS_TOKEN",
                   "chat": os.environ.get("DATABRICKS_CHAT", "databricks-meta-llama-3-3-70b-instruct"),
                   "embed": os.environ.get("DATABRICKS_EMBED", "databricks-gte-large-en")},
}
