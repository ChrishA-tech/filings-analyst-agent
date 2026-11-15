"""Download the latest 10-K of each company from SEC EDGAR, and XBRL facts used as evaluation ground truth."""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import date
from pathlib import Path
from typing import Dict, Optional

from . import config as C


def _get(url: str, retries: int = 3) -> bytes:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": C.SEC_USER_AGENT, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


def latest_10k(cik: int) -> Dict:
    sub = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
    r = sub["filings"]["recent"]
    for i, form in enumerate(r["form"]):
        if form == "10-K":
            return {"accession": r["accessionNumber"][i], "primary": r["primaryDocument"][i],
                    "report_date": r["reportDate"][i], "filing_date": r["filingDate"][i]}
    raise ValueError(f"No 10-K found for CIK {cik}")


def download(out: Path = None) -> Dict[str, Dict]:
    out = out or C.DATA / "sec"
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for ticker, (name, cik) in C.COMPANIES.items():
        f = latest_10k(cik)
        acc = f["accession"].replace("-", "")
        html = _get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{f['primary']}")
        (out / f"{ticker}_10k.html").write_bytes(html)
        facts = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        (out / f"{ticker}_facts.json").write_bytes(facts)
        meta[ticker] = {"name": name, "cik": cik, **f}
        time.sleep(0.5)                      # stay well under SEC's 10 requests/second limit
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


REVENUE_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenuesNetOfInterestExpense")


def fy_fact(facts: Dict, tags, accession: str, report_date: str, instant: bool = False) -> Optional[float]:
    """Value reported in THIS 10-K for the fiscal year ending report_date (12-month duration, or instant)."""
    end = date.fromisoformat(report_date)
    for tag in tags:
        units = facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD", [])
        for u in units:
            if u.get("accn") != accession or u.get("end") != report_date:
                continue
            if instant and "start" not in u:
                return float(u["val"])
            if not instant and "start" in u and 350 <= (end - date.fromisoformat(u["start"])).days <= 380:
                return float(u["val"])
    return None


if __name__ == "__main__":
    for t, m in download().items():
        print(t, m["report_date"], m["accession"])
