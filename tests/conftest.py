import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HTML = """<html><body><div style="display:none">hidden xbrl junk</div>
<p>Table of Contents</p><p>Item 1. Business</p><p>Item 1A. Risk Factors</p><p>Item 8. Financial Statements</p>
<p>Item 1. Business</p><p>{name} designs widgets and sells cloud widget services worldwide. {name} had about 1,200 full-time employees.</p>
<p>Item 1A. Risk Factors</p><p>Competition in the widget market is intense and could reduce {name} margins.</p>
<p>A cybersecurity breach or data breach could harm {name} reputation and operations.</p>
<p>Item 7. Management's Discussion and Analysis</p><p>Total revenue increased because widget demand grew.</p>
<p>Item 8. Financial Statements</p><table><tr><td>Total revenue</td><td>$</td><td>{rev:,}</td></tr>
<tr><td>Net income</td><td>$</td><td>{ni:,}</td></tr></table>
<p>{name} total net sales were $ {rev:,} million for the fiscal year.</p></body></html>"""


def facts(acc, end, rev, ni, assets):
    def u(v, start=True):
        d = {"val": v, "accn": acc, "end": end}
        if start:
            d["start"] = "2024-01-01"
        return {"units": {"USD": [d]}}
    return {"facts": {"us-gaap": {"Revenues": u(rev * 1_000_000), "NetIncomeLoss": u(ni * 1_000_000),
                                  "Assets": u(assets, start=False)}}}


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    from analyst import config as C
    sec = tmp_path / "sec"
    sec.mkdir()
    meta = {}
    for t, name, rev, ni in [("AAA", "Acme", 5_000, 700), ("BBB", "Bolt", 9_100, 1_200)]:
        (sec / f"{t}_10k.html").write_text(HTML.format(name=name, rev=rev, ni=ni))
        (sec / f"{t}_facts.json").write_text(json.dumps(facts(f"acc-{t}", "2024-12-31", rev, ni, 42_000_000)))
        meta[t] = {"name": name, "cik": 1, "accession": f"acc-{t}", "primary": "x.htm", "report_date": "2024-12-31",
                   "filing_date": "2025-02-01"}
    (sec / "meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(C, "DATA", tmp_path)
    from analyst.pipeline import build_corpus
    return sec, build_corpus(sec, tmp_path / "chunks.jsonl")
