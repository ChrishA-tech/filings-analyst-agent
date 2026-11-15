"""10-K HTML -> plain text -> sections (Item 1, 1A, 7, 8, ...) -> overlapping chunks with metadata."""
from __future__ import annotations

import html as htmllib
import re
from html.parser import HTMLParser
from typing import Dict, List

from . import config as C

_BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.stack = [], 0, []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "div"):
            style = (dict(attrs).get("style") or "").replace(" ", "").lower()
            hide = tag != "div" or "display:none" in style
            self.stack.append(hide)
            self.skip += hide
        if tag in _BLOCK:
            self.out.append("\n")
        if tag in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "div") and self.stack:
            self.skip -= self.stack.pop()
        if tag in _BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_to_text(raw: str) -> str:
    p = _Text()
    p.feed(raw)
    t = htmllib.unescape("".join(p.out)).replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()


_ITEM = re.compile(r"^[\s|]*item\s+(1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9|10|11|12|13|14|15|16)[\.\s:]", re.I | re.M)


def sections(text: str) -> Dict[str, str]:
    """Split on 'Item N.' headings; keeps the LAST occurrence of each item (the first is usually the contents page)."""
    hits = [(m.start(), m.group(1).upper()) for m in _ITEM.finditer(text)]
    out: Dict[str, str] = {}
    for i, (pos, item) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        body = text[pos:end].strip()
        if len(body) > len(out.get(item, "")):
            out[item] = body
    return out or {"FULL": text}


def chunk(text: str, size: int = C.CHUNK_CHARS, overlap: int = C.CHUNK_OVERLAP) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size:                         # very long paragraph (or table): hard split
            chunks.append((cur + "\n\n" + p[:size]).strip() if cur else p[:size])
            cur, p = "", p[size - overlap:]
        if len(cur) + len(p) + 2 > size and cur:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n\n" + p
        else:
            cur = (cur + "\n\n" + p).strip()
    if cur:
        chunks.append(cur)
    return chunks


def documents(ticker: str, company: str, year: str, raw_html: str) -> List[Dict]:
    docs = []
    for item, body in sections(html_to_text(raw_html)).items():
        for i, c in enumerate(chunk(body)):
            docs.append({"id": f"{ticker}-{year}-item{item}-{i:04d}", "text": c,
                         "meta": {"ticker": ticker, "company": company, "fiscal_year": year, "item": item}})
    return docs
