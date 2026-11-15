"""Expose the analyst as an MCP tool, so Claude Desktop / any MCP client can call it.  python -m analyst.mcp_server"""
import os

from mcp.server.fastmcp import FastMCP

from .agents import ask
from .pipeline import make

mcp = FastMCP("filings-analyst")
_, _, APP = make(os.environ.get("ANALYST_BACKEND", "ollama"))


@mcp.tool()
def ask_filings(question: str, ticker: str = "") -> dict:
    """Answer a question about a company's latest SEC 10-K (AAPL, MSFT, NVDA, AMZN, GOOGL) with verified citations."""
    out = ask(APP, question, ticker or None)
    out.pop("verdicts", None)
    return out


if __name__ == "__main__":
    mcp.run()
