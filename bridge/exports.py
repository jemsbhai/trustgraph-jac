"""TrustGraph export bridge — converts Jac walker JSON-LD output to trustandverify Reports.

Extracts the save_exports() logic from trustgraph.jac into testable Python.
The Jac walker calls this module instead of inlining the export logic.

Usage from Jac:
    import from bridge.exports { save_exports }
    save_exports(jsonld_output, query_text, "markdown", "sqlite", "trustgraph.db")
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

from jsonld_ex.confidence_algebra import Opinion

from trustandverify.core.models import Claim, Conflict, Report, Verdict
from trustandverify.export.html import HtmlExporter
from trustandverify.export.jsonld import JsonLdExporter
from trustandverify.export.markdown import MarkdownExporter


def jsonld_to_report(jsonld_output: dict, query_text: str) -> Report:
    """Convert a Jac walker's JSON-LD output dict into a trustandverify Report.

    This is the bridge between the Jac walker's output format and
    trustandverify's data model — allowing re-use of all exporters
    and storage backends.
    """
    tv_claims: list[Claim] = []
    for jc in jsonld_output.get("ex:claims", []):
        conf = jc.get("ex:confidence", {})
        op = Opinion(
            belief=conf.get("ex:belief", 0.0),
            disbelief=conf.get("ex:disbelief", 0.0),
            uncertainty=conf.get("ex:uncertainty", 1.0),
            base_rate=conf.get("ex:baseRate", 0.5),
        )
        p = op.projected_probability()
        if p >= 0.7:
            v = Verdict.SUPPORTED
        elif p > 0.3:
            v = Verdict.CONTESTED
        else:
            v = Verdict.REFUTED

        tv_claims.append(Claim(text=jc.get("ex:claimText", ""), opinion=op, verdict=v))

    tv_conflicts: list[Conflict] = []
    for c in jsonld_output.get("ex:conflicts", []):
        tv_conflicts.append(Conflict(
            claim_text=c.get("claim", ""),
            conflict_degree=c.get("conflict_degree", 0.0),
            num_supporting=c.get("num_supporting", 0),
            num_contradicting=c.get("num_contradicting", 0),
        ))

    return Report(
        id=str(uuid.uuid4()),
        query=query_text,
        claims=tv_claims,
        conflicts=tv_conflicts,
        summary=jsonld_output.get("ex:summary", ""),
        created_at=datetime.now(timezone.utc),
    )


def save_exports(
    jsonld_output: dict,
    query_text: str,
    format: str = "jsonld",
    storage_backend: str = "memory",
    db_path: str = "trustgraph.db",
) -> Report:
    """Save report in requested formats and optionally persist to storage.

    Args:
        jsonld_output:   The JSON-LD dict produced by the Jac walker.
        query_text:      The original research question.
        format:          Export format — jsonld, markdown, html, or all.
        storage_backend: Storage backend — memory or sqlite.
        db_path:         Path to SQLite DB (only used if storage_backend='sqlite').

    Returns:
        The trustandverify Report object (for further use by the caller).
    """
    report = jsonld_to_report(jsonld_output, query_text)

    # JSON-LD (always saved)
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(jsonld_output, f, indent=2)
    print("  JSON-LD saved to output.json")

    if format in ("markdown", "all"):
        MarkdownExporter().render_to_file(report, "output.md")
        print("  Markdown saved to output.md")

    if format in ("html", "all"):
        HtmlExporter().render_to_file(report, "output.html")
        print("  HTML saved to output.html")

    # Storage
    if storage_backend == "sqlite":
        from trustandverify.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(db_path)
        asyncio.run(storage.save_report(report))
        print(f"  Report saved to SQLite: {db_path} (id={report.id})")

    return report
