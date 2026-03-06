"""Tests for bridge/exports.py — JSON-LD to Report conversion and file export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jsonld_ex.confidence_algebra import Opinion

from bridge.exports import jsonld_to_report, save_exports
from trustandverify.core.models import Claim, Conflict, Report, Verdict


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_jsonld() -> dict:
    """Minimal JSON-LD output matching what the Jac walker produces."""
    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "ex": "https://jsonld-ex.org/vocab#",
            "prov": "http://www.w3.org/ns/prov#",
        },
        "@type": "ex:TrustGraphReport",
        "ex:query": "Is coffee healthy?",
        "ex:generatedAt": "2026-03-01T12:00:00+00:00",
        "ex:claims": [
            {
                "@type": "ex:VerifiedClaim",
                "ex:claimText": "Coffee contains antioxidants.",
                "ex:confidence": {
                    "@type": "ex:SubjectiveOpinion",
                    "ex:belief": 0.7,
                    "ex:disbelief": 0.1,
                    "ex:uncertainty": 0.2,
                    "ex:baseRate": 0.5,
                    "ex:projectedProbability": 0.8,
                },
                "ex:sources": [],
            },
            {
                "@type": "ex:VerifiedClaim",
                "ex:claimText": "Coffee causes insomnia.",
                "ex:confidence": {
                    "@type": "ex:SubjectiveOpinion",
                    "ex:belief": 0.3,
                    "ex:disbelief": 0.4,
                    "ex:uncertainty": 0.3,
                    "ex:baseRate": 0.5,
                    "ex:projectedProbability": 0.45,
                },
                "ex:sources": [],
            },
        ],
        "ex:conflicts": [
            {
                "claim": "Coffee causes insomnia",
                "conflict_degree": 0.35,
                "num_supporting": 2,
                "num_contradicting": 1,
            },
        ],
        "ex:summary": "Coffee has health benefits but may cause sleep issues.",
    }


# ── jsonld_to_report ──────────────────────────────────────────────────────────


class TestJsonldToReport:
    def test_returns_report(self, sample_jsonld):
        report = jsonld_to_report(sample_jsonld, "Is coffee healthy?")
        assert isinstance(report, Report)

    def test_query_preserved(self, sample_jsonld):
        report = jsonld_to_report(sample_jsonld, "Is coffee healthy?")
        assert report.query == "Is coffee healthy?"

    def test_summary_preserved(self, sample_jsonld):
        report = jsonld_to_report(sample_jsonld, "q")
        assert report.summary == "Coffee has health benefits but may cause sleep issues."

    def test_claims_count(self, sample_jsonld):
        report = jsonld_to_report(sample_jsonld, "q")
        assert len(report.claims) == 2

    def test_claim_text(self, sample_jsonld):
        report = jsonld_to_report(sample_jsonld, "q")
        assert report.claims[0].text == "Coffee contains antioxidants."
        assert report.claims[1].text == "Coffee causes insomnia."

    def test_claim_opinions_round_trip(self, sample_jsonld):
        report = jsonld_to_report(sample_jsonld, "q")
        op = report.claims[0].opinion
        assert isinstance(op, Opinion)
        assert abs(op.belief - 0.7) < 1e-6
        assert abs(op.disbelief - 0.1) < 1e-6
        assert abs(op.uncertainty - 0.2) < 1e-6

    def test_opinion_additivity(self, sample_jsonld):
        """b + d + u must equal 1.0 for every claim opinion."""
        report = jsonld_to_report(sample_jsonld, "q")
        for claim in report.claims:
            total = claim.opinion.belief + claim.opinion.disbelief + claim.opinion.uncertainty
            assert abs(total - 1.0) < 1e-6

    def test_verdict_supported(self, sample_jsonld):
        """P >= 0.7 should produce SUPPORTED."""
        report = jsonld_to_report(sample_jsonld, "q")
        assert report.claims[0].verdict == Verdict.SUPPORTED

    def test_verdict_contested(self, sample_jsonld):
        """0.3 < P < 0.7 should produce CONTESTED."""
        report = jsonld_to_report(sample_jsonld, "q")
        assert report.claims[1].verdict == Verdict.CONTESTED

    def test_verdict_refuted(self):
        """P <= 0.3 should produce REFUTED."""
        jsonld = {
            "ex:claims": [{
                "ex:claimText": "False claim",
                "ex:confidence": {
                    "ex:belief": 0.05,
                    "ex:disbelief": 0.9,
                    "ex:uncertainty": 0.05,
                    "ex:baseRate": 0.5,
                },
            }],
            "ex:conflicts": [],
            "ex:summary": "",
        }
        report = jsonld_to_report(jsonld, "q")
        assert report.claims[0].verdict == Verdict.REFUTED

    def test_conflicts_count(self, sample_jsonld):
        report = jsonld_to_report(sample_jsonld, "q")
        assert len(report.conflicts) == 1

    def test_conflict_fields(self, sample_jsonld):
        report = jsonld_to_report(sample_jsonld, "q")
        c = report.conflicts[0]
        assert c.claim_text == "Coffee causes insomnia"
        assert c.conflict_degree == 0.35
        assert c.num_supporting == 2
        assert c.num_contradicting == 1

    def test_empty_claims(self):
        jsonld = {"ex:claims": [], "ex:conflicts": [], "ex:summary": "Nothing."}
        report = jsonld_to_report(jsonld, "q")
        assert report.claims == []
        assert report.conflicts == []

    def test_missing_confidence_defaults_to_vacuous(self):
        """Claim without ex:confidence should get a vacuous opinion."""
        jsonld = {
            "ex:claims": [{"ex:claimText": "No confidence"}],
            "ex:conflicts": [],
            "ex:summary": "",
        }
        report = jsonld_to_report(jsonld, "q")
        op = report.claims[0].opinion
        assert abs(op.uncertainty - 1.0) < 1e-6

    def test_report_has_id(self, sample_jsonld):
        report = jsonld_to_report(sample_jsonld, "q")
        assert report.id is not None
        assert len(report.id) > 0

    def test_report_has_created_at(self, sample_jsonld):
        report = jsonld_to_report(sample_jsonld, "q")
        assert isinstance(report.created_at, datetime)


# ── save_exports ──────────────────────────────────────────────────────────────


class TestSaveExports:
    def test_writes_json_file(self, sample_jsonld, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        report = save_exports(sample_jsonld, "Is coffee healthy?")
        assert (tmp_path / "output.json").exists()
        data = json.loads((tmp_path / "output.json").read_text())
        assert data["ex:query"] == "Is coffee healthy?"

    def test_returns_report(self, sample_jsonld, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        report = save_exports(sample_jsonld, "q")
        assert isinstance(report, Report)

    def test_markdown_format(self, sample_jsonld, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_exports(sample_jsonld, "q", format="markdown")
        assert (tmp_path / "output.md").exists()
        content = (tmp_path / "output.md").read_text(encoding="utf-8")
        assert "TrustGraph" in content

    def test_html_format(self, sample_jsonld, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_exports(sample_jsonld, "q", format="html")
        assert (tmp_path / "output.html").exists()
        content = (tmp_path / "output.html").read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_all_format(self, sample_jsonld, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_exports(sample_jsonld, "q", format="all")
        assert (tmp_path / "output.json").exists()
        assert (tmp_path / "output.md").exists()
        assert (tmp_path / "output.html").exists()

    def test_jsonld_only_no_extra_files(self, sample_jsonld, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_exports(sample_jsonld, "q", format="jsonld")
        assert (tmp_path / "output.json").exists()
        assert not (tmp_path / "output.md").exists()
        assert not (tmp_path / "output.html").exists()

    def test_sqlite_storage(self, sample_jsonld, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        db_path = str(tmp_path / "test.db")
        report = save_exports(
            sample_jsonld, "Is coffee healthy?",
            storage_backend="sqlite", db_path=db_path,
        )
        # Verify report was persisted
        from trustandverify.storage.sqlite import SQLiteStorage
        import asyncio
        storage = SQLiteStorage(db_path)
        retrieved = asyncio.run(storage.get_report(report.id))
        assert retrieved is not None
        assert retrieved.query == "Is coffee healthy?"

    def test_memory_storage_no_db_file(self, sample_jsonld, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        save_exports(sample_jsonld, "q", storage_backend="memory")
        # No .db file should be created
        assert not (tmp_path / "trustgraph.db").exists()
