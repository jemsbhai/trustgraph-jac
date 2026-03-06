"""Tests for bridge/confidence.py — the drop-in replacement for the hackathon bridge."""

import pytest
from jsonld_ex.confidence_algebra import Opinion

from bridge.confidence import (
    apply_trust_discount,
    build_jsonld_claim,
    detect_conflicts,
    detect_conflicts_within_claim,
    flip_opinion,
    fuse_evidence,
    opinion_summary,
    scalar_to_opinion,
)


# ── Re-export sanity checks ──────────────────────────────────────────────────
# These verify that the bridge module actually re-exports the trustandverify
# scoring functions (not None, not some other object).


class TestReExports:
    """All re-exported functions should be callable and match trustandverify."""

    def test_scalar_to_opinion_is_callable(self):
        assert callable(scalar_to_opinion)

    def test_flip_opinion_is_callable(self):
        assert callable(flip_opinion)

    def test_fuse_evidence_is_callable(self):
        assert callable(fuse_evidence)

    def test_apply_trust_discount_is_callable(self):
        assert callable(apply_trust_discount)

    def test_detect_conflicts_within_claim_is_callable(self):
        assert callable(detect_conflicts_within_claim)

    def test_opinion_summary_is_callable(self):
        assert callable(opinion_summary)

    def test_scalar_to_opinion_returns_opinion(self):
        op = scalar_to_opinion(0.8)
        assert isinstance(op, Opinion)
        assert op.belief > 0

    def test_flip_opinion_swaps_belief_disbelief(self):
        op = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1, base_rate=0.5)
        flipped = flip_opinion(op)
        assert abs(flipped.belief - 0.2) < 1e-6
        assert abs(flipped.disbelief - 0.7) < 1e-6

    def test_fuse_evidence_reduces_uncertainty(self):
        ops = [
            scalar_to_opinion(0.8),
            scalar_to_opinion(0.75),
            scalar_to_opinion(0.85),
        ]
        fused = fuse_evidence(ops)
        # Fusion of agreeing evidence should reduce uncertainty
        assert fused.uncertainty < ops[0].uncertainty


# ── build_jsonld_claim ────────────────────────────────────────────────────────


class TestBuildJsonldClaim:
    """Tests for the JSON-LD document builder."""

    def test_basic_structure(self, high_belief_opinion, sample_sources):
        doc = build_jsonld_claim("Test claim", high_belief_opinion, sample_sources)
        assert doc["@type"] == "ex:VerifiedClaim"
        assert doc["ex:claimText"] == "Test claim"
        assert "ex:confidence" in doc
        assert "prov:wasGeneratedBy" in doc
        assert doc["ex:sources"] is sample_sources

    def test_confidence_fields(self, high_belief_opinion, sample_sources):
        doc = build_jsonld_claim("Claim", high_belief_opinion, sample_sources)
        conf = doc["ex:confidence"]
        assert conf["@type"] == "ex:SubjectiveOpinion"
        assert conf["ex:belief"] == round(high_belief_opinion.belief, 4)
        assert conf["ex:disbelief"] == round(high_belief_opinion.disbelief, 4)
        assert conf["ex:uncertainty"] == round(high_belief_opinion.uncertainty, 4)
        assert conf["ex:baseRate"] == round(high_belief_opinion.base_rate, 4)

    def test_projected_probability_calculation(self, high_belief_opinion, sample_sources):
        doc = build_jsonld_claim("Claim", high_belief_opinion, sample_sources)
        expected = high_belief_opinion.belief + high_belief_opinion.base_rate * high_belief_opinion.uncertainty
        assert doc["ex:confidence"]["ex:projectedProbability"] == round(expected, 4)

    def test_provenance_fields(self, high_belief_opinion, sample_sources):
        doc = build_jsonld_claim("Claim", high_belief_opinion, sample_sources)
        prov = doc["prov:wasGeneratedBy"]
        assert prov["@type"] == "prov:Activity"
        assert prov["prov:wasAssociatedWith"] == "TrustGraph Agent"
        assert "prov:endedAtTime" in prov

    def test_no_conflict_key_when_none(self, high_belief_opinion, sample_sources):
        doc = build_jsonld_claim("Claim", high_belief_opinion, sample_sources, conflict=None)
        assert "ex:conflict" not in doc

    def test_conflict_included_when_provided(self, high_belief_opinion, sample_sources):
        conflict = {"conflict_degree": 0.45, "num_supporting": 2, "num_contradicting": 1}
        doc = build_jsonld_claim("Claim", high_belief_opinion, sample_sources, conflict=conflict)
        assert doc["ex:conflict"] == conflict

    def test_vacuous_opinion(self, vacuous_opinion, sample_sources):
        doc = build_jsonld_claim("Unknown claim", vacuous_opinion, sample_sources)
        conf = doc["ex:confidence"]
        assert conf["ex:belief"] == 0.0
        assert conf["ex:disbelief"] == 0.0
        assert conf["ex:uncertainty"] == 1.0
        assert conf["ex:projectedProbability"] == 0.5  # base_rate * 1.0

    def test_empty_sources_list(self, high_belief_opinion):
        doc = build_jsonld_claim("Claim", high_belief_opinion, [])
        assert doc["ex:sources"] == []

    def test_values_are_rounded_to_4dp(self):
        # Use an opinion with values that have many decimal places
        op = Opinion(belief=1/3, disbelief=4/9, uncertainty=2/9, base_rate=0.5)
        doc = build_jsonld_claim("Claim", op, [])
        conf = doc["ex:confidence"]
        assert conf["ex:belief"] == 0.3333
        assert conf["ex:disbelief"] == 0.4444
        assert conf["ex:uncertainty"] == 0.2222


# ── detect_conflicts (deprecated) ────────────────────────────────────────────


class TestDetectConflictsDeprecated:
    """Tests for the deprecated pairwise detect_conflicts function."""

    def test_no_conflicts_with_agreeing_opinions(self):
        ops = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1, base_rate=0.5),
            Opinion(belief=0.75, disbelief=0.15, uncertainty=0.1, base_rate=0.5),
        ]
        result = detect_conflicts(ops, threshold=0.3)
        assert result == []

    def test_detects_conflict_between_opposing_opinions(self):
        ops = [
            Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05, base_rate=0.5),
            Opinion(belief=0.05, disbelief=0.9, uncertainty=0.05, base_rate=0.5),
        ]
        result = detect_conflicts(ops, threshold=0.3)
        assert len(result) >= 1
        assert result[0]["pair"] == (0, 1)
        assert result[0]["conflict_degree"] > 0.3

    def test_conflict_structure(self):
        ops = [
            Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05, base_rate=0.5),
            Opinion(belief=0.05, disbelief=0.9, uncertainty=0.05, base_rate=0.5),
        ]
        result = detect_conflicts(ops, threshold=0.0)
        c = result[0]
        assert "pair" in c
        assert "conflict_degree" in c
        assert "opinion_a" in c
        assert "opinion_b" in c
        assert "belief" in c["opinion_a"]
        assert "disbelief" in c["opinion_a"]
        assert "uncertainty" in c["opinion_a"]

    def test_empty_list(self):
        assert detect_conflicts([], threshold=0.3) == []

    def test_single_opinion(self):
        ops = [Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1, base_rate=0.5)]
        assert detect_conflicts(ops, threshold=0.3) == []

    def test_threshold_sensitivity(self):
        """Higher threshold should find fewer conflicts."""
        ops = [
            Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1, base_rate=0.5),
            Opinion(belief=0.2, disbelief=0.7, uncertainty=0.1, base_rate=0.5),
        ]
        low_threshold = detect_conflicts(ops, threshold=0.1)
        high_threshold = detect_conflicts(ops, threshold=0.9)
        assert len(low_threshold) >= len(high_threshold)

    def test_values_rounded_to_4dp(self):
        ops = [
            Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05, base_rate=0.5),
            Opinion(belief=0.05, disbelief=0.9, uncertainty=0.05, base_rate=0.5),
        ]
        result = detect_conflicts(ops, threshold=0.0)
        c = result[0]
        # All values should have at most 4 decimal places
        assert c["conflict_degree"] == round(c["conflict_degree"], 4)
        assert c["opinion_a"]["belief"] == round(c["opinion_a"]["belief"], 4)
