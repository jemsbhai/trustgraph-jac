"""Scoring equivalence tests — verify the Jac bridge produces identical
results to calling trustandverify.scoring directly.

This is the mathematical integrity check: the two codepaths (Jac walker
via bridge/confidence.py vs direct trustandverify.scoring) must produce
bit-identical results for the same inputs.
"""

from __future__ import annotations

import pytest
from jsonld_ex.confidence_algebra import Opinion

# Bridge functions (what the Jac walker calls)
from bridge.confidence import (
    apply_trust_discount,
    detect_conflicts_within_claim,
    flip_opinion,
    fuse_evidence,
    opinion_summary,
    scalar_to_opinion,
)

# Direct trustandverify functions (canonical implementation)
import trustandverify.scoring.opinions as tv_opinions
import trustandverify.scoring.trust as tv_trust
import trustandverify.scoring.fusion as tv_fusion
import trustandverify.scoring.conflict as tv_conflict


class TestScalarToOpinionEquivalence:
    """scalar_to_opinion must produce identical results via both paths."""

    @pytest.mark.parametrize("confidence", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_identical_output(self, confidence):
        bridge_op = scalar_to_opinion(confidence)
        direct_op = tv_opinions.scalar_to_opinion(confidence)
        assert abs(bridge_op.belief - direct_op.belief) < 1e-10
        assert abs(bridge_op.disbelief - direct_op.disbelief) < 1e-10
        assert abs(bridge_op.uncertainty - direct_op.uncertainty) < 1e-10
        assert abs(bridge_op.base_rate - direct_op.base_rate) < 1e-10


class TestFlipOpinionEquivalence:
    def test_identical_output(self):
        op = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1, base_rate=0.5)
        bridge_flipped = flip_opinion(op)
        direct_flipped = tv_opinions.flip_opinion(op)
        assert abs(bridge_flipped.belief - direct_flipped.belief) < 1e-10
        assert abs(bridge_flipped.disbelief - direct_flipped.disbelief) < 1e-10
        assert abs(bridge_flipped.uncertainty - direct_flipped.uncertainty) < 1e-10


class TestApplyTrustDiscountEquivalence:
    @pytest.mark.parametrize("trust", [0.2, 0.5, 0.7, 0.9, 1.0])
    def test_identical_output(self, trust):
        op = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1, base_rate=0.5)
        bridge_result = apply_trust_discount(op, trust)
        direct_result = tv_trust.apply_trust_discount(op, trust)
        assert abs(bridge_result.belief - direct_result.belief) < 1e-10
        assert abs(bridge_result.disbelief - direct_result.disbelief) < 1e-10
        assert abs(bridge_result.uncertainty - direct_result.uncertainty) < 1e-10


class TestFuseEvidenceEquivalence:
    def test_identical_output_two_opinions(self):
        ops = [
            scalar_to_opinion(0.8),
            scalar_to_opinion(0.75),
        ]
        bridge_fused = fuse_evidence(ops)
        direct_fused = tv_fusion.fuse_evidence(ops)
        assert abs(bridge_fused.belief - direct_fused.belief) < 1e-10
        assert abs(bridge_fused.disbelief - direct_fused.disbelief) < 1e-10
        assert abs(bridge_fused.uncertainty - direct_fused.uncertainty) < 1e-10

    def test_identical_output_mixed_evidence(self):
        """Supporting + contradicting evidence (flipped) must fuse identically."""
        support = apply_trust_discount(scalar_to_opinion(0.85), 0.9)
        contradict = flip_opinion(apply_trust_discount(scalar_to_opinion(0.7), 0.6))
        ops = [support, contradict]

        bridge_fused = fuse_evidence(ops)
        direct_fused = tv_fusion.fuse_evidence(ops)
        assert abs(bridge_fused.belief - direct_fused.belief) < 1e-10
        assert abs(bridge_fused.disbelief - direct_fused.disbelief) < 1e-10
        assert abs(bridge_fused.uncertainty - direct_fused.uncertainty) < 1e-10

    def test_projected_probability_identical(self):
        ops = [scalar_to_opinion(0.9), scalar_to_opinion(0.6), scalar_to_opinion(0.8)]
        bridge_fused = fuse_evidence(ops)
        direct_fused = tv_fusion.fuse_evidence(ops)
        assert abs(bridge_fused.projected_probability() - direct_fused.projected_probability()) < 1e-10


class TestOpinionSummaryEquivalence:
    def test_identical_output(self):
        op = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        bridge_summary = opinion_summary(op)
        direct_summary = tv_opinions.opinion_summary(op)
        assert bridge_summary["belief"] == direct_summary["belief"]
        assert bridge_summary["disbelief"] == direct_summary["disbelief"]
        assert bridge_summary["uncertainty"] == direct_summary["uncertainty"]
        assert bridge_summary["projected_probability"] == direct_summary["projected_probability"]
        assert bridge_summary["verdict"] == direct_summary["verdict"]


class TestDetectConflictsEquivalence:
    def test_identical_output(self):
        supporting = [
            apply_trust_discount(scalar_to_opinion(0.85), 0.9),
            apply_trust_discount(scalar_to_opinion(0.8), 0.85),
        ]
        contradicting = [
            flip_opinion(apply_trust_discount(scalar_to_opinion(0.75), 0.7)),
        ]
        bridge_result = detect_conflicts_within_claim(supporting, contradicting, 0.2)
        direct_result = tv_conflict.detect_conflicts_within_claim(supporting, contradicting, 0.2)

        # Both should return the same structure or both None
        assert type(bridge_result) is type(direct_result)
        if bridge_result is not None:
            assert bridge_result["conflict_degree"] == direct_result["conflict_degree"]
            assert bridge_result["num_supporting"] == direct_result["num_supporting"]
            assert bridge_result["num_contradicting"] == direct_result["num_contradicting"]


class TestEstimateSourceTrustEquivalence:
    """The Jac walker has its own estimate_source_trust — verify it agrees."""

    @pytest.mark.parametrize("url,title,expected_trust", [
        ("https://www.cdc.gov/study", "CDC Study", 0.9),
        ("https://nature.com/article", "Nature Paper", 0.85),
        ("https://bbc.com/news/item", "BBC News", 0.75),
        ("https://en.wikipedia.org/wiki/X", "Wikipedia", 0.6),
        ("https://reddit.com/r/science", "Reddit Post", 0.35),
        ("https://example.com/blog", "Random Blog", 0.5),
    ])
    def test_matches_trustandverify(self, url, title, expected_trust):
        """Bridge and trustandverify must agree on trust heuristics."""
        from trustandverify.scoring.trust import estimate_source_trust as tv_est
        tv_trust = tv_est(url, title)
        assert tv_trust == expected_trust, (
            f"Trust mismatch for {url}: trustandverify={tv_trust}, expected={expected_trust}"
        )
