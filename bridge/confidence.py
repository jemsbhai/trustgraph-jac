"""TrustGraph confidence bridge — drop-in replacement for the hackathon version.

All functions delegate to trustandverify.scoring, which is the properly
packaged, tested implementation. The function signatures are identical to
the hackathon bridge/confidence.py so existing Jac import statements work
unchanged:

    import from bridge.confidence {
        scalar_to_opinion, flip_opinion, fuse_evidence,
        apply_trust_discount, detect_conflicts_within_claim,
        opinion_summary, build_jsonld_claim
    }
"""

from __future__ import annotations

from datetime import datetime, timezone

# Re-export everything the Jac walker imports from this module.
# trustandverify.scoring is the canonical implementation.
from trustandverify.scoring import (
    apply_trust_discount,
    detect_conflicts_within_claim,
    flip_opinion,
    fuse_evidence,
    opinion_summary,
    scalar_to_opinion,
)
from trustandverify.scoring.conflict import detect_conflicts_within_claim  # noqa: F811


def build_jsonld_claim(
    claim_text: str,
    opinion: object,
    sources: list[dict],
    conflict: dict | None = None,
) -> dict:
    """Build a JSON-LD document for a scored claim.

    Identical output to the hackathon version. Uses trustandverify's
    Opinion.projected_probability() method.
    """
    proj = float(opinion.belief + opinion.base_rate * opinion.uncertainty)
    doc: dict = {
        "@type": "ex:VerifiedClaim",
        "ex:claimText": claim_text,
        "ex:confidence": {
            "@type": "ex:SubjectiveOpinion",
            "ex:belief": round(float(opinion.belief), 4),
            "ex:disbelief": round(float(opinion.disbelief), 4),
            "ex:uncertainty": round(float(opinion.uncertainty), 4),
            "ex:baseRate": round(float(opinion.base_rate), 4),
            "ex:projectedProbability": round(proj, 4),
        },
        "prov:wasGeneratedBy": {
            "@type": "prov:Activity",
            "prov:wasAssociatedWith": "TrustGraph Agent",
            "prov:endedAtTime": datetime.now(timezone.utc).isoformat(),
        },
        "ex:sources": sources,
    }
    if conflict:
        doc["ex:conflict"] = conflict
    return doc


# Keep detect_conflicts for backward compatibility (deprecated, but not removed)
def detect_conflicts(opinions: list, threshold: float = 0.3) -> list[dict]:
    """Deprecated — use detect_conflicts_within_claim instead.

    Kept for backward compatibility with any code that imported this
    from the hackathon bridge.
    """
    from trustandverify.scoring.conflict import pairwise_conflict
    conflicts = []
    for i in range(len(opinions)):
        for j in range(i + 1, len(opinions)):
            conf = pairwise_conflict(opinions[i], opinions[j])
            if conf > threshold:
                conflicts.append({
                    "pair": (i, j),
                    "conflict_degree": round(float(conf), 4),
                    "opinion_a": {
                        "belief": round(float(opinions[i].belief), 4),
                        "disbelief": round(float(opinions[i].disbelief), 4),
                        "uncertainty": round(float(opinions[i].uncertainty), 4),
                    },
                    "opinion_b": {
                        "belief": round(float(opinions[j].belief), 4),
                        "disbelief": round(float(opinions[j].disbelief), 4),
                        "uncertainty": round(float(opinions[j].uncertainty), 4),
                    },
                })
    return conflicts
