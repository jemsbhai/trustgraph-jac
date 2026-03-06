"""Shared fixtures for trustgraph-jac tests."""

import pytest
from jsonld_ex.confidence_algebra import Opinion


@pytest.fixture
def high_belief_opinion():
    """Opinion with high belief (confident support)."""
    return Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1, base_rate=0.5)


@pytest.fixture
def low_belief_opinion():
    """Opinion with low belief (confident contradiction)."""
    return Opinion(belief=0.1, disbelief=0.8, uncertainty=0.1, base_rate=0.5)


@pytest.fixture
def uncertain_opinion():
    """Opinion with high uncertainty (little evidence)."""
    return Opinion(belief=0.2, disbelief=0.1, uncertainty=0.7, base_rate=0.5)


@pytest.fixture
def vacuous_opinion():
    """Vacuous opinion — total ignorance."""
    return Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)


@pytest.fixture
def sample_sources():
    """Sample source list for build_jsonld_claim tests."""
    return [
        {
            "title": "Nature Study on Topic",
            "url": "https://nature.com/article/123",
            "trust_score": 0.85,
            "evidence": "Evidence text here",
            "supports": True,
        },
        {
            "title": "BBC Report",
            "url": "https://bbc.com/news/456",
            "trust_score": 0.75,
            "evidence": "More evidence",
            "supports": False,
        },
    ]
