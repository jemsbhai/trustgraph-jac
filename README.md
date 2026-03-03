# trustgraph-jac

**TrustGraph** Jac/Jaseci OSP walker for agentic knowledge verification.

Built on top of [trustandverify](https://github.com/jemsbhai/trustandverify) — the Python package that implements all confidence algebra, search backends, storage, and export. This package provides the Jac/Jaseci graph layer: OSP nodes, edges, byLLM functions, and the agentic walker.

Enhanced from the Velric Miami Hackathon 2026 implementation. Fully additive — hackathon usage works unchanged.

---

## Install

```bash
pip install trustgraph-jac
```

Requires API keys:

```bash
export GEMINI_API_KEY=...
export TAVILY_API_KEY=...
```

---

## Usage

### Same as hackathon (unchanged)

```bash
jac run trustgraph.jac
jac run trustgraph.jac "Is nuclear energy safer than solar?"
jac run trustgraph.jac "Is coffee healthy?" --claims 5
```

### New in 0.2.0

```bash
# Multi-backend search (Tavily + Brave + Bing, deduped)
jac run trustgraph.jac "Is remote work productive?" --search multi

# Export all formats
jac run trustgraph.jac "Is coffee healthy?" --format all

# Persist to SQLite
jac run trustgraph.jac "Is coffee healthy?" --storage sqlite --db reports.db

# Full options
jac run trustgraph.jac "Is nuclear energy safer than solar?" \
  --claims 5 \
  --search multi \
  --storage sqlite \
  --db reports.db \
  --format all
```

---

## What's in this package

| File | Purpose |
|---|---|
| `trustgraph.jac` | Main walker — additive to hackathon, new flags for backends |
| `bridge/confidence.py` | Drop-in replacement for hackathon `bridge/confidence.py` — delegates to `trustandverify.scoring` |
| `tools/search.py` | Drop-in replacement for hackathon `tools/search.py` — delegates to `trustandverify.search` |

## What moved to trustandverify

All confidence algebra, search backends, storage, and export logic now lives in the `trustandverify` Python package. The Jac layer focuses on what Jac is good at: graph structure, byLLM functions, and the agentic walker pattern.

---

## Hackathon compatibility

The bridge modules preserve the exact import signatures from the hackathon:

```jac
import from bridge.confidence {
    scalar_to_opinion, flip_opinion, fuse_evidence,
    apply_trust_discount, detect_conflicts_within_claim,
    opinion_summary, build_jsonld_claim
}
import from tools.search { web_search }
```

These still work — they now delegate to `trustandverify` instead of having inline implementations.

---

## New node fields (0.2.0)

The `Claim` node now stores explicit opinion components for graph queries:

```
claim.opinion_belief      # float
claim.opinion_disbelief   # float  
claim.opinion_uncertainty # float
```

Previously only `claim.confidence` (projected probability) and `claim.status` (verdict string) were stored.

---

*Part of the TrustGraph project. Built at the Velric Miami Hackathon 2026.*
