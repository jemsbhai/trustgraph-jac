"""TrustGraph — Agentic Knowledge Verification with Confidence Algebra.

Enhanced from the Velric Miami Hackathon 2026 implementation.
Delegates confidence algebra and search to the trustandverify Python package
while keeping all graph structure, byLLM functions, and walker logic in Jac.

New in 0.2.0 (additive — defaults match hackathon behaviour):
  --search  tavily|brave|bing|multi   (default: tavily)
  --llm     gemini|openai|anthropic   (default: gemini)
  --storage memory|sqlite             (default: memory)
  --db      path/to/reports.db        (default: trustgraph.db)
  --format  jsonld|markdown|html|all  (default: jsonld)
"""

import from byllm.lib { Model }
import from litellm { completion as litellm_completion }
import from tools.search { web_search, multi_search }
import from bridge.confidence {
    scalar_to_opinion,
    flip_opinion,
    fuse_evidence,
    apply_trust_discount,
    detect_conflicts,
    detect_conflicts_within_claim,
    opinion_summary,
    build_jsonld_claim
}
import json;
import re;
import from datetime { datetime, timezone }

glob llm = Model(model_name="gemini/gemini-2.5-flash");

# ──────────────────────────────────────────────
# Node Types (expanded from hackathon graph.jac)
# ──────────────────────────────────────────────

node Query {
    has text: str;
    has created_at: str = "";
}

node Claim {
    has text: str;
    has confidence: float = 0.0;
    has status: str = "unverified";
    has opinion_data: dict = {};
    # New in 0.2.0 — explicit opinion components for graph queries
    has opinion_belief: float = 0.0;
    has opinion_disbelief: float = 0.0;
    has opinion_uncertainty: float = 1.0;
}

node Source {
    has url: str = "";
    has title: str = "";
    has content: str = "";
    has trust_score: float = 0.5;
    has source_type: str = "web";
}

node Evidence {
    has text: str;
    has supports_claim: bool = True;
    has relevance: float = 0.0;
    has confidence_raw: float = 0.5;
}

node ReportNode {
    has summary: str = "";
    has overall_confidence: float = 0.0;
    has claims_data: list = [];
    has conflicts: list = [];
    has jsonld_output: dict = {};
    # New in 0.2.0
    has report_id: str = "";
    has query_text: str = "";
}

# ──────────────────────────────────────────────
# Edge Types
# ──────────────────────────────────────────────

edge Spawns {}
edge SupportsEdge {}
edge ContradictsEdge {}
edge DerivedFrom {}
edge HasEvidence {}
edge HasClaim {}
edge HasReport {}

# ──────────────────────────────────────────────
# byLLM Functions (identical to hackathon)
# ──────────────────────────────────────────────

"""Given a research question, decompose it into 3-5 specific, verifiable factual claims.
Each claim should be a concrete statement that can be checked against evidence.
Return only the claims as a list of strings."""
def decompose_query(question: str) -> list[str]
    by llm();

def decompose_query_n(question: str, num_claims: int) -> list[str] {
    """Decompose a question into exactly N verifiable claims."""
    prompt = (
        f"Given the following research question, decompose it into exactly {num_claims} "
        f"specific, verifiable factual claims. Each claim should be a concrete statement "
        f"that can be checked against evidence.\n\n"
        f"Question: {question}\n\n"
        f"Return ONLY a JSON array of {num_claims} strings. No markdown, no extra text.\n"
        f'Example: ["Claim one here", "Claim two here"]'
    );
    response = litellm_completion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}]
    );
    raw = response.choices[0].message.content;
    cleaned = raw.strip();
    if cleaned.startswith("```") {
        first_nl = cleaned.find("\n");
        if first_nl > 0 {
            cleaned = cleaned[first_nl + 1:];
        }
        if cleaned.rstrip().endswith("```") {
            cleaned = cleaned.rstrip()[:-3].rstrip();
        }
    }
    try {
        result = json.loads(cleaned);
        if isinstance(result, list) {
            return result;
        }
    } except Exception {
        _pass = 0;
    }
    try {
        start = raw.find("[");
        end = raw.rfind("]");
        if start >= 0 and end > start {
            result = json.loads(raw[start:end + 1]);
            if isinstance(result, list) {
                return result;
            }
        }
    } except Exception {
        _pass = 0;
    }
    print("      WARNING: Could not parse decompose response, falling back to byLLM.");
    return decompose_query(question);
}

"""Given a claim and text from a source, extract the key evidence.
Return ONLY a valid JSON object (no markdown, no extra text) with these fields:
- "evidence": the specific text that supports or contradicts the claim
- "supports": true if the evidence supports the claim, false if it contradicts
- "relevance": a float 0-1 indicating how directly relevant this evidence is
- "confidence": a float 0-1 indicating how confident you are in this assessment
Example: {"evidence": "Studies show X", "supports": true, "relevance": 0.8, "confidence": 0.85}"""
def extract_evidence(claim: str, source_text: str) -> str
    by llm();

"""Given a claim and all the evidence collected for and against it,
write a concise 2-3 sentence assessment summarizing the finding.
Include the confidence level and note any conflicts between sources."""
def assess_claim(claim: str, supporting_evidence: list[str], contradicting_evidence: list[str], confidence_score: float) -> str
    by llm();

"""Given a research question and a list of assessed claims with confidence scores,
write a concise executive summary (3-5 sentences) of the overall findings.
Highlight areas of agreement and disagreement among sources."""
def write_summary(question: str, assessed_claims: list[str]) -> str
    by llm();

"""Given a claim, generate a good web search query to find evidence for or against it.
Return just the search query string, optimized for finding research and data."""
def claim_to_search_query(claim: str) -> str
    by llm();

# ──────────────────────────────────────────────
# Helpers (identical to hackathon)
# ──────────────────────────────────────────────

def parse_evidence_json(raw: str) -> dict {
    """Robustly parse LLM evidence output, handling markdown fences and extra text."""
    defaults = {"evidence": raw, "supports": True, "relevance": 0.5, "confidence": 0.5};
    raw = raw.replace("\xa0", " ").replace("\u200b", "").replace("\u2009", " ").replace("\u202f", " ");
    try {
        return json.loads(raw);
    } except Exception {
        _pass = 0;
    }
    try {
        cleaned = raw.strip();
        if cleaned.startswith("```") {
            first_newline = cleaned.find("\n");
            if first_newline > 0 {
                cleaned = cleaned[first_newline + 1:];
            }
            if cleaned.rstrip().endswith("```") {
                cleaned = cleaned.rstrip()[:-3].rstrip();
            }
        }
        return json.loads(cleaned);
    } except Exception {
        _pass = 0;
    }
    try {
        start = raw.find("{");
        end = raw.rfind("}");
        if start >= 0 and end > start {
            candidate = raw[start:end + 1];
            parsed = json.loads(candidate);
            if "evidence" in parsed or "confidence" in parsed {
                return parsed;
            }
        }
    } except Exception {
        _pass = 0;
    }
    print(f"        WARNING: Could not parse evidence JSON. Raw[0:120]: {raw[:120]}");
    return defaults;
}

def estimate_source_trust(url: str, title: str) -> float {
    """Heuristic trust scoring based on source domain."""
    url_lower = url.lower();
    if ".gov" in url_lower or ".edu" in url_lower {
        return 0.9;
    }
    if "nature.com" in url_lower or "sciencedirect" in url_lower
       or "pubmed" in url_lower or "arxiv" in url_lower
       or "nber.org" in url_lower {
        return 0.85;
    }
    if "reuters" in url_lower or "bbc.com" in url_lower
       or "nytimes" in url_lower or "wsj.com" in url_lower
       or "economist" in url_lower {
        return 0.75;
    }
    if "wikipedia" in url_lower {
        return 0.6;
    }
    if "reddit.com" in url_lower or "quora.com" in url_lower {
        return 0.35;
    }
    return 0.5;
}

# New in 0.2.0 — export helpers
def save_exports(jsonld_output: dict, query_text: str, format: str, storage_backend: str, db_path: str) -> None {
    """Save report in requested formats and persist to storage backend."""
    import from trustandverify.export.jsonld { JsonLdExporter }
    import from trustandverify.export.markdown { MarkdownExporter }
    import from trustandverify.export.html { HtmlExporter }
    import from trustandverify.core.models { Report, Claim, Conflict, Verdict }
    import from datetime { datetime, timezone }
    import uuid;

    # Build a trustandverify Report from the jsonld output for export
    tv_claims = [];
    for jc in jsonld_output.get("ex:claims", []) {
        conf = jc.get("ex:confidence", {});
        import from jsonld_ex.confidence_algebra { Opinion };
        op = Opinion(
            belief=conf.get("ex:belief", 0.0),
            disbelief=conf.get("ex:disbelief", 0.0),
            uncertainty=conf.get("ex:uncertainty", 1.0),
            base_rate=conf.get("ex:baseRate", 0.5)
        );
        p = op.projected_probability();
        if p >= 0.7 {
            v = Verdict.SUPPORTED;
        } elif p > 0.3 {
            v = Verdict.CONTESTED;
        } else {
            v = Verdict.REFUTED;
        }
        tv_claims.append(Claim(text=jc.get("ex:claimText", ""), opinion=op, verdict=v));
    }
    tv_conflicts = [];
    for c in jsonld_output.get("ex:conflicts", []) {
        import from trustandverify.core.models { Conflict };
        tv_conflicts.append(Conflict(
            claim_text=c.get("claim", ""),
            conflict_degree=c.get("conflict_degree", 0.0),
            num_supporting=c.get("num_supporting", 0),
            num_contradicting=c.get("num_contradicting", 0)
        ));
    }
    report = Report(
        id=str(uuid.uuid4()),
        query=query_text,
        claims=tv_claims,
        conflicts=tv_conflicts,
        summary=jsonld_output.get("ex:summary", ""),
        created_at=datetime.now(timezone.utc)
    );

    # JSON-LD (always saved)
    with open("output.json", "w") as f {
        json.dump(jsonld_output, f, indent=2);
    }
    print("  JSON-LD saved to output.json");

    if format == "markdown" or format == "all" {
        MarkdownExporter().render_to_file(report, "output.md");
        print("  Markdown saved to output.md");
    }
    if format == "html" or format == "all" {
        HtmlExporter().render_to_file(report, "output.html");
        print("  HTML saved to output.html");
    }

    # Storage
    if storage_backend == "sqlite" {
        import from trustandverify.storage.sqlite { SQLiteStorage };
        import asyncio;
        storage = SQLiteStorage(db_path);
        asyncio.run(storage.save_report(report));
        print(f"  Report saved to SQLite: {db_path} (id={report.id})");
    }
}

# ──────────────────────────────────────────────
# Main Walker
# ──────────────────────────────────────────────

walker TrustGraphAgent {
    has query_text: str = "";
    has max_search_per_claim: int = 3;
    has num_claims: int = 0;
    has report: dict = {};
    # New in 0.2.0
    has search_backend: str = "tavily";    # tavily|brave|bing|multi
    has storage_backend: str = "memory";   # memory|sqlite
    has db_path: str = "trustgraph.db";
    has export_format: str = "jsonld";     # jsonld|markdown|html|all

    can run with entry {
        print(f"\n{'='*60}");
        print(f"  TrustGraph — Agentic Knowledge Verification");
        print(f"  trustandverify {self.search_backend} search · {self.storage_backend} storage");
        print(f"{'='*60}");
        print(f"\n  Query: {self.query_text}\n");

        # ── STEP 1: PLAN ──
        if self.num_claims > 0 {
            print(f"[1/5] PLAN — Decomposing query into {self.num_claims} claims...");
            claims_text = decompose_query_n(self.query_text, self.num_claims);
        } else {
            print("[1/5] PLAN — Decomposing query into verifiable claims...");
            claims_text = decompose_query(self.query_text);
        }
        print(f"      Found {len(claims_text)} claims to verify.\n");

        q = Query(text=self.query_text, created_at=str(datetime.now(timezone.utc)));
        here ++> q;

        claim_nodes: list = [];
        for i in range(len(claims_text)) {
            c = Claim(text=claims_text[i]);
            q +>: Spawns :+> c;
            claim_nodes.append(c);
        }

        all_opinions: list = [];
        all_conflicts: list = [];
        assessed_claims: list = [];
        all_sources_per_claim: list = [];

        for ci in range(len(claim_nodes)) {
            claim = claim_nodes[ci];
            print(f"[2/5] SEARCH — Claim {ci+1}: {claim.text[:70]}...");

            search_query = claim_to_search_query(claim.text);

            # New in 0.2.0: multi-backend search
            if self.search_backend == "multi" {
                results = multi_search(search_query, self.max_search_per_claim);
            } else {
                results = web_search(search_query, self.max_search_per_claim);
            }
            print(f"      Found {len(results)} sources.");

            print(f"[3/5] EXTRACT — Analyzing evidence...");
            evidence_opinions: list = [];
            supporting: list = [];
            contradicting: list = [];
            supporting_opinions: list = [];
            contradicting_opinions: list = [];
            claim_sources: list = [];

            for ri in range(len(results)) {
                r = results[ri];
                trust = estimate_source_trust(r["url"], r["title"]);
                src = Source(
                    url=r["url"],
                    title=r["title"],
                    content=r["content"][:500],
                    trust_score=trust,
                    source_type="web"
                );

                raw = extract_evidence(claim.text, r["content"]);
                ev_data = parse_evidence_json(raw);

                ev = Evidence(
                    text=str(ev_data.get("evidence", raw))[:300],
                    supports_claim=bool(ev_data.get("supports", True)),
                    relevance=float(ev_data.get("relevance", 0.5)),
                    confidence_raw=float(ev_data.get("confidence", 0.5))
                );

                ev +>: DerivedFrom :+> src;
                claim +>: HasEvidence :+> ev;

                if ev.supports_claim {
                    supporting.append(ev.text);
                } else {
                    contradicting.append(ev.text);
                }

                claim_sources.append({
                    "title": r["title"],
                    "url": r["url"],
                    "trust_score": trust,
                    "evidence": ev.text[:150],
                    "supports": ev.supports_claim
                });

                raw_opinion = scalar_to_opinion(ev.confidence_raw);
                discounted = apply_trust_discount(raw_opinion, trust);

                if ev.supports_claim {
                    evidence_opinions.append(discounted);
                    supporting_opinions.append(discounted);
                } else {
                    flipped = flip_opinion(discounted);
                    evidence_opinions.append(flipped);
                    contradicting_opinions.append(flipped);
                }

                sup_label = "supports" if ev.supports_claim else "CONTRADICTS";
                print(f"        [{r['title'][:40]}] trust={trust} conf={ev.confidence_raw:.2f} {sup_label}");
            }

            print(f"[4/5] SCORE — Fusing evidence with Subjective Logic...");
            if len(evidence_opinions) > 0 {
                fused = fuse_evidence(evidence_opinions);
                summary = opinion_summary(fused);
                claim.confidence = summary["projected_probability"];
                claim.status = summary["verdict"];
                claim.opinion_data = summary;
                # New in 0.2.0 — store components on node for graph queries
                claim.opinion_belief = summary["belief"];
                claim.opinion_disbelief = summary["disbelief"];
                claim.opinion_uncertainty = summary["uncertainty"];
                all_opinions.append(fused);

                print(f"      Result: {summary['verdict']} (P={summary['projected_probability']})");
                print(f"      Opinion: b={summary['belief']} d={summary['disbelief']} u={summary['uncertainty']}");
            } else {
                claim.status = "no_evidence";
                print(f"      No evidence found.");
            }

            claim_conflict = detect_conflicts_within_claim(
                supporting_opinions, contradicting_opinions, 0.2
            );
            if claim_conflict is not None {
                all_conflicts.append({
                    "claim": claim.text[:80],
                    "conflict_degree": claim_conflict["conflict_degree"],
                    "num_supporting": claim_conflict["num_supporting"],
                    "num_contradicting": claim_conflict["num_contradicting"]
                });
                print(f"      CONFLICT: {claim_conflict['num_supporting']} support, {claim_conflict['num_contradicting']} contradict (degree={claim_conflict['conflict_degree']})");
            }

            all_sources_per_claim.append(claim_sources);

            assessment = assess_claim(claim.text, supporting, contradicting, claim.confidence);
            assessed_claims.append(
                f"[{claim.status.upper()} P={claim.confidence:.2f}] {claim.text}\n  Assessment: {assessment}"
            );
            print(f"      Assessment: {assessment[:100]}...\n");
        }

        print(f"[5/5] REPORT — Generating summary...\n");
        summary_text = write_summary(self.query_text, assessed_claims);

        jsonld_claims: list = [];
        for ci in range(len(claim_nodes)) {
            c = claim_nodes[ci];
            if len(all_opinions) > ci {
                sources = [];
                if len(all_sources_per_claim) > ci {
                    sources = all_sources_per_claim[ci];
                }
                jc = build_jsonld_claim(c.text, all_opinions[ci], sources);
                jsonld_claims.append(jc);
            }
        }

        jsonld_output = {
            "@context": {
                "@vocab": "https://schema.org/",
                "ex": "https://jsonld-ex.org/vocab#",
                "prov": "http://www.w3.org/ns/prov#"
            },
            "@type": "ex:TrustGraphReport",
            "ex:query": self.query_text,
            "ex:generatedAt": str(datetime.now(timezone.utc)),
            "ex:claims": jsonld_claims,
            "ex:conflicts": all_conflicts,
            "ex:summary": summary_text
        };

        print(f"{'='*60}");
        print(f"  TRUSTGRAPH VERIFICATION REPORT");
        print(f"{'='*60}");
        print(f"\n  Query: {self.query_text}\n");
        print("  CLAIMS:");
        for ci in range(len(assessed_claims)) {
            print(f"    {ci+1}. {assessed_claims[ci]}\n");
        }
        if len(all_conflicts) > 0 {
            print("  CONFLICTS DETECTED:");
            for conf in all_conflicts {
                print(f"    - {conf['claim']}");
                print(f"      {conf['num_supporting']} support vs {conf['num_contradicting']} contradict (degree={conf['conflict_degree']})");
            }
            print("");
        }
        print(f"  SUMMARY:\n    {summary_text}\n");
        print(f"{'='*60}");

        self.report = jsonld_output;

        # Save all requested formats + storage
        save_exports(
            jsonld_output,
            self.query_text,
            self.export_format,
            self.storage_backend,
            self.db_path
        );
    }
}

# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

with entry {
    import sys;
    import os;

    query = "Is remote work more productive than office work?";
    num_claims = 0;
    search_backend = "tavily";
    storage_backend = "memory";
    db_path = "trustgraph.db";
    export_format = "jsonld";

    # File-based config (UI compatibility — identical to hackathon)
    if os.path.exists("_query.txt") {
        with open("_query.txt", "r") as f {
            file_query = f.read().strip();
        }
        if len(file_query) > 0 {
            query = file_query;
        }
    }
    if os.path.exists("_config.json") {
        try {
            with open("_config.json", "r") as f {
                config = json.loads(f.read());
            }
            if "num_claims" in config {
                num_claims = int(config["num_claims"]);
            }
            if "search_backend" in config {
                search_backend = config["search_backend"];
            }
            if "export_format" in config {
                export_format = config["export_format"];
            }
        } except Exception {
            _pass = 0;
        }
    }

    # CLI args
    args = sys.argv[1:];
    flag_map = {
        "--claims":  "num_claims",
        "--search":  "search_backend",
        "--storage": "storage_backend",
        "--db":      "db_path",
        "--format":  "export_format",
    };
    i = 0;
    remaining_args = [];
    while i < len(args) {
        if args[i] in flag_map {
            if i + 1 < len(args) {
                flag_name = flag_map[args[i]];
                val = args[i + 1];
                if flag_name == "num_claims" {
                    num_claims = int(val);
                } elif flag_name == "search_backend" {
                    search_backend = val;
                } elif flag_name == "storage_backend" {
                    storage_backend = val;
                } elif flag_name == "db_path" {
                    db_path = val;
                } elif flag_name == "export_format" {
                    export_format = val;
                }
                i += 2;
            } else {
                i += 1;
            }
        } else {
            remaining_args.append(args[i]);
            i += 1;
        }
    }
    if len(remaining_args) > 0 {
        query = " ".join(remaining_args);
    }

    agent = TrustGraphAgent(
        query_text=query,
        num_claims=num_claims,
        search_backend=search_backend,
        storage_backend=storage_backend,
        db_path=db_path,
        export_format=export_format
    );
    root spawn agent;
}
