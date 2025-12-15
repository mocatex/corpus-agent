"""
Mocked implementations of NLP tool outputs for testing and demonstration purposes.
These functions fabricate plausible-looking outputs without performing any real NLP.
"""

from __future__ import annotations
import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple
def mock_nlp_tool_outputs(retrieval_result: dict, nlp_plan: dict) -> dict:
    """
    Very lightweight mocked execution of a subset of NLP tools.

    This function does NOT run any real NLP. It only fabricates plausible-looking
    outputs to demonstrate what the downstream analytics layer might see.
    """
    documents = retrieval_result.get("documents", [])
    chosen_tools = nlp_plan.get("chosen_tools", [])

    # Group documents by year (assuming Postgres returns 'year' in each row)
    docs_by_year = {}
    for doc in documents:
        year = doc.get("year", "unknown")
        docs_by_year.setdefault(year, []).append(doc)

    outputs = {}

    def _safe_year_key(y: Any) -> Tuple[int, str]:
        """Sort years deterministically; unknowns go last."""
        try:
            yi = int(y)
            return (0, str(yi))
        except Exception:
            return (1, str(y))

    def _hash01(s: str) -> float:
        """Deterministic pseudo-random float in [0,1)."""
        h = hashlib.md5(s.encode("utf-8"), usedforsecurity=False).hexdigest()
        return (int(h[:8], 16) % 10_000_000) / 10_000_000.0

    def _doc_len(doc: Dict[str, Any]) -> int:
        return len((doc.get("body") or ""))

    def _doc_sentiment(doc: Dict[str, Any]) -> float:
        # Same deterministic mapping used elsewhere
        return (_doc_len(doc) % 800) / 400.0 - 1.0

    def _doc_topic(doc: Dict[str, Any]) -> str:
        labels = ["Economy / Markets", "Technology / Innovation", "Politics / Regulation"]
        return labels[_doc_len(doc) % len(labels)]

    def _doc_emotion(doc: Dict[str, Any]) -> str:
        labels = ["neutral", "joy", "sadness", "anger", "fear", "surprise"]
        return labels[_doc_len(doc) % len(labels)]

    def _doc_source(doc: Dict[str, Any]) -> str:
        # Prefer a stored field if present
        return (
            doc.get("source_domain")
            or doc.get("source")
            or doc.get("domain")
            or "unknown"
        )
    # --- Mock sentiment_over_time ---
    if "sentiment_over_time" in chosen_tools:
        series = []
        for year, year_docs in sorted(docs_by_year.items(), key=lambda x: _safe_year_key(x[0])):
            # Fake "sentiment" as a deterministic function of average body length
            if year_docs:
                lengths = [len((d.get("body") or "")) for d in year_docs]
                avg_len = sum(lengths) / max(len(lengths), 1)
                avg_sentiment = (avg_len % 800) / 400.0 - 1.0  # maps roughly to [-1, +1]
            else:
                avg_sentiment = 0.0
            series.append(
                {
                    "year": year,
                    "avg_sentiment": round(avg_sentiment, 3),
                    "n_docs": len(year_docs),
                }
            )
        outputs["sentiment_over_time"] = {"series": series}

    # --- Mock topic_trend_over_time ---
    if "topic_trend_over_time" in chosen_tools:
        topics = []
        for idx, label in enumerate(["Economy / Markets", "Technology / Innovation", "Politics / Regulation"], start=1):
            timeline = []
            for year, year_docs in sorted(docs_by_year.items(), key=lambda x: _safe_year_key(x[0])):
                weight = (len(year_docs) + idx) % 10 / 10.0
                timeline.append({"year": year, "weight": round(weight, 2)})
            topics.append(
                {
                    "topic_id": idx,
                    "label": label,
                    "timeline": timeline,
                }
            )
        outputs["topic_trend_over_time"] = {"topics": topics}

    # --- Mock event_detection ---
    if "event_detection" in chosen_tools:
        events = []
        for year, year_docs in sorted(docs_by_year.items(), key=lambda x: _safe_year_key(x[0])):
            if not year_docs:
                continue
            # Just take the first document title as a "major event" placeholder
            first_doc = year_docs[0]
            events.append(
                {
                    "year": year,
                    "title": first_doc.get("title", "Unknown event"),
                    "approx_date": first_doc.get("date", f"{year}-01-01"),
                    "evidence_doc_ids": [first_doc.get("id")],
                }
            )
        outputs["event_detection"] = {"events": events}

    # --- Mock salient_quote_extraction ---
    if "salient_quote_extraction" in chosen_tools:
        quotes = []
        for doc in documents[:10]:
            body = (doc.get("body") or "").replace("\n", " ")
            snippet = body[:200]
            if not snippet:
                continue
            quotes.append(
                {
                    "doc_id": doc.get("id"),
                    "year": doc.get("year"),
                    "quote": snippet + ("..." if len(body) > 200 else ""),
                }
            )
        outputs["salient_quote_extraction"] = {"quotes": quotes}

    # --- Mock volatility_of_sentiment ---
    if "volatility_of_sentiment" in chosen_tools and "sentiment_over_time" in outputs:
        series = outputs["sentiment_over_time"]["series"]
        deltas = []
        for i in range(1, len(series)):
            prev = series[i - 1]["avg_sentiment"]
            curr = series[i]["avg_sentiment"]
            deltas.append(abs(curr - prev))
        volatility = sum(deltas) / max(len(deltas), 1) if deltas else 0.0
        outputs["volatility_of_sentiment"] = {"avg_delta": round(volatility, 3)}

    # --- Mock emotion_distribution_over_time ---
    if "emotion_distribution_over_time" in chosen_tools:
        dist = []
        for year, year_docs in sorted(docs_by_year.items(), key=lambda x: _safe_year_key(x[0])):
            counts = defaultdict(int)
            for d in year_docs:
                counts[_doc_emotion(d)] += 1
            total = max(sum(counts.values()), 1)
            dist.append(
                {
                    "year": year,
                    "distribution": {k: round(v / total, 3) for k, v in sorted(counts.items())},
                    "n_docs": len(year_docs),
                }
            )
        outputs["emotion_distribution_over_time"] = {"series": dist}

    # --- Mock dynamic_topic_shift_detection ---
    if "dynamic_topic_shift_detection" in chosen_tools:
        # Identify years where average topic "mix" changes abruptly (mocked via hashed signal)
        shifts = []
        years_sorted = [y for y, _ in sorted(docs_by_year.items(), key=lambda x: _safe_year_key(x[0]))]
        for i in range(1, len(years_sorted)):
            y_prev, y_curr = years_sorted[i - 1], years_sorted[i]
            score = abs(_hash01(f"topicmix:{y_curr}") - _hash01(f"topicmix:{y_prev}"))
            if score > 0.35:
                shifts.append(
                    {
                        "from_year": y_prev,
                        "to_year": y_curr,
                        "shift_score": round(score, 3),
                        "hypothesis": "Topic emphasis appears to change between these years.",
                    }
                )
        outputs["dynamic_topic_shift_detection"] = {"shifts": shifts}

    # --- Mock framing_shift_detection ---
    if "framing_shift_detection" in chosen_tools:
        frames = ["risk/crisis", "innovation/growth", "ethics/regulation"]
        series = []
        for year, year_docs in sorted(docs_by_year.items(), key=lambda x: _safe_year_key(x[0])):
            # Deterministic weights by year
            w = [_hash01(f"frame:{year}:{f}") for f in frames]
            s = sum(w) or 1.0
            series.append({"year": year, "frames": {f: round(v / s, 3) for f, v in zip(frames, w)}, "n_docs": len(year_docs)})
        outputs["framing_shift_detection"] = {"series": series, "frames": frames}

    # --- Mock burst_detection ---
    if "burst_detection" in chosen_tools:
        bursts = []
        for year, year_docs in sorted(docs_by_year.items(), key=lambda x: _safe_year_key(x[0])):
            # A simple burst score: document count normalized
            n = len(year_docs)
            burst_score = round(min(1.0, n / 50.0), 3)
            if burst_score >= 0.6:
                # sample a few evidence docs
                evidence = [d.get("id") for d in year_docs[:5] if d.get("id") is not None]
                bursts.append({"year": year, "burst_score": burst_score, "evidence_doc_ids": evidence})
        outputs["burst_detection"] = {"bursts": bursts}

    # --- Mock named_entity_trend_tracking ---
    if "named_entity_trend_tracking" in chosen_tools:
        # Use title tokens as pseudo-entities (deterministic, no NLP)
        entity_counts_by_year: Dict[str, Dict[str, int]] = {}
        for year, year_docs in docs_by_year.items():
            counts = defaultdict(int)
            for d in year_docs:
                title = (d.get("title") or "").strip()
                # take up to 3 "entities" as first capitalized-like tokens
                toks = [t for t in title.replace("'", "").split() if t[:1].isupper()]
                for t in toks[:3]:
                    counts[t] += 1
            entity_counts_by_year[str(year)] = dict(counts)

        # Build a compact timeline for top entities overall
        overall = defaultdict(int)
        for yc in entity_counts_by_year.values():
            for ent, c in yc.items():
                overall[ent] += c
        top_ents = [e for e, _ in sorted(overall.items(), key=lambda kv: kv[1], reverse=True)[:10]]

        timelines = []
        for ent in top_ents:
            timeline = []
            for year, _docs in sorted(docs_by_year.items(), key=lambda x: _safe_year_key(x[0])):
                timeline.append({"year": year, "count": int(entity_counts_by_year.get(str(year), {}).get(ent, 0))})
            timelines.append({"entity": ent, "timeline": timeline})

        outputs["named_entity_trend_tracking"] = {"top_entities": top_ents, "timelines": timelines}

    # --- Mock relationship_graph_extraction ---
    if "relationship_graph_extraction" in chosen_tools:
        # Construct a tiny co-occurrence graph from pseudo-entities in titles
        edges = defaultdict(int)
        nodes = set()
        for d in documents[:500]:
            title = (d.get("title") or "")
            toks = [t for t in title.replace("'", "").split() if t[:1].isupper()]
            ents = toks[:4]
            for e in ents:
                nodes.add(e)
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    a, b = sorted((ents[i], ents[j]))
                    edges[(a, b)] += 1

        edge_list = [
            {"source": a, "target": b, "weight": w}
            for (a, b), w in sorted(edges.items(), key=lambda kv: kv[1], reverse=True)[:30]
        ]
        outputs["relationship_graph_extraction"] = {"nodes": sorted(nodes)[:50], "edges": edge_list}

    # --- Mock stance_detection_over_time ---
    if "stance_detection_over_time" in chosen_tools:
        stances = ["support", "oppose", "neutral"]
        series = []
        for year, year_docs in sorted(docs_by_year.items(), key=lambda x: _safe_year_key(x[0])):
            # Derive stance mix from hashed year + question proxy (not available here) -> year only
            w = [_hash01(f"stance:{year}:{s}") for s in stances]
            ssum = sum(w) or 1.0
            series.append({"year": year, "distribution": {s: round(v / ssum, 3) for s, v in zip(stances, w)}, "n_docs": len(year_docs)})
        outputs["stance_detection_over_time"] = {"series": series}

    # --- Mock fact_density_estimation ---
    if "fact_density_estimation" in chosen_tools:
        # Estimate "fact density" from punctuation / number presence
        samples = []
        for d in documents[:200]:
            body = (d.get("body") or "")
            if not body:
                continue
            numish = sum(ch.isdigit() for ch in body)
            punct = sum(ch in ",.;:!?" for ch in body)
            density = (numish + punct) / max(len(body), 1)
            samples.append(density)
        avg = sum(samples) / max(len(samples), 1) if samples else 0.0
        outputs["fact_density_estimation"] = {"avg_fact_density": round(avg, 4), "n_sampled_docs": len(samples)}

    # --- Mock source_bias_comparison ---
    if "source_bias_comparison" in chosen_tools:
        # Compare average sentiment by source domain (mocked)
        by_src: Dict[str, List[float]] = defaultdict(list)
        for d in documents:
            by_src[_doc_source(d)].append(_doc_sentiment(d))
        rows = []
        for src, vals in sorted(by_src.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]:
            rows.append({"source": src, "avg_sentiment": round(sum(vals) / max(len(vals), 1), 3), "n_docs": len(vals)})
        outputs["source_bias_comparison"] = {"by_source": rows}

    # --- Mock temporal_segmentation ---
    if "temporal_segmentation" in chosen_tools:
        # Split into coarse phases based on years present
        years_sorted = [y for y, _ in sorted(docs_by_year.items(), key=lambda x: _safe_year_key(x[0]))]
        segments = []
        if years_sorted:
            mid = len(years_sorted) // 2
            segments.append({"label": "phase_1", "years": years_sorted[:mid]})
            segments.append({"label": "phase_2", "years": years_sorted[mid:]})
        outputs["temporal_segmentation"] = {"segments": segments}

    # --- Mock document_clustering ---
    if "document_clustering" in chosen_tools:
        # Deterministic cluster assignment by hash of id
        clusters: Dict[str, List[int]] = defaultdict(list)
        for d in documents:
            doc_id = d.get("id")
            if doc_id is None:
                continue
            c = int(_hash01(f"cluster:{doc_id}") * 3)  # 0..2
            clusters[f"cluster_{c}"] .append(int(doc_id))
        outputs["document_clustering"] = {
            "clusters": [{"cluster": k, "doc_ids": v[:50], "size": len(v)} for k, v in sorted(clusters.items())]
        }

    # --- Mock keyphrase_evolution_tracking ---
    if "keyphrase_evolution_tracking" in chosen_tools:
        phrases = ["privacy", "regulation", "innovation", "competition", "security"]
        series = []
        for year, _year_docs in sorted(docs_by_year.items(), key=lambda x: _safe_year_key(x[0])):
            weights = [_hash01(f"kp:{year}:{p}") for p in phrases]
            s = sum(weights) or 1.0
            series.append({"year": year, "keyphrases": {p: round(w / s, 3) for p, w in zip(phrases, weights)}})
        outputs["keyphrase_evolution_tracking"] = {"series": series, "keyphrases": phrases}

    # --- Mock anomaly_detection_in_discourse ---
    if "anomaly_detection_in_discourse" in chosen_tools:
        anomalies = []
        for year, year_docs in sorted(docs_by_year.items(), key=lambda x: _safe_year_key(x[0])):
            # Use deviation in avg sentiment as anomaly score
            if not year_docs:
                continue
            vals = [_doc_sentiment(d) for d in year_docs]
            avg = sum(vals) / max(len(vals), 1)
            score = abs(avg)  # far from neutral
            if score > 0.75:
                anomalies.append({"year": year, "anomaly_score": round(score, 3), "n_docs": len(year_docs)})
        outputs["anomaly_detection_in_discourse"] = {"anomalies": anomalies}

    # --- Mock confidence_scoring_of_results ---
    if "confidence_scoring_of_results" in chosen_tools:
        # Confidence based on coverage: more docs + more years -> higher confidence
        n_docs = len(documents)
        n_years = len(docs_by_year)
        conf = min(1.0, (n_docs / 80.0) * 0.6 + (n_years / 6.0) * 0.4)
        outputs["confidence_scoring_of_results"] = {
            "confidence": round(conf, 3),
            "signals": {"n_docs": n_docs, "n_years": n_years},
        }

    return outputs