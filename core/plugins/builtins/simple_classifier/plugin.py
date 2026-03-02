"""
Simple Classifier

Heuristic + enrichment-aware classifier. Optionally upgrades to OpenAI.

In heuristic mode: combines heuristic_baseline features with theme/stance
enrichment signals (if available) for improved classification.

In openai mode: uses zero-shot LLM classification with evidence.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.plugins.base import BasePlugin, PluginContext, PluginRunResult
from shared_models import PluginKind, PluginMetadata, PluginParameterSpec, PredLabel


# Import heuristic scoring from sibling plugin
def _heuristic_score(text: str) -> tuple[float, dict]:
    """Compute heuristic AI score and features. Returns (score, features)."""
    import re
    from collections import Counter

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r'\b\w+\b', text.lower())
    if not sentences or not words:
        return 0.5, {}

    sent_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    avg_sent = sum(sent_lengths) / len(sent_lengths)
    variance = (
        sum((l - avg_sent) ** 2 for l in sent_lengths) / len(sent_lengths)
        if len(sent_lengths) > 1 else 0
    )
    ngrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
    ngram_counts = Counter(ngrams)
    repeated = sum(c - 1 for c in ngram_counts.values() if c > 1)
    rep_ratio = repeated / max(len(ngrams), 1)
    punct = sum(1 for c in text if c in ",:;()") / max(len(text), 1)
    starts_with_i = text.strip().lower().startswith("i ")
    personal = any(m in text.lower() for m in [
        "i am", "i have", "my family", "my husband", "my wife",
        "my child", "i work", "i rely", "i depend",
    ])
    ai_opener = any(text.lower().startswith(o) or f" {o}" in text.lower() for o in [
        "as a healthcare", "as a patient", "i am writing to",
        "i urge you to", "i strongly", "it is essential", "it is crucial",
        "this policy", "the proposed rule",
    ])

    score = 0.5
    if avg_sent > 30: score += 0.15
    elif avg_sent > 20: score += 0.05
    if variance < 5 and len(sentences) > 2: score += 0.12
    score += rep_ratio * 0.3
    if punct > 0.04: score += 0.08
    if personal: score -= 0.2
    if starts_with_i: score -= 0.08
    if ai_opener: score += 0.15
    if len(words) > 300: score += 0.08

    features = {
        "avg_sentence_length": round(avg_sent, 2),
        "sentence_length_variance": round(variance, 2),
        "repetition_ratio": round(rep_ratio, 4),
        "starts_with_i": starts_with_i,
        "has_personal": personal,
        "has_ai_opener": ai_opener,
        "word_count": len(words),
    }
    return max(0.0, min(1.0, score)), features


def _openai_classify_batch(texts: list[str], comment_ids: list[str], model: str) -> list[dict]:
    """Zero-shot LLM classification."""
    import json
    import config
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    SYSTEM = """You are an expert at detecting AI-generated public regulatory comments.
For each comment, output:
- pred_label: "human", "ai_generated", or "uncertain"
- score_ai: float 0-1 (probability of AI-generated)
- confidence: float 0-1 (confidence in the prediction)
- reasoning: 1-sentence explanation

Reply as a JSON array in the same order as input."""

    batch_text = "\n\n".join(f"[{i+1}]: {t[:600]}" for i, t in enumerate(texts))
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": batch_text}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        items = parsed if isinstance(parsed, list) else list(parsed.values())[0]
        results = []
        for item in items[:len(texts)]:
            results.append({
                "pred_label": item.get("pred_label", "uncertain"),
                "score_ai": float(item.get("score_ai", 0.5)),
                "confidence": float(item.get("confidence", 0.5)),
                "explanation_json": {"reasoning": item.get("reasoning", ""), "model": model},
            })
        return results
    except Exception as e:
        return [{"pred_label": "uncertain", "score_ai": 0.5, "confidence": 0.0,
                 "explanation_json": {"error": str(e)}} for _ in texts]


class Plugin(BasePlugin):
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="simple_classifier",
            version="0.1.0",
            kind=PluginKind.CLASSIFIER,
            author="anti_slop_team",
            description="Heuristic + enrichment-aware classifier (no API key required in heuristic mode)",
            entrypoint="plugin.py:Plugin",
            inputs={"required": ["comment_text"], "optional": ["enrichments.theme_stance_enricher"]},
            outputs={"per_comment": ["pred_label", "score_ai", "confidence", "explanation_json"]},
            parameters={
                "provider": PluginParameterSpec(type="enum", values=["heuristic", "openai"], default="heuristic"),
                "model": PluginParameterSpec(type="string", default="gpt-4.1-mini"),
                "threshold": PluginParameterSpec(type="float", default=0.5, min=0.0, max=1.0),
                "use_enrichments": PluginParameterSpec(type="bool", default=True),
            },
        )

    def validate_inputs(self, dataset, available_artifacts, params) -> list[str]:
        errors = []
        if "comment_text" not in dataset.columns:
            errors.append("Missing required column: comment_text")
        if params.get("provider") == "openai":
            import config
            if not config.OPENAI_API_KEY:
                errors.append("OPENAI_API_KEY not set in environment")
        return errors

    def run(self, dataset, available_artifacts, params, context) -> PluginRunResult:
        provider = str(params.get("provider", "heuristic"))
        model = str(params.get("model", "gpt-4.1-mini"))
        threshold = float(params.get("threshold", 0.5))
        use_enrichments = bool(params.get("use_enrichments", True))

        # Load enrichments if available
        enrichments_map: dict[str, dict] = {}
        enrichments = available_artifacts.get("enrichments", {})
        for run_name, enrich_df in enrichments.items():
            if "theme_stance_enricher" in run_name and not enrich_df.empty:
                for _, row in enrich_df.iterrows():
                    enrichments_map[row["comment_id"]] = row.to_dict()
                break

        results = []
        error_count = 0

        if provider == "openai":
            valid = dataset[dataset["comment_text"].notna() & (dataset["comment_text"] != "")]
            batch_size = 30
            rows_list = list(valid.iterrows())
            for batch_start in range(0, len(rows_list), batch_size):
                batch = rows_list[batch_start:batch_start + batch_size]
                texts = [str(r["comment_text"]) for _, r in batch]
                cids = [r["comment_id"] for _, r in batch]
                batch_results = _openai_classify_batch(texts, cids, model)
                for cid, res in zip(cids, batch_results):
                    res["comment_id"] = cid
                    results.append(res)
        else:
            # Heuristic mode
            for _, row in dataset.iterrows():
                try:
                    text = str(row.get("comment_text", "") or "").strip()
                    cid = row["comment_id"]

                    if len(text) < 20:
                        results.append({
                            "comment_id": cid, "pred_label": "uncertain",
                            "score_ai": 0.5, "confidence": 0.2,
                            "explanation_json": {"reason": "too_short"},
                        })
                        continue

                    score, features = _heuristic_score(text)

                    # Enrichment adjustments
                    if use_enrichments and cid in enrichments_map:
                        enrich = enrichments_map[cid]
                        stance = enrich.get("stance", "neutral")
                        # Strong pro-stance on general topics → more AI
                        if stance == "pro" and enrich.get("theme_primary") in [
                            "general_support", "coverage_policy"
                        ]:
                            score = min(1.0, score + 0.05)
                        # Personal anecdote stance → more human
                        if enrich.get("theme_primary") in ["maternal_health", "mental_health", "chronic_disease"]:
                            score = max(0.0, score - 0.05)
                        features["enrichment_theme"] = enrich.get("theme_primary", "")
                        features["enrichment_stance"] = stance

                    if score >= threshold:
                        pred = "ai_generated"
                    elif score < (1 - threshold):
                        pred = "human"
                    else:
                        pred = "uncertain"

                    confidence = abs(score - 0.5) * 2
                    results.append({
                        "comment_id": cid,
                        "pred_label": pred,
                        "score_ai": round(score, 4),
                        "confidence": round(min(confidence, 1.0), 4),
                        "explanation_json": features,
                    })
                except Exception as e:
                    error_count += 1
                    results.append({
                        "comment_id": row.get("comment_id", "__err__"),
                        "pred_label": "uncertain", "score_ai": 0.5,
                        "confidence": 0.0, "explanation_json": {"error": str(e)},
                    })

        df = pd.DataFrame(results)
        return PluginRunResult(
            kind=PluginKind.CLASSIFIER,
            per_comment_df=df,
            run_metadata={"provider": provider, "model_name": model if provider == "openai" else "simple_heuristic_v0.1"},
            logs=[f"Classified {len(results)} comments"],
            error_count=error_count,
        )
