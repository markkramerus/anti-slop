"""
Theme and Stance Enricher

Extracts thematic categories and pro/con/neutral stance from comments.

Local mode (default): keyword-based taxonomy for healthcare/CMS domain.
OpenAI mode: structured LLM extraction using function calling.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.plugins.base import BasePlugin, PluginContext, PluginRunResult
from shared_models import PluginKind, PluginMetadata, PluginParameterSpec


# ── Local keyword taxonomy (healthcare/CMS domain) ────────────────────────────

THEME_KEYWORDS: dict[str, list[str]] = {
    "coverage_policy": [
        "coverage", "covered", "benefit", "insurance", "medicaid", "medicare",
        "prior authorization", "step therapy", "formulary", "drug coverage",
    ],
    "cost_affordability": [
        "cost", "afford", "expensive", "price", "premium", "copay", "deductible",
        "out-of-pocket", "financial", "burden", "low income", "subsidy",
    ],
    "access_providers": [
        "access", "provider", "doctor", "physician", "specialist", "hospital",
        "network", "rural", "shortage", "availability",
    ],
    "mental_health": [
        "mental health", "behavioral health", "substance", "addiction",
        "depression", "anxiety", "therapy", "counseling", "psychiatric",
    ],
    "maternal_health": [
        "maternal", "pregnancy", "prenatal", "postpartum", "birth", "mother",
        "infant", "newborn", "obstetric",
    ],
    "chronic_disease": [
        "diabetes", "heart disease", "cancer", "chronic", "hypertension",
        "asthma", "copd", "kidney disease", "autoimmune",
    ],
    "administrative_burden": [
        "paperwork", "administrative", "red tape", "bureaucratic", "prior auth",
        "appeals", "denial", "claim", "billing", "documentation",
    ],
    "social_determinants": [
        "housing", "food insecurity", "transportation", "social determinants",
        "sdoh", "poverty", "community", "social support",
    ],
    "general_support": [
        "i support", "i agree", "this is good", "thank you", "great proposal",
        "please pass", "please approve",
    ],
    "general_opposition": [
        "i oppose", "i disagree", "this is bad", "harmful", "reject",
        "do not pass", "against this rule",
    ],
}

PRO_KEYWORDS = [
    "support", "agree", "approve", "favor", "endorse", "beneficial",
    "important", "necessary", "essential", "crucial", "good", "positive",
    "help", "improve", "protect", "expand", "ensure",
]

CON_KEYWORDS = [
    "oppose", "disagree", "against", "reject", "harmful", "dangerous",
    "bad", "problem", "concern", "worry", "hurt", "damage", "cut",
    "reduce", "limit", "restrict", "deny",
]


def _classify_local(text: str) -> dict:
    """Keyword-based theme and stance extraction."""
    text_lower = text.lower()

    # Theme matching
    theme_scores: dict[str, int] = {}
    for theme, keywords in THEME_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            theme_scores[theme] = score

    sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
    theme_labels = [t for t, _ in sorted_themes[:3]]
    theme_primary = theme_labels[0] if theme_labels else "general"

    # Stance scoring
    pro_score = sum(1 for kw in PRO_KEYWORDS if kw in text_lower)
    con_score = sum(1 for kw in CON_KEYWORDS if kw in text_lower)
    total = pro_score + con_score

    if total == 0:
        stance = "neutral"
        stance_conf = 0.5
    elif pro_score > con_score:
        stance = "pro"
        stance_conf = min(0.95, 0.5 + 0.1 * (pro_score - con_score))
    elif con_score > pro_score:
        stance = "con"
        stance_conf = min(0.95, 0.5 + 0.1 * (con_score - pro_score))
    else:
        stance = "mixed"
        stance_conf = 0.4

    return {
        "theme_primary": theme_primary,
        "theme_labels": theme_labels,
        "stance": stance,
        "stance_confidence": round(stance_conf, 3),
        "explanation_json": {
            "theme_scores": theme_scores,
            "pro_score": pro_score,
            "con_score": con_score,
            "method": "keyword",
        },
    }


def _classify_openai_batch(texts: list[str], comment_ids: list[str], model: str) -> list[dict]:
    """OpenAI batch classification using structured output."""
    import config
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    SYSTEM_PROMPT = """You are an expert analyst classifying public regulatory comments.
For each comment, extract:
1. theme_primary: the single most relevant theme (e.g., coverage_policy, cost_affordability, access_providers, mental_health, maternal_health, chronic_disease, administrative_burden, general_support, general_opposition, other)
2. theme_labels: up to 3 relevant themes as a list
3. stance: "pro" (supports the proposed rule), "con" (opposes), "neutral", or "mixed"
4. stance_confidence: float 0-1

Respond with a JSON array, one object per comment, in the same order."""

    batch_text = "\n\n".join(
        f"[Comment {i+1}]: {text[:500]}" for i, text in enumerate(texts)
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": batch_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        # Handle both {"results": [...]} and [...]
        if isinstance(parsed, dict):
            items = parsed.get("results", parsed.get("comments", list(parsed.values())[0]))
        else:
            items = parsed

        results = []
        for i, item in enumerate(items[:len(texts)]):
            results.append({
                "theme_primary": item.get("theme_primary", "general"),
                "theme_labels": item.get("theme_labels", []),
                "stance": item.get("stance", "neutral"),
                "stance_confidence": float(item.get("stance_confidence", 0.5)),
                "explanation_json": {"method": "openai", "model": model},
            })
        return results
    except Exception as e:
        return [
            {
                "theme_primary": "general",
                "theme_labels": [],
                "stance": "neutral",
                "stance_confidence": 0.5,
                "explanation_json": {"error": str(e), "method": "openai"},
            }
            for _ in texts
        ]


class Plugin(BasePlugin):
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="theme_stance_enricher",
            version="0.1.0",
            kind=PluginKind.ENRICHMENT,
            author="anti_slop_team",
            description="Extracts themes and stance (local keyword or OpenAI)",
            entrypoint="plugin.py:Plugin",
            inputs={"required": ["comment_text"], "optional": ["metadata"]},
            outputs={"per_comment": ["theme_primary", "theme_labels", "stance", "stance_confidence", "explanation_json"]},
            parameters={
                "provider": PluginParameterSpec(type="enum", values=["local", "openai"], default="local"),
                "model": PluginParameterSpec(type="string", default="gpt-4.1-mini"),
                "batch_size": PluginParameterSpec(type="int", default=50),
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
        provider = params.get("provider", "local")
        model = params.get("model", "gpt-4.1-mini")
        batch_size = int(params.get("batch_size", 50))

        results = []
        error_count = 0

        valid = dataset[dataset["comment_text"].notna() & (dataset["comment_text"] != "")]

        if provider == "local":
            for _, row in valid.iterrows():
                try:
                    result = _classify_local(str(row["comment_text"]))
                    result["comment_id"] = row["comment_id"]
                    results.append(result)
                except Exception as e:
                    error_count += 1
                    results.append({
                        "comment_id": row["comment_id"],
                        "theme_primary": "general",
                        "theme_labels": [],
                        "stance": "neutral",
                        "stance_confidence": 0.5,
                        "explanation_json": {"error": str(e)},
                    })
        else:
            # OpenAI batched
            rows_list = list(valid.iterrows())
            for batch_start in range(0, len(rows_list), batch_size):
                batch = rows_list[batch_start:batch_start + batch_size]
                texts = [str(r["comment_text"]) for _, r in batch]
                cids = [r["comment_id"] for _, r in batch]
                batch_results = _classify_openai_batch(texts, cids, model)
                for cid, res in zip(cids, batch_results):
                    res["comment_id"] = cid
                    results.append(res)
                context.log(f"  Processed {min(batch_start + batch_size, len(rows_list))}/{len(rows_list)}")

        # Handle rows with missing text
        missing_ids = set(dataset["comment_id"]) - {r["comment_id"] for r in results}
        for cid in missing_ids:
            results.append({
                "comment_id": cid,
                "theme_primary": "general",
                "theme_labels": [],
                "stance": "neutral",
                "stance_confidence": 0.5,
                "explanation_json": {"reason": "empty_text"},
            })

        df = pd.DataFrame(results)
        return PluginRunResult(
            kind=PluginKind.ENRICHMENT,
            per_comment_df=df,
            run_metadata={"provider": provider, "model_name": model if provider == "openai" else "keyword_v0.1"},
            logs=[f"Enriched {len(results)} comments ({error_count} errors)"],
            error_count=error_count,
        )
