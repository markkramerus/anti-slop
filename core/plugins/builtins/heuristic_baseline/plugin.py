"""
Heuristic Baseline Classifier

No API key required. Uses lexical/structural heuristics to score
each comment's likelihood of being AI-generated.

Features used:
  - avg_sentence_length: longer sentences → more AI-like
  - sentence_length_variance: low variance → more AI-like (uniform structure)
  - repetition_ratio: high ratio of repeated phrases → AI template signal
  - punct_density: high punctuation density → AI formatting signal
  - starts_with_i: first person singular opener → human signal
  - has_personal_anecdote: mentions personal experience → human signal
  - avg_word_length: longer words → slightly more AI-like
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


class Plugin(BasePlugin):
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="heuristic_baseline",
            version="0.1.0",
            kind=PluginKind.CLASSIFIER,
            author="anti_slop_team",
            description="Lexical/structural heuristic AI detector (no API key required)",
            entrypoint="plugin.py:Plugin",
            inputs={"required": ["comment_text"]},
            outputs={"per_comment": ["pred_label", "score_ai", "confidence", "explanation_json"]},
            parameters={
                "threshold": PluginParameterSpec(type="float", default=0.5, min=0.0, max=1.0),
                "min_length": PluginParameterSpec(type="int", default=50),
            },
        )

    def validate_inputs(self, dataset, available_artifacts, params) -> list[str]:
        errors = []
        if "comment_text" not in dataset.columns:
            errors.append("Missing required column: comment_text")
        return errors

    def run(
        self,
        dataset: pd.DataFrame,
        available_artifacts: dict[str, Any],
        params: dict[str, Any],
        context: PluginContext,
    ) -> PluginRunResult:
        threshold = float(params.get("threshold", 0.5))
        min_length = int(params.get("min_length", 50))

        results = []
        error_count = 0

        for _, row in dataset.iterrows():
            try:
                comment_id = row["comment_id"]
                text = str(row.get("comment_text", "") or "").strip()

                if len(text) < min_length:
                    results.append({
                        "comment_id": comment_id,
                        "pred_label": PredLabel.UNCERTAIN.value,
                        "score_ai": 0.5,
                        "confidence": 0.3,
                        "explanation_json": {"reason": "too_short"},
                    })
                    continue

                features = _extract_features(text)
                score = _score(features)

                if score >= threshold:
                    pred = PredLabel.AI_GENERATED.value
                elif score < (1 - threshold):
                    pred = PredLabel.HUMAN.value
                else:
                    pred = PredLabel.UNCERTAIN.value

                confidence = abs(score - 0.5) * 2  # 0 at boundary, 1 at extremes

                results.append({
                    "comment_id": comment_id,
                    "pred_label": pred,
                    "score_ai": round(score, 4),
                    "confidence": round(min(confidence, 1.0), 4),
                    "explanation_json": features,
                })
            except Exception as e:
                error_count += 1
                results.append({
                    "comment_id": row.get("comment_id", "__error__"),
                    "pred_label": PredLabel.UNCERTAIN.value,
                    "score_ai": 0.5,
                    "confidence": 0.0,
                    "explanation_json": {"error": str(e)},
                })

        df = pd.DataFrame(results)
        return PluginRunResult(
            kind=PluginKind.CLASSIFIER,
            per_comment_df=df,
            run_metadata={"provider": "local", "model_name": "heuristic_v0.1"},
            logs=[f"Processed {len(results)} comments"],
            error_count=error_count,
        )


def _extract_features(text: str) -> dict:
    """Extract lexical/structural features from comment text."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r'\b\w+\b', text.lower())

    if not sentences:
        return {}
    if not words:
        return {}

    # Sentence length features
    sent_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    avg_sent_len = sum(sent_lengths) / len(sent_lengths)
    variance = (
        sum((l - avg_sent_len) ** 2 for l in sent_lengths) / len(sent_lengths)
        if len(sent_lengths) > 1 else 0
    )

    # Word features
    avg_word_len = sum(len(w) for w in words) / len(words)

    # Repetition: check for repeated 3-word phrases
    ngrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
    ngram_counts = Counter(ngrams)
    repeated = sum(c - 1 for c in ngram_counts.values() if c > 1)
    repetition_ratio = repeated / max(len(ngrams), 1)

    # Punctuation density
    punct_count = sum(1 for c in text if c in ",:;()")
    punct_density = punct_count / max(len(text), 1)

    # Personal signals (human markers)
    starts_with_i = text.strip().lower().startswith("i ")
    personal_markers = ["i am", "i have", "my family", "my husband", "my wife",
                        "my child", "i work", "as a", "i rely", "i depend"]
    has_personal = any(m in text.lower() for m in personal_markers)

    # AI style markers
    ai_openers = [
        "as a healthcare", "as a patient", "as someone who",
        "i am writing to", "i urge you to", "i strongly",
        "it is essential", "it is crucial", "this policy",
        "the proposed rule", "i support", "i oppose",
    ]
    has_ai_opener = any(text.lower().startswith(o) or f" {o}" in text.lower() for o in ai_openers)

    return {
        "avg_sentence_length": round(avg_sent_len, 2),
        "sentence_length_variance": round(variance, 2),
        "avg_word_length": round(avg_word_len, 2),
        "repetition_ratio": round(repetition_ratio, 4),
        "punct_density": round(punct_density, 4),
        "starts_with_i": starts_with_i,
        "has_personal_anecdote": has_personal,
        "has_ai_opener": has_ai_opener,
        "word_count": len(words),
        "sentence_count": len(sentences),
    }


def _score(features: dict) -> float:
    """
    Combine features into a single AI likelihood score [0, 1].
    Higher = more likely AI-generated.
    """
    if not features:
        return 0.5

    score = 0.5  # Start neutral

    # Long sentences → AI
    avg_sent = features.get("avg_sentence_length", 15)
    if avg_sent > 30:
        score += 0.15
    elif avg_sent > 20:
        score += 0.05
    elif avg_sent < 10:
        score -= 0.05

    # Low sentence length variance → AI (uniform structure)
    variance = features.get("sentence_length_variance", 50)
    if variance < 5 and features.get("sentence_count", 1) > 2:
        score += 0.12
    elif variance > 100:
        score -= 0.08

    # Repetition → AI
    rep = features.get("repetition_ratio", 0)
    score += rep * 0.3

    # High punctuation density → AI (list-style)
    punct = features.get("punct_density", 0)
    if punct > 0.04:
        score += 0.08

    # Personal anecdote → human
    if features.get("has_personal_anecdote"):
        score -= 0.2
    if features.get("starts_with_i"):
        score -= 0.08

    # AI openers → AI
    if features.get("has_ai_opener"):
        score += 0.15

    # Very long comments → slightly more AI
    word_count = features.get("word_count", 50)
    if word_count > 300:
        score += 0.08
    elif word_count < 30:
        score -= 0.05

    return max(0.0, min(1.0, score))
