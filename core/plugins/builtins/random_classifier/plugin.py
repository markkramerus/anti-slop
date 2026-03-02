"""
Random Classifier

Assigns each comment a random label (ai_generated / human / uncertain)
drawn from configurable probability weights and a fixed random seed.

Purpose: plumbing / framework smoke-test. Metrics on a truth key should
come out near chance, confirming end-to-end execution without any real
signal from the classifier itself.
"""

from __future__ import annotations

import random
from typing import Any

import pandas as pd
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.plugins.base import BasePlugin, PluginContext, PluginRunResult
from shared_models import PluginKind, PluginMetadata, PluginParameterSpec


class Plugin(BasePlugin):
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="random_classifier",
            version="0.1.0",
            kind=PluginKind.CLASSIFIER,
            author="anti_slop_team",
            description="Random baseline classifier for framework plumbing tests",
            entrypoint="plugin.py:Plugin",
            inputs={"required": ["comment_text"]},
            outputs={"per_comment": ["pred_label", "score_ai", "confidence", "explanation_json"]},
            parameters={
                "seed": PluginParameterSpec(
                    type="int", default=42, description="Random seed for reproducible outputs"
                ),
                "p_ai": PluginParameterSpec(
                    type="float", default=0.33, min=0.0, max=1.0,
                    description="Probability of ai_generated",
                ),
                "p_human": PluginParameterSpec(
                    type="float", default=0.33, min=0.0, max=1.0,
                    description="Probability of human",
                ),
                "p_uncertain": PluginParameterSpec(
                    type="float", default=0.34, min=0.0, max=1.0,
                    description="Probability of uncertain",
                ),
            },
        )

    def validate_inputs(
        self,
        dataset: pd.DataFrame,
        available_artifacts: dict[str, Any],
        params: dict[str, Any],
    ) -> list[str]:
        errors: list[str] = []
        if "comment_id" not in dataset.columns:
            errors.append("Missing required column: comment_id")
        if "comment_text" not in dataset.columns:
            errors.append("Missing required column: comment_text")
        p_ai = float(params.get("p_ai", 0.33))
        p_human = float(params.get("p_human", 0.33))
        p_uncertain = float(params.get("p_uncertain", 0.34))
        if (p_ai + p_human + p_uncertain) <= 0:
            errors.append("At least one probability (p_ai, p_human, p_uncertain) must be > 0")
        return errors

    def run(
        self,
        dataset: pd.DataFrame,
        available_artifacts: dict[str, Any],
        params: dict[str, Any],
        context: PluginContext,
    ) -> PluginRunResult:
        seed = int(params.get("seed", 42))
        p_ai = float(params.get("p_ai", 0.33))
        p_human = float(params.get("p_human", 0.33))
        p_uncertain = float(params.get("p_uncertain", 0.34))

        # Normalize weights so they sum to 1
        total = p_ai + p_human + p_uncertain
        p_ai /= total
        p_human /= total
        p_uncertain /= total

        rng = random.Random(seed)
        labels = ["ai_generated", "human", "uncertain"]
        weights = [p_ai, p_human, p_uncertain]

        results: list[dict[str, Any]] = []
        n = len(dataset)

        # Use enumerate so the loop counter is independent of the DataFrame index
        for loop_idx, (_, row) in enumerate(dataset.iterrows()):
            comment_id = row["comment_id"]
            pred = rng.choices(labels, weights=weights, k=1)[0]

            # Score mimics what a real classifier would produce
            if pred == "ai_generated":
                score_ai = rng.uniform(0.5, 1.0)
            elif pred == "human":
                score_ai = rng.uniform(0.0, 0.5)
            else:
                score_ai = rng.uniform(0.4, 0.6)

            confidence = abs(score_ai - 0.5) * 2.0

            results.append(
                {
                    "comment_id": comment_id,
                    "pred_label": pred,
                    "score_ai": round(float(score_ai), 4),
                    "confidence": round(float(confidence), 4),
                    "explanation_json": {
                        "method": "random_baseline",
                        "seed": seed,
                        "probabilities": {
                            "ai_generated": round(p_ai, 4),
                            "human": round(p_human, 4),
                            "uncertain": round(p_uncertain, 4),
                        },
                    },
                }
            )

            if loop_idx % 500 == 0 and n > 0:
                context.log(f"Processed {loop_idx + 1}/{n} comments")

        df = pd.DataFrame(results)
        return PluginRunResult(
            kind=PluginKind.CLASSIFIER,
            per_comment_df=df,
            run_metadata={
                "provider": "local",
                "model_name": "random_classifier_v0.1",
                "seed": seed,
            },
            logs=[f"Randomly classified {len(df)} comments"],
            error_count=0,
        )
