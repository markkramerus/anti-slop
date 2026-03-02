"""
Tests for the plugin system: discovery, execution, heuristic_baseline, theme_stance_enricher.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_dataset():
    return pd.DataFrame([
        {"comment_id": "C-001", "comment_text": "I am writing to express my strong support for this important policy that will help many patients access the coverage they need."},
        {"comment_id": "C-002", "comment_text": "I have relied on Medicare for years. My doctor says this change will hurt me personally. Please don't do this."},
        {"comment_id": "C-003", "comment_text": "This is a short one."},
    ])


class TestPluginDiscovery:
    def test_builtin_plugins_discovered(self):
        from core.plugins.manager import discover_plugins
        registry = discover_plugins()
        metas = registry.all_metadata()
        names = [m.name for m in metas]
        assert "heuristic_baseline" in names
        assert "theme_stance_enricher" in names
        assert "simple_classifier" in names

    def test_plugin_instantiation(self):
        from core.plugins.manager import get_registry
        registry = get_registry(refresh=True)
        plugin = registry.instantiate("heuristic_baseline")
        assert plugin is not None

    def test_plugin_metadata_fields(self):
        from core.plugins.manager import get_registry
        registry = get_registry(refresh=True)
        meta = registry.get_metadata("theme_stance_enricher")
        assert meta.kind.value == "enrichment"
        assert "comment_text" in meta.inputs.get("required", [])


class TestHeuristicBaseline:
    def test_runs_on_sample(self, sample_dataset):
        from core.plugins.manager import get_registry
        from core.plugins.base import PluginContext

        registry = get_registry(refresh=True)
        plugin = registry.instantiate("heuristic_baseline")
        assert plugin is not None

        ctx = PluginContext(project_id="test", run_id="run_test", params={})
        result = plugin.run(sample_dataset, {}, {}, ctx)

        assert len(result.per_comment_df) == 3
        assert "pred_label" in result.per_comment_df.columns
        assert "score_ai" in result.per_comment_df.columns

    def test_score_in_range(self, sample_dataset):
        from core.plugins.manager import get_registry
        from core.plugins.base import PluginContext

        registry = get_registry(refresh=True)
        plugin = registry.instantiate("heuristic_baseline")
        ctx = PluginContext(project_id="test", run_id="run_test", params={})
        result = plugin.run(sample_dataset, {}, {}, ctx)

        scores = result.per_comment_df["score_ai"].dropna()
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_short_comment_uncertain(self, sample_dataset):
        from core.plugins.manager import get_registry
        from core.plugins.base import PluginContext

        registry = get_registry(refresh=True)
        plugin = registry.instantiate("heuristic_baseline")
        ctx = PluginContext(project_id="test", run_id="run_test", params={"min_length": 50})
        result = plugin.run(sample_dataset, {}, {}, ctx)

        short_row = result.per_comment_df[result.per_comment_df["comment_id"] == "C-003"]
        assert short_row.iloc[0]["pred_label"] == "uncertain"


class TestThemeStanceEnricher:
    def test_runs_local_mode(self, sample_dataset):
        from core.plugins.manager import get_registry
        from core.plugins.base import PluginContext

        registry = get_registry(refresh=True)
        plugin = registry.instantiate("theme_stance_enricher")
        assert plugin is not None

        ctx = PluginContext(project_id="test", run_id="run_test", params={"provider": "local"})
        result = plugin.run(sample_dataset, {}, {"provider": "local"}, ctx)

        assert len(result.per_comment_df) == 3
        assert "theme_primary" in result.per_comment_df.columns
        assert "stance" in result.per_comment_df.columns

    def test_personal_comment_stance(self, sample_dataset):
        from core.plugins.manager import get_registry
        from core.plugins.base import PluginContext

        registry = get_registry(refresh=True)
        plugin = registry.instantiate("theme_stance_enricher")
        ctx = PluginContext(project_id="test", run_id="run_test", params={"provider": "local"})
        result = plugin.run(sample_dataset, {}, {"provider": "local"}, ctx)

        # C-002 mentions personal reliance on Medicare → likely "con" or "mixed"
        row = result.per_comment_df[result.per_comment_df["comment_id"] == "C-002"]
        assert row.iloc[0]["stance"] in ["con", "mixed", "neutral"]

    def test_all_comments_covered(self, sample_dataset):
        from core.plugins.manager import get_registry
        from core.plugins.base import PluginContext

        registry = get_registry(refresh=True)
        plugin = registry.instantiate("theme_stance_enricher")
        ctx = PluginContext(project_id="test", run_id="run_test", params={"provider": "local"})
        result = plugin.run(sample_dataset, {}, {"provider": "local"}, ctx)

        assert set(result.per_comment_df["comment_id"]) == {"C-001", "C-002", "C-003"}


class TestRandomClassifier:
    def test_discovered_by_registry(self):
        from core.plugins.manager import discover_plugins
        registry = discover_plugins()
        names = [m.name for m in registry.all_metadata()]
        assert "random_classifier" in names

    def test_is_classifier_kind(self):
        from core.plugins.manager import get_registry
        from shared_models import PluginKind
        registry = get_registry(refresh=True)
        meta = registry.get_metadata("random_classifier")
        assert meta is not None
        assert meta.kind == PluginKind.CLASSIFIER

    def test_runs_and_returns_all_rows(self, sample_dataset):
        from core.plugins.manager import get_registry
        from core.plugins.base import PluginContext

        registry = get_registry(refresh=True)
        plugin = registry.instantiate("random_classifier")
        assert plugin is not None

        ctx = PluginContext(project_id="test", run_id="run_rand", params={})
        result = plugin.run(sample_dataset, {}, {}, ctx)

        assert len(result.per_comment_df) == 3
        assert set(result.per_comment_df["comment_id"]) == {"C-001", "C-002", "C-003"}

    def test_output_columns_present(self, sample_dataset):
        from core.plugins.manager import get_registry
        from core.plugins.base import PluginContext

        registry = get_registry(refresh=True)
        plugin = registry.instantiate("random_classifier")
        ctx = PluginContext(project_id="test", run_id="run_rand", params={})
        result = plugin.run(sample_dataset, {}, {}, ctx)

        df = result.per_comment_df
        for col in ("comment_id", "pred_label", "score_ai", "confidence", "explanation_json"):
            assert col in df.columns, f"Missing column: {col}"

    def test_scores_in_range(self, sample_dataset):
        from core.plugins.manager import get_registry
        from core.plugins.base import PluginContext

        registry = get_registry(refresh=True)
        plugin = registry.instantiate("random_classifier")
        ctx = PluginContext(project_id="test", run_id="run_rand", params={})
        result = plugin.run(sample_dataset, {}, {}, ctx)

        scores = result.per_comment_df["score_ai"].dropna()
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

        confidence = result.per_comment_df["confidence"].dropna()
        assert (confidence >= 0.0).all()
        assert (confidence <= 1.0).all()

    def test_pred_labels_valid(self, sample_dataset):
        from core.plugins.manager import get_registry
        from core.plugins.base import PluginContext

        registry = get_registry(refresh=True)
        plugin = registry.instantiate("random_classifier")
        ctx = PluginContext(project_id="test", run_id="run_rand", params={})
        result = plugin.run(sample_dataset, {}, {}, ctx)

        valid_labels = {"ai_generated", "human", "uncertain"}
        for label in result.per_comment_df["pred_label"]:
            assert label in valid_labels, f"Unexpected label: {label}"

    def test_reproducible_with_same_seed(self, sample_dataset):
        from core.plugins.manager import get_registry
        from core.plugins.base import PluginContext

        registry = get_registry(refresh=True)

        ctx_a = PluginContext(project_id="test", run_id="run_a", params={})
        plugin_a = registry.instantiate("random_classifier")
        result_a = plugin_a.run(sample_dataset, {}, {"seed": 99}, ctx_a)

        ctx_b = PluginContext(project_id="test", run_id="run_b", params={})
        plugin_b = registry.instantiate("random_classifier")
        result_b = plugin_b.run(sample_dataset, {}, {"seed": 99}, ctx_b)

        assert list(result_a.per_comment_df["pred_label"]) == list(result_b.per_comment_df["pred_label"])
        assert list(result_a.per_comment_df["score_ai"]) == list(result_b.per_comment_df["score_ai"])

    def test_different_seeds_may_differ(self, sample_dataset):
        """With different seeds on a larger dataset the outputs should differ."""
        from core.plugins.manager import get_registry
        from core.plugins.base import PluginContext

        # Build a bigger dataset so the probability of all labels matching is negligible
        big_dataset = pd.concat([sample_dataset] * 20, ignore_index=True)
        big_dataset["comment_id"] = [f"C-{i:04d}" for i in range(len(big_dataset))]

        registry = get_registry(refresh=True)

        ctx_a = PluginContext(project_id="test", run_id="run_a", params={})
        result_a = registry.instantiate("random_classifier").run(big_dataset, {}, {"seed": 1}, ctx_a)

        ctx_b = PluginContext(project_id="test", run_id="run_b", params={})
        result_b = registry.instantiate("random_classifier").run(big_dataset, {}, {"seed": 2}, ctx_b)

        assert list(result_a.per_comment_df["pred_label"]) != list(result_b.per_comment_df["pred_label"])

    def test_validation_rejects_missing_columns(self):
        from core.plugins.manager import get_registry

        registry = get_registry(refresh=True)
        plugin = registry.instantiate("random_classifier")

        bad_df = pd.DataFrame([{"other_col": "x"}])
        errors = plugin.validate_inputs(bad_df, {}, {"p_ai": 0.33, "p_human": 0.33, "p_uncertain": 0.34})
        assert any("comment_id" in e for e in errors)
        assert any("comment_text" in e for e in errors)

    def test_run_metadata_fields(self, sample_dataset):
        from core.plugins.manager import get_registry
        from core.plugins.base import PluginContext

        registry = get_registry(refresh=True)
        plugin = registry.instantiate("random_classifier")
        ctx = PluginContext(project_id="test", run_id="run_rand", params={})
        result = plugin.run(sample_dataset, {}, {}, ctx)

        assert result.run_metadata.get("provider") == "local"
        assert "random_classifier" in result.run_metadata.get("model_name", "")
        assert result.error_count == 0
