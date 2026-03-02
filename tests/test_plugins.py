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
