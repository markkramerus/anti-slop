"""
End-to-end smoke tests.

Tests the full pipeline:
  ingest → normalize → label store → heuristic baseline → metrics
without starting the Streamlit UI.
"""

from __future__ import annotations

import textwrap

import pandas as pd
import pytest


SAMPLE_CSV = textwrap.dedent("""\
Document ID,Comment,State/Province
REAL-001,I have relied on this program for ten years. My family depends on it. Please don't cut coverage.,NY
REAL-002,As a nurse I see patients every day who need these benefits to survive. This is a lifeline.,CA
AI-001,I am writing to express my strong support for this important proposed rule which will ensure essential coverage for many patients.,TX
AI-002,It is crucial that this policy be implemented to protect the healthcare access of vulnerable populations across the nation.,FL
AI-003,The proposed rule represents a significant step forward in ensuring that all Americans have access to the comprehensive coverage they deserve.,OH
""")

TRUTH_CSV = textwrap.dedent("""\
Document ID,type
REAL-001,real
REAL-002,real
AI-001,synthetic
AI-002,synthetic
AI-003,synthetic
""")


@pytest.fixture
def smoke_project(tmp_path, monkeypatch):
    """Set up a complete project from scratch."""
    import config
    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path)

    from core.ingest.csv_loader import load_csv_bytes
    from core.ingest.normalizer import normalize_dataframe
    from core.ingest.schema_mapper import detect_mapping
    from core.ingest.validators import validate_and_summarize
    from core.storage.paths import ensure_project_dirs, normalized_comments_path
    from core.storage.project_store import create_project, update_manifest
    from core.storage.tables import write_table

    raw = SAMPLE_CSV.encode("utf-8")
    df_raw, fingerprint = load_csv_bytes(raw)
    mapping = detect_mapping(df_raw)
    _, norm_df = normalize_dataframe(df_raw, mapping)
    summary = validate_and_summarize(norm_df, fingerprint)

    manifest = create_project(
        project_name="Smoke Test",
        source_filename="test.csv",
        dataset_fingerprint=fingerprint,
    )
    write_table(norm_df, normalized_comments_path(manifest.project_id))
    manifest.ingest_summary = summary
    update_manifest(manifest)

    return manifest, norm_df


class TestEndToEndSmoke:
    def test_ingest_creates_valid_manifest(self, smoke_project):
        manifest, norm_df = smoke_project
        assert manifest.project_id
        assert manifest.ingest_summary.total_rows == 5
        assert manifest.ingest_summary.rows_missing_comment == 0

    def test_annotation_round_trip(self, smoke_project, monkeypatch, tmp_path):
        import config
        monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path)
        manifest, norm_df = smoke_project

        from core.annotations.label_store import get_annotation, save_annotation
        from shared_models import ManualAnnotation, ManualLabel

        ann = ManualAnnotation(comment_id="REAL-001", label=ManualLabel.HUMAN)
        save_annotation(manifest.project_id, ann)

        retrieved = get_annotation(manifest.project_id, "REAL-001")
        assert retrieved is not None
        assert retrieved.label == ManualLabel.HUMAN

    def test_heuristic_classifier_runs(self, smoke_project):
        manifest, norm_df = smoke_project

        from core.plugins.base import PluginContext
        from core.plugins.manager import get_registry

        registry = get_registry(refresh=True)
        plugin = registry.instantiate("heuristic_baseline")
        assert plugin is not None

        ctx = PluginContext(project_id=manifest.project_id, run_id="smoke_run", params={})
        result = plugin.run(norm_df, {}, {}, ctx)

        assert len(result.per_comment_df) == 5
        assert set(result.per_comment_df.columns) >= {"comment_id", "pred_label", "score_ai"}

    def test_enricher_runs(self, smoke_project):
        manifest, norm_df = smoke_project

        from core.plugins.base import PluginContext
        from core.plugins.manager import get_registry

        registry = get_registry(refresh=True)
        plugin = registry.instantiate("theme_stance_enricher")
        ctx = PluginContext(project_id=manifest.project_id, run_id="enrich_run", params={"provider": "local"})
        result = plugin.run(norm_df, {}, {"provider": "local"}, ctx)

        assert len(result.per_comment_df) == 5
        assert "theme_primary" in result.per_comment_df.columns

    def test_metrics_against_truth_key(self, smoke_project):
        manifest, norm_df = smoke_project

        from core.eval.metrics import compute_metrics
        from core.eval.truth_loader import join_truth_key, load_truth_key
        from core.plugins.base import PluginContext
        from core.plugins.manager import get_registry

        # Run classifier
        registry = get_registry(refresh=True)
        plugin = registry.instantiate("heuristic_baseline")
        ctx = PluginContext(project_id=manifest.project_id, run_id="smoke_eval", params={})
        result = plugin.run(norm_df, {}, {}, ctx)

        # Load truth
        truth_df, _ = load_truth_key(TRUTH_CSV.encode("utf-8"))
        matched_comments, diag = join_truth_key(norm_df, truth_df)
        assert diag.matched_rows == 5

        # Build eval df
        eval_df = matched_comments[["comment_id", "truth_label"]].merge(
            result.per_comment_df[["comment_id", "pred_label", "score_ai"]],
            on="comment_id",
            how="inner",
        )
        assert len(eval_df) == 5

        # Compute metrics
        metrics = compute_metrics(eval_df, uncertain_handling="map_to_human")
        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.confusion_matrix.tp + metrics.confusion_matrix.tn + \
               metrics.confusion_matrix.fp + metrics.confusion_matrix.fn == 5
