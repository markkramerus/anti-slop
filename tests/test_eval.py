"""
Tests for truth key loading, joining, and metrics computation.
"""

from __future__ import annotations

import textwrap

import pandas as pd
import pytest

from core.eval.truth_loader import join_truth_key, load_truth_key
from core.eval.metrics import compute_metrics


TRUTH_CSV = textwrap.dedent("""\
Document ID,type
ABC-001,real
ABC-002,synthetic
ABC-003,real
ABC-004,synthetic
ABC-005,real
""")

COMBINED_CSV = textwrap.dedent("""\
Document ID,type
ABC-001,real
ABC-002,synthetic
ABC-003,real
ABC-004,synthetic
ABC-005,real
""")


def make_bytes(s: str) -> bytes:
    return s.encode("utf-8")


class TestTruthLoader:
    def test_load_basic(self):
        df, warnings = load_truth_key(make_bytes(TRUTH_CSV))
        assert len(df) == 5
        assert "comment_id" in df.columns
        assert "truth_label" in df.columns
        assert len(warnings) == 0

    def test_label_normalization(self):
        df, _ = load_truth_key(make_bytes(TRUTH_CSV))
        labels = set(df["truth_label"])
        assert labels == {"human", "ai_generated"}

    def test_unrecognized_label_warns(self):
        csv = "Document ID,type\nABC-001,unknown_label\n"
        df, warnings = load_truth_key(make_bytes(csv))
        assert len(warnings) > 0
        assert len(df) == 0  # row dropped

    def test_join_all_match(self):
        truth_df, _ = load_truth_key(make_bytes(TRUTH_CSV))
        comments_df = pd.DataFrame({
            "comment_id": ["ABC-001", "ABC-002", "ABC-003", "ABC-004", "ABC-005"],
            "comment_text": ["text"] * 5,
        })
        matched, diag = join_truth_key(comments_df, truth_df)
        assert diag.matched_rows == 5
        assert diag.coverage_pct == 100.0
        assert len(matched) == 5

    def test_join_partial_match(self):
        truth_df, _ = load_truth_key(make_bytes(TRUTH_CSV))
        comments_df = pd.DataFrame({
            "comment_id": ["ABC-001", "ABC-002", "EXTRA-999"],
            "comment_text": ["text"] * 3,
        })
        matched, diag = join_truth_key(comments_df, truth_df)
        assert diag.matched_rows == 2
        assert diag.unmatched_truth_rows == 3

    def test_case_insensitive_join(self):
        truth_df, _ = load_truth_key(make_bytes(TRUTH_CSV))
        comments_df = pd.DataFrame({
            "comment_id": ["abc-001"],  # lowercase
            "comment_text": ["text"],
        })
        matched, diag = join_truth_key(comments_df, truth_df)
        assert diag.matched_rows == 1

    def test_duplicate_truth_keys_handled(self):
        dup_csv = "Document ID,type\nABC-001,real\nABC-001,synthetic\n"
        truth_df, _ = load_truth_key(make_bytes(dup_csv))
        comments_df = pd.DataFrame({"comment_id": ["ABC-001"], "comment_text": ["text"]})
        matched, diag = join_truth_key(comments_df, truth_df)
        assert diag.duplicate_truth_keys >= 1
        assert len(matched) == 1  # only one row after dedup


class TestMetrics:
    def _make_eval_df(self):
        return pd.DataFrame({
            "comment_id": [f"C-{i}" for i in range(10)],
            "truth_label": ["ai_generated"] * 5 + ["human"] * 5,
            "pred_label": ["ai_generated"] * 4 + ["human"] + ["human"] * 4 + ["ai_generated"],
            "score_ai": [0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1, 0.15, 0.25, 0.85],
        })

    def test_compute_basic_metrics(self):
        df = self._make_eval_df()
        metrics = compute_metrics(df)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.precision_ai <= 1.0
        assert 0.0 <= metrics.recall_ai <= 1.0
        assert 0.0 <= metrics.f1_ai <= 1.0

    def test_confusion_matrix_sums(self):
        df = self._make_eval_df()
        metrics = compute_metrics(df)
        cm = metrics.confusion_matrix
        total = cm.tp + cm.tn + cm.fp + cm.fn
        assert total == 10

    def test_support_counts(self):
        df = self._make_eval_df()
        metrics = compute_metrics(df)
        assert metrics.support_ai == 5
        assert metrics.support_human == 5

    def test_uncertain_exclude(self):
        df = pd.DataFrame({
            "truth_label": ["ai_generated", "human", "ai_generated"],
            "pred_label": ["ai_generated", "uncertain", "human"],
            "comment_id": ["C-1", "C-2", "C-3"],
        })
        metrics = compute_metrics(df, uncertain_handling="exclude")
        # Only 2 rows remain (C-2 excluded)
        assert metrics.confusion_matrix.tp + metrics.confusion_matrix.fn == 2

    def test_uncertain_map_to_human(self):
        df = pd.DataFrame({
            "truth_label": ["ai_generated", "human"],
            "pred_label": ["uncertain", "uncertain"],
            "comment_id": ["C-1", "C-2"],
        })
        metrics = compute_metrics(df, uncertain_handling="map_to_human")
        # uncertain → human means all AI truth become FN, all human truth become TN
        assert metrics.confusion_matrix.fn == 1
        assert metrics.confusion_matrix.tn == 1

    def test_roc_auc_computed(self):
        df = self._make_eval_df()
        metrics = compute_metrics(df, score_col="score_ai")
        assert metrics.roc_auc is not None
        assert 0.0 <= metrics.roc_auc <= 1.0

    def test_perfect_classifier(self):
        df = pd.DataFrame({
            "truth_label": ["ai_generated"] * 3 + ["human"] * 3,
            "pred_label": ["ai_generated"] * 3 + ["human"] * 3,
            "comment_id": [f"C-{i}" for i in range(6)],
        })
        metrics = compute_metrics(df)
        assert metrics.accuracy == 1.0
        assert metrics.f1_ai == 1.0
