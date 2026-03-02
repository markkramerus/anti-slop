"""
Export service — produces CSV/Parquet artifacts for research output.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

from core.storage.paths import (
    annotations_path,
    classifier_runs_dir,
    enrichments_dir,
    exports_dir,
    normalized_comments_path,
)
from core.storage.tables import read_table


def export_annotations(project_id: str, fmt: str = "csv") -> bytes:
    df = read_table(annotations_path(project_id))
    return _serialize(df, fmt)


def export_predictions(project_id: str, run_id: str, fmt: str = "csv") -> bytes:
    path = classifier_runs_dir(project_id) / f"{run_id}_predictions.parquet"
    df = read_table(path)
    return _serialize(df, fmt)


def export_enrichments(project_id: str, run_id: str, fmt: str = "csv") -> bytes:
    path = enrichments_dir(project_id) / f"{run_id}_output.parquet"
    df = read_table(path)
    return _serialize(df, fmt)


def export_merged(project_id: str, fmt: str = "csv") -> bytes:
    """Export comments + annotations merged."""
    comments = read_table(normalized_comments_path(project_id))
    ann = read_table(annotations_path(project_id))
    if not ann.empty:
        merged = comments.merge(ann[["comment_id", "label", "note"]], on="comment_id", how="left")
    else:
        merged = comments
    return _serialize(merged, fmt)


def export_metrics_json(metrics_dict: dict) -> bytes:
    return json.dumps(metrics_dict, indent=2, default=str).encode("utf-8")


def _serialize(df: pd.DataFrame, fmt: str) -> bytes:
    if fmt == "parquet":
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return buf.getvalue()
    else:
        return df.to_csv(index=False).encode("utf-8")
