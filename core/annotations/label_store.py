"""
Persistence layer for manual annotations (labels + notes).

Annotations are stored as a Parquet table at:
  projects/<project_id>/annotations/annotations.parquet

Each comment_id has at most one annotation (upsert on save).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from shared_models import ManualAnnotation, ManualLabel
from core.storage.paths import annotations_path
from core.storage.tables import read_table, upsert_table


# ── Read ──────────────────────────────────────────────────────────────────────


def load_annotations(project_id: str) -> pd.DataFrame:
    """Load all annotations as a DataFrame. Returns empty DF if none exist."""
    return read_table(annotations_path(project_id))


def get_annotation(project_id: str, comment_id: str) -> ManualAnnotation | None:
    """Return the annotation for a single comment, or None if unlabeled."""
    df = load_annotations(project_id)
    if df.empty:
        return None
    row = df[df["comment_id"] == comment_id]
    if row.empty:
        return None
    r = row.iloc[0]
    return ManualAnnotation(
        annotation_id=str(r.get("annotation_id", "")),
        comment_id=str(r["comment_id"]),
        label=ManualLabel(r["label"]),
        confidence=float(r.get("confidence", 1.0)),
        note=r.get("note") or None,
        created_at=_parse_dt(r.get("created_at")),
        updated_at=_parse_dt(r.get("updated_at")),
    )


# ── Write ────────────────────────────────────────────────────────────────────


def save_annotation(project_id: str, annotation: ManualAnnotation) -> None:
    """Upsert a single annotation (insert or update by comment_id)."""
    row = {
        "annotation_id": annotation.annotation_id,
        "comment_id": annotation.comment_id,
        "label": annotation.label.value,
        "confidence": annotation.confidence,
        "note": annotation.note or "",
        "created_at": annotation.created_at.isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    df_new = pd.DataFrame([row])
    path = annotations_path(project_id)
    upsert_table(df_new, path, key_col="comment_id")


def delete_annotation(project_id: str, comment_id: str) -> None:
    """Remove an annotation (un-label a comment)."""
    from core.storage.tables import delete_rows
    delete_rows(annotations_path(project_id), "comment_id", [comment_id])


# ── Bulk helpers ──────────────────────────────────────────────────────────────


def label_counts(project_id: str) -> dict[str, int]:
    """Return {label: count} dict for quick progress tracking."""
    df = load_annotations(project_id)
    if df.empty:
        return {label.value: 0 for label in ManualLabel}
    counts = df["label"].value_counts().to_dict()
    # Ensure all labels present
    return {label.value: int(counts.get(label.value, 0)) for label in ManualLabel}


def annotated_comment_ids(project_id: str) -> set[str]:
    """Return set of comment_ids that have been labeled."""
    df = load_annotations(project_id)
    if df.empty:
        return set()
    return set(df["comment_id"].tolist())


# ── Private helpers ───────────────────────────────────────────────────────────


def _parse_dt(val: object) -> datetime:
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val)
        except ValueError:
            pass
    return datetime.utcnow()
