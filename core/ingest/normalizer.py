"""
Converts a raw DataFrame (with a resolved ColumnMapping) into
canonical CommentRecord objects and a normalized Parquet table.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from core.ingest.schema_mapper import ColumnMapping
from shared_models import (
    CommentMetadata,
    CommentRecord,
    IngestStatus,
)


def normalize_dataframe(
    df: pd.DataFrame,
    mapping: ColumnMapping,
) -> tuple[list[CommentRecord], pd.DataFrame]:
    """
    Normalize a raw DataFrame into CommentRecord objects.

    Args:
        df: Raw source DataFrame
        mapping: Resolved ColumnMapping

    Returns:
        (records, normalized_df)
        - records: list of CommentRecord (for downstream processing)
        - normalized_df: flat DataFrame suitable for Parquet storage
    """
    records: list[CommentRecord] = []

    for idx, row in df.iterrows():
        comment_id = _str_or_none(row.get(mapping.id_col))
        comment_text = _str_or_none(row.get(mapping.comment_col))

        status = IngestStatus.OK
        warning: str | None = None

        if not comment_id:
            status = IngestStatus.ERROR
            warning = f"Missing comment_id at row {idx}"
            comment_id = f"__missing_id_{idx}__"

        if not comment_text:
            status = IngestStatus.WARNING
            warning = (warning or "") + f" Missing comment_text at row {idx}"
            comment_text = ""

        # Build metadata dict
        meta_kwargs: dict[str, Any] = {}
        for src_col, canonical_field in mapping.metadata_mapping.items():
            val = _str_or_none(row.get(src_col))
            meta_kwargs[canonical_field] = val

        metadata = CommentMetadata(**meta_kwargs)

        # Full raw row (serialized to strings for storage)
        raw_row = {str(k): _str_or_none(v) for k, v in row.items()}

        record = CommentRecord(
            comment_id=comment_id,
            comment_text=comment_text,
            source_row_index=int(idx),
            metadata=metadata,
            raw_row=raw_row,
            ingest_status=status,
            ingest_warning=warning,
        )
        records.append(record)

    normalized_df = _records_to_df(records)
    return records, normalized_df


def _records_to_df(records: list[CommentRecord]) -> pd.DataFrame:
    """Flatten CommentRecord objects into a wide DataFrame for Parquet."""
    rows = []
    for r in records:
        row: dict[str, Any] = {
            "comment_id": r.comment_id,
            "comment_text": r.comment_text,
            "source_row_index": r.source_row_index,
            "text_hash": r.text_hash,
            "ingest_status": r.ingest_status.value,
            "ingest_warning": r.ingest_warning,
        }
        # Flatten metadata fields
        for field, val in r.metadata.model_dump().items():
            row[f"meta_{field}"] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure consistent column ordering
    base_cols = [
        "comment_id", "comment_text", "source_row_index",
        "text_hash", "ingest_status", "ingest_warning",
    ]
    meta_cols = sorted([c for c in df.columns if c.startswith("meta_")])
    return df[base_cols + meta_cols]


def _str_or_none(val: Any) -> str | None:
    """Convert a value to string, returning None for blanks and NaN."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None
