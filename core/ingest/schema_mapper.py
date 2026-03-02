"""
Maps source CSV columns to canonical internal field names.

The primary required mappings are:
  Document ID  →  comment_id
  Comment      →  comment_text

All other columns are mapped to their canonical metadata equivalents
(if recognized) or passed through as-is in raw_row.
"""

from __future__ import annotations

import pandas as pd


# ── Default column name candidates ───────────────────────────────────────────

_ID_CANDIDATES = ["Document ID", "document_id", "DocumentID", "doc_id", "id"]
_COMMENT_CANDIDATES = ["Comment", "comment", "comment_text", "text", "body"]

# Source CSV column → CommentMetadata field name
_METADATA_MAP: dict[str, str] = {
    "Agency ID": "agency_id",
    "Docket ID": "docket_id",
    "Tracking Number": "tracking_number",
    "Document Type": "document_type",
    "Posted Date": "posted_date",
    "Is Withdrawn?": "is_withdrawn",
    "Title": "title",
    "Topics": "topics",
    "Duplicate Comments": "duplicate_comments",
    "First Name": "first_name",
    "Last Name": "last_name",
    "City": "city",
    "State/Province": "state_province",
    "Country": "country",
    "Organization Name": "organization_name",
    "Government Agency": "government_agency",
    "Category": "category",
    "Attachment Files": "attachment_files",
}


class ColumnMapping:
    """Resolved mapping from source columns to canonical names."""

    def __init__(
        self,
        id_col: str,
        comment_col: str,
        metadata_mapping: dict[str, str],
    ) -> None:
        self.id_col = id_col
        self.comment_col = comment_col
        self.metadata_mapping = metadata_mapping  # source_col → metadata field

    def __repr__(self) -> str:
        return (
            f"ColumnMapping(id_col={self.id_col!r}, "
            f"comment_col={self.comment_col!r}, "
            f"metadata_cols={list(self.metadata_mapping.keys())})"
        )


def detect_mapping(
    df: pd.DataFrame,
    id_col_override: str | None = None,
    comment_col_override: str | None = None,
) -> ColumnMapping:
    """
    Auto-detect column mapping from a raw DataFrame.

    Raises ValueError if required columns cannot be identified.
    """
    cols = list(df.columns)

    # Resolve ID column
    if id_col_override:
        if id_col_override not in cols:
            raise ValueError(
                f"Specified ID column {id_col_override!r} not found. "
                f"Available columns: {cols}"
            )
        id_col = id_col_override
    else:
        id_col = _find_first(cols, _ID_CANDIDATES)
        if not id_col:
            raise ValueError(
                f"Could not find a Document ID column. "
                f"Expected one of {_ID_CANDIDATES}. "
                f"Available columns: {cols}. "
                f"Please specify the column name manually."
            )

    # Resolve Comment column
    if comment_col_override:
        if comment_col_override not in cols:
            raise ValueError(
                f"Specified Comment column {comment_col_override!r} not found. "
                f"Available columns: {cols}"
            )
        comment_col = comment_col_override
    else:
        comment_col = _find_first(cols, _COMMENT_CANDIDATES)
        if not comment_col:
            raise ValueError(
                f"Could not find a Comment text column. "
                f"Expected one of {_COMMENT_CANDIDATES}. "
                f"Available columns: {cols}. "
                f"Please specify the column name manually."
            )

    # Build metadata mapping (source col → canonical field name)
    meta_map: dict[str, str] = {}
    for src_col, canonical in _METADATA_MAP.items():
        if src_col in cols:
            meta_map[src_col] = canonical

    return ColumnMapping(
        id_col=id_col,
        comment_col=comment_col,
        metadata_mapping=meta_map,
    )


def _find_first(cols: list[str], candidates: list[str]) -> str | None:
    """Return first candidate that appears in cols (case-insensitive fallback)."""
    col_set = set(cols)
    # Exact match first
    for c in candidates:
        if c in col_set:
            return c
    # Case-insensitive match
    lower_map = {col.lower(): col for col in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None
