"""
Truth key loading and join diagnostics.

The truth key CSV uses:
  - document_id column (case-insensitive join to comment_id)
  - type column: "real" → "human", "synthetic" → "ai_generated"
"""

from __future__ import annotations

import io

import pandas as pd

from shared_models import TruthJoinDiagnostics, TruthLabel, TruthRecord

# ── Label normalization ───────────────────────────────────────────────────────

TRUTH_LABEL_MAP: dict[str, str] = {
    "real": "human",
    "human": "human",
    "synthetic": "ai_generated",
    "ai": "ai_generated",
    "ai_generated": "ai_generated",
    "fake": "ai_generated",
    "generated": "ai_generated",
}


def normalize_truth_label(raw: str) -> TruthLabel | None:
    """Normalize a raw truth label string to TruthLabel enum, or None if unrecognized."""
    canonical = TRUTH_LABEL_MAP.get(str(raw).strip().lower())
    if canonical is None:
        return None
    return TruthLabel(canonical)


# ── Truth key loading ─────────────────────────────────────────────────────────

def load_truth_key(
    source: bytes | str | io.IOBase,
    id_col: str | None = None,
    label_col: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load a truth key CSV and return a normalized DataFrame.

    Args:
        source: raw bytes, filepath string, or file-like object
        id_col: override auto-detected ID column name
        label_col: override auto-detected label column name

    Returns:
        (normalized_df, warnings)
        normalized_df has columns: comment_id, truth_label
    """
    warnings: list[str] = []

    if isinstance(source, bytes):
        df = pd.read_csv(io.BytesIO(source), dtype=str)
    elif isinstance(source, str):
        df = pd.read_csv(source, dtype=str)
    else:
        df = pd.read_csv(source, dtype=str)

    cols = list(df.columns)

    # Detect ID column
    if id_col:
        if id_col not in cols:
            raise ValueError(f"Specified ID column {id_col!r} not in truth key. Columns: {cols}")
    else:
        candidates = ["Document ID", "document_id", "DocumentID", "id", "doc_id"]
        id_col = next((c for c in candidates if c in cols), None)
        if id_col is None:
            # Try case-insensitive
            lower_map = {c.lower(): c for c in cols}
            id_col = next((lower_map[c.lower()] for c in candidates if c.lower() in lower_map), None)
        if id_col is None:
            raise ValueError(f"Could not find ID column in truth key. Columns: {cols}")

    # Detect label column
    if label_col:
        if label_col not in cols:
            raise ValueError(f"Specified label column {label_col!r} not in truth key.")
    else:
        label_candidates = ["type", "label", "truth", "truth_label", "ground_truth", "class"]
        label_col = next((c for c in label_candidates if c in cols), None)
        if label_col is None:
            lower_map = {c.lower(): c for c in cols}
            label_col = next((lower_map[c.lower()] for c in label_candidates if c.lower() in lower_map), None)
        if label_col is None:
            raise ValueError(f"Could not find label column in truth key. Columns: {cols}")

    # Normalize
    normalized_rows = []
    unrecognized = []
    for _, row in df.iterrows():
        cid = str(row[id_col]).strip() if pd.notna(row[id_col]) else ""
        raw_label = str(row[label_col]).strip() if pd.notna(row[label_col]) else ""
        truth = normalize_truth_label(raw_label)
        if truth is None:
            unrecognized.append(raw_label)
            continue
        normalized_rows.append({"comment_id": cid, "truth_label": truth.value})

    if unrecognized:
        unique_unrecognized = list(set(unrecognized))
        warnings.append(
            f"{len(unrecognized)} rows had unrecognized labels: {unique_unrecognized[:5]}. "
            f"Recognized values: {list(TRUTH_LABEL_MAP.keys())}"
        )

    normalized_df = pd.DataFrame(normalized_rows)
    return normalized_df, warnings


def join_truth_key(
    comments_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> tuple[pd.DataFrame, TruthJoinDiagnostics]:
    """
    Join truth key onto comments_df by comment_id.

    Returns:
        (matched_df, diagnostics)
        matched_df: inner join result with columns comment_id, truth_label, ...
    """
    # Case-insensitive join
    truth_df = truth_df.copy()
    truth_df["_join_key"] = truth_df["comment_id"].str.strip().str.upper()

    comments_df = comments_df.copy()
    comments_df["_join_key"] = comments_df["comment_id"].str.strip().str.upper()

    # Detect duplicates in truth
    dup_truth = int(truth_df["_join_key"].duplicated().sum())
    if dup_truth > 0:
        # Keep first occurrence
        truth_df = truth_df.drop_duplicates(subset="_join_key", keep="first")

    matched = comments_df.merge(
        truth_df[["_join_key", "truth_label"]],
        on="_join_key",
        how="inner",
    ).drop(columns=["_join_key"])

    truth_keys = set(truth_df["_join_key"])
    comment_keys = set(comments_df["_join_key"])
    matched_keys = truth_keys & comment_keys

    diagnostics = TruthJoinDiagnostics(
        total_truth_rows=len(truth_df),
        matched_rows=len(matched_keys),
        unmatched_truth_rows=len(truth_keys - comment_keys),
        unmatched_dataset_rows=len(comment_keys - truth_keys),
        duplicate_truth_keys=dup_truth,
        coverage_pct=round(100.0 * len(matched_keys) / max(len(truth_keys), 1), 2),
        warnings=[f"{dup_truth} duplicate keys in truth file (kept first)"] if dup_truth > 0 else [],
    )

    return matched, diagnostics
