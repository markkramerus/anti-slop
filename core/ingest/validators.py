"""
Validates a normalized comments DataFrame and produces an IngestSummary.

Run after normalization to give the analyst a clear picture of data quality
before proceeding to review/analysis.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from shared_models import IngestSummary, LengthDistribution


def validate_and_summarize(
    normalized_df: pd.DataFrame,
    dataset_fingerprint: str,
) -> IngestSummary:
    """
    Compute IngestSummary from a normalized comments DataFrame.

    Args:
        normalized_df: output of normalizer.normalize_dataframe
        dataset_fingerprint: SHA-256 of the raw source file

    Returns:
        IngestSummary with counts, distribution, warnings, errors
    """
    warnings: list[str] = []
    errors: list[str] = []

    total_rows = len(normalized_df)
    rows_with_comment = int((normalized_df["comment_text"].notna() & (normalized_df["comment_text"] != "")).sum())
    rows_missing_comment = total_rows - rows_with_comment

    # Duplicate comment IDs
    dup_ids = int(normalized_df["comment_id"].duplicated().sum())
    if dup_ids > 0:
        errors.append(f"{dup_ids} duplicate Document ID(s) found — this should be zero.")

    # Duplicate text hashes (likely copy-paste duplicates)
    dup_hashes = int(normalized_df["text_hash"].duplicated().sum())
    if dup_hashes > 0:
        warnings.append(f"{dup_hashes} duplicate text hash(es) found (possible copy-paste submissions).")

    if rows_missing_comment > 0:
        warnings.append(f"{rows_missing_comment} row(s) have missing comment text and will be skipped in analysis.")

    # Comment length distribution (characters)
    text_col = normalized_df["comment_text"].fillna("").astype(str)
    lengths = text_col.str.len()

    dist = LengthDistribution(
        min=int(lengths.min()),
        max=int(lengths.max()),
        mean=float(lengths.mean()),
        median=float(lengths.median()),
        p25=float(lengths.quantile(0.25)),
        p75=float(lengths.quantile(0.75)),
    )

    return IngestSummary(
        total_rows=total_rows,
        rows_with_comment=rows_with_comment,
        rows_missing_comment=rows_missing_comment,
        duplicate_comment_ids=dup_ids,
        duplicate_text_hashes=dup_hashes,
        length_distribution=dist,
        dataset_fingerprint=dataset_fingerprint,
        warnings=warnings,
        errors=errors,
    )
