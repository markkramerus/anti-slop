"""
Parquet read/write helpers with consistent schema handling.

All table I/O in the platform goes through these helpers to
ensure encoding consistency and optional CSV mirrors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import config


def write_table(df: pd.DataFrame, path: Path, *, csv_mirror: bool | None = None) -> None:
    """Persist a DataFrame as Parquet (and optionally a CSV mirror)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine=config.PARQUET_ENGINE, index=False)

    should_mirror = csv_mirror if csv_mirror is not None else config.SAVE_CSV_MIRRORS
    if should_mirror:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)


def read_table(path: Path) -> pd.DataFrame:
    """Load a Parquet table; returns empty DataFrame if file missing."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, engine=config.PARQUET_ENGINE)


def append_table(df_new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Append rows to an existing Parquet table (or create it)."""
    existing = read_table(path)
    combined = pd.concat([existing, df_new], ignore_index=True)
    write_table(combined, path)
    return combined


def upsert_table(
    df_new: pd.DataFrame,
    path: Path,
    key_col: str,
) -> pd.DataFrame:
    """
    Upsert rows into a Parquet table keyed on `key_col`.
    Existing rows with matching keys are replaced; new rows are appended.
    """
    existing = read_table(path)
    if existing.empty:
        write_table(df_new, path)
        return df_new

    # Drop existing rows that have updated versions in df_new
    keys_to_update = set(df_new[key_col].tolist())
    kept = existing[~existing[key_col].isin(keys_to_update)]
    combined = pd.concat([kept, df_new], ignore_index=True)
    write_table(combined, path)
    return combined


def delete_rows(path: Path, key_col: str, keys: list[Any]) -> pd.DataFrame:
    """Delete rows matching `keys` from a Parquet table."""
    df = read_table(path)
    if df.empty:
        return df
    df = df[~df[key_col].isin(keys)]
    write_table(df, path)
    return df
