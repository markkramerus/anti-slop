"""
Raw CSV loading with encoding detection.
Returns a raw DataFrame plus a file-level fingerprint hash.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


# Encodings to try in order
_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]


def load_csv(path: Path | str) -> tuple[pd.DataFrame, str]:
    """
    Load a CSV file into a DataFrame.

    Returns:
        (df, file_fingerprint)
        - df: raw DataFrame
        - file_fingerprint: SHA-256 of the raw file bytes
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    fingerprint = hashlib.sha256(raw_bytes).hexdigest()

    df: pd.DataFrame | None = None
    last_err: Exception | None = None
    for enc in _ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False, dtype=str)
            break
        except (UnicodeDecodeError, Exception) as e:
            last_err = e

    if df is None:
        raise ValueError(f"Could not read CSV {path}: {last_err}")

    return df, fingerprint


def load_csv_bytes(raw_bytes: bytes, filename: str = "upload.csv") -> tuple[pd.DataFrame, str]:
    """
    Load a CSV from raw bytes (e.g., from Streamlit file upload).

    Returns:
        (df, file_fingerprint)
    """
    fingerprint = hashlib.sha256(raw_bytes).hexdigest()

    import io
    df: pd.DataFrame | None = None
    last_err: Exception | None = None
    for enc in _ENCODINGS:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, low_memory=False, dtype=str)
            break
        except (UnicodeDecodeError, Exception) as e:
            last_err = e

    if df is None:
        raise ValueError(f"Could not parse uploaded CSV {filename!r}: {last_err}")

    return df, fingerprint
