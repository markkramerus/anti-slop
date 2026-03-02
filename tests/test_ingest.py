"""
Tests for the ingest pipeline: csv_loader, schema_mapper, normalizer, validators.
"""

from __future__ import annotations

import io
import textwrap

import pandas as pd
import pytest

from core.ingest.csv_loader import load_csv_bytes
from core.ingest.normalizer import normalize_dataframe
from core.ingest.schema_mapper import detect_mapping
from core.ingest.validators import validate_and_summarize


SAMPLE_CSV = textwrap.dedent("""\
Document ID,Comment,First Name,State/Province
ABC-001,This is a human comment.,Alice,NY
ABC-002,Another comment here.,Bob,CA
ABC-003,,Carol,TX
ABC-001,Duplicate ID comment.,Dave,FL
""")


def make_csv_bytes(content: str) -> bytes:
    return content.encode("utf-8")


class TestCsvLoader:
    def test_load_basic(self):
        raw = make_csv_bytes(SAMPLE_CSV)
        df, fingerprint = load_csv_bytes(raw, filename="test.csv")
        assert len(df) == 4
        assert "Document ID" in df.columns
        assert "Comment" in df.columns
        assert len(fingerprint) == 64  # sha256 hex

    def test_fingerprint_stable(self):
        raw = make_csv_bytes(SAMPLE_CSV)
        _, fp1 = load_csv_bytes(raw)
        _, fp2 = load_csv_bytes(raw)
        assert fp1 == fp2

    def test_fingerprint_differs_on_different_content(self):
        raw1 = make_csv_bytes(SAMPLE_CSV)
        raw2 = make_csv_bytes(SAMPLE_CSV + "\nABC-005,Extra row.,Eve,OH\n")
        _, fp1 = load_csv_bytes(raw1)
        _, fp2 = load_csv_bytes(raw2)
        assert fp1 != fp2


class TestSchemaMapper:
    def test_detect_standard_columns(self):
        raw = make_csv_bytes(SAMPLE_CSV)
        df, _ = load_csv_bytes(raw)
        mapping = detect_mapping(df)
        id_col = mapping["id_col"] if isinstance(mapping, dict) else mapping.id_col
        comment_col = mapping["comment_col"] if isinstance(mapping, dict) else mapping.comment_col
        assert id_col == "Document ID"
        assert comment_col == "Comment"

    def test_override_columns(self):
        csv = "doc_id,text\nX-001,Hello world.\n"
        df, _ = load_csv_bytes(make_csv_bytes(csv))
        mapping = detect_mapping(df, id_col_override="doc_id", comment_col_override="text")
        id_col = mapping["id_col"] if isinstance(mapping, dict) else mapping.id_col
        comment_col = mapping["comment_col"] if isinstance(mapping, dict) else mapping.comment_col
        assert id_col == "doc_id"
        assert comment_col == "text"

    def test_missing_required_column_raises(self):
        csv = "only_col\nvalue1\n"
        df, _ = load_csv_bytes(make_csv_bytes(csv))
        with pytest.raises(ValueError):
            detect_mapping(df)


class TestNormalizer:
    def test_basic_normalization(self):
        raw = make_csv_bytes(SAMPLE_CSV)
        df, _ = load_csv_bytes(raw)
        mapping = detect_mapping(df)
        records, norm_df = normalize_dataframe(df, mapping)
        assert "comment_id" in norm_df.columns
        assert "comment_text" in norm_df.columns
        assert "text_hash" in norm_df.columns
        assert len(norm_df) == 4

    def test_comment_id_mapped_from_document_id(self):
        raw = make_csv_bytes(SAMPLE_CSV)
        df, _ = load_csv_bytes(raw)
        mapping = detect_mapping(df)
        _, norm_df = normalize_dataframe(df, mapping)
        assert "ABC-001" in norm_df["comment_id"].values

    def test_text_hash_computed(self):
        raw = make_csv_bytes(SAMPLE_CSV)
        df, _ = load_csv_bytes(raw)
        mapping = detect_mapping(df)
        _, norm_df = normalize_dataframe(df, mapping)
        valid = norm_df[norm_df["comment_text"].notna() & (norm_df["comment_text"] != "")]
        assert valid["text_hash"].notna().all()
        assert (valid["text_hash"].str.len() == 64).all()


class TestValidators:
    def test_summary_totals(self):
        raw = make_csv_bytes(SAMPLE_CSV)
        df, fp = load_csv_bytes(raw)
        mapping = detect_mapping(df)
        _, norm_df = normalize_dataframe(df, mapping)
        summary = validate_and_summarize(norm_df, fp)
        assert summary.total_rows == 4
        assert summary.rows_missing_comment == 1
        assert summary.duplicate_comment_ids == 1  # ABC-001 appears twice

    def test_length_distribution(self):
        raw = make_csv_bytes(SAMPLE_CSV)
        df, fp = load_csv_bytes(raw)
        mapping = detect_mapping(df)
        _, norm_df = normalize_dataframe(df, mapping)
        summary = validate_and_summarize(norm_df, fp)
        assert summary.length_distribution.min >= 0
        assert summary.length_distribution.max > 0

    def test_fingerprint_stored(self):
        raw = make_csv_bytes(SAMPLE_CSV)
        df, fp = load_csv_bytes(raw)
        mapping = detect_mapping(df)
        _, norm_df = normalize_dataframe(df, mapping)
        summary = validate_and_summarize(norm_df, fp)
        assert summary.dataset_fingerprint == fp
