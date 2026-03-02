"""
Embedding generation service.

Supports:
  - local: sentence-transformers (runs entirely offline)
  - openai: OpenAI text-embedding-* models (requires OPENAI_API_KEY)

Long-running. Yields (progress_fraction, log_message) tuples for UI display.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from core.features.text_utils import normalize_text, truncate_text, batch_texts
from core.storage.paths import (
    embedding_manifest_path,
    embedding_vectors_path,
    embeddings_dir,
)
from core.storage.tables import read_table, write_table
from shared_models import EmbeddingProvider, EmbeddingRunManifest


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_cache(project_id: str, run_id: str) -> dict[str, list[float]]:
    """Load existing cached embeddings as {text_hash: vector}."""
    path = embedding_vectors_path(project_id, run_id)
    if not path.exists():
        return {}
    df = read_table(path)
    if df.empty or "text_hash" not in df.columns:
        return {}
    # Vector columns are all columns except comment_id and text_hash
    vec_cols = [c for c in df.columns if c.startswith("emb_")]
    cache = {}
    for _, row in df.iterrows():
        cache[row["text_hash"]] = row[vec_cols].tolist()
    return cache


def _save_vectors(
    project_id: str,
    run_id: str,
    comment_ids: list[str],
    text_hashes: list[str],
    vectors: list[list[float]],
) -> None:
    """Persist embedding vectors to Parquet."""
    dim = len(vectors[0]) if vectors else 0
    vec_cols = [f"emb_{i}" for i in range(dim)]
    rows = []
    for cid, thash, vec in zip(comment_ids, text_hashes, vectors):
        row = {"comment_id": cid, "text_hash": thash}
        for col, val in zip(vec_cols, vec):
            row[col] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    path = embedding_vectors_path(project_id, run_id)
    write_table(df, path)


def _save_manifest(project_id: str, manifest: EmbeddingRunManifest) -> None:
    path = embedding_manifest_path(project_id, manifest.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest.model_dump(), f, indent=2, default=str)


# ── Local embeddings (sentence-transformers) ──────────────────────────────────

def _embed_local(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> Generator[tuple[float, str], None, list[list[float]]]:
    """Generator: yields progress, returns list of embedding vectors."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )

    yield 0.02, f"Loading model {model_name!r}…"
    model = SentenceTransformer(model_name)

    all_vectors: list[list[float]] = []
    batches = batch_texts(texts, batch_size)
    total = len(batches)

    for i, batch in enumerate(batches):
        vectors = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_vectors.extend(vectors.tolist())
        frac = 0.02 + 0.95 * (i + 1) / total
        yield frac, f"  Encoded batch {i+1}/{total} ({len(batch)} texts)"

    return all_vectors


# ── OpenAI embeddings ─────────────────────────────────────────────────────────

def _embed_openai(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> Generator[tuple[float, str], None, list[list[float]]]:
    """Generator: yields progress, returns list of embedding vectors."""
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    all_vectors: list[list[float]] = []
    batches = batch_texts(texts, batch_size)
    total = len(batches)

    yield 0.01, f"Starting OpenAI embeddings ({model_name!r})…"

    for i, batch in enumerate(batches):
        response = client.embeddings.create(model=model_name, input=batch)
        vecs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_vectors.extend(vecs)
        frac = 0.01 + 0.97 * (i + 1) / total
        yield frac, f"  Batch {i+1}/{total} ({len(batch)} texts)"

    return all_vectors


# ── Public API ────────────────────────────────────────────────────────────────

def generate_embeddings(
    project_id: str,
    comments_df: pd.DataFrame,
    provider: str = "local",
    model_name: str | None = None,
    batch_size: int | None = None,
    run_id: str | None = None,
) -> Generator[tuple[float, str], None, EmbeddingRunManifest]:
    """
    Generate embeddings for all comments in comments_df.

    Yields (progress_fraction, log_message) during processing.
    Returns an EmbeddingRunManifest on completion.

    Args:
        project_id: active project
        comments_df: normalized comments DataFrame (must have comment_id, comment_text, text_hash)
        provider: "local" or "openai"
        model_name: override default model
        batch_size: override default batch size
        run_id: optional fixed run_id (for reproducibility)
    """
    import uuid

    run_id = run_id or str(uuid.uuid4())
    model_name = model_name or config.DEFAULT_EMBEDDING_MODEL
    batch_size = batch_size or config.DEFAULT_EMBEDDING_BATCH_SIZE
    prov = EmbeddingProvider(provider)

    yield 0.0, f"Starting embedding run {run_id[:8]}… ({provider}/{model_name})"
    yield 0.01, f"  {len(comments_df)} comments to embed"

    # Filter to rows with non-empty text
    valid = comments_df[
        comments_df["comment_text"].notna() & (comments_df["comment_text"] != "")
    ].copy()

    if valid.empty:
        raise ValueError("No valid comment texts to embed.")

    # Prepare texts
    texts = [
        truncate_text(normalize_text(str(t)))
        for t in valid["comment_text"].tolist()
    ]
    comment_ids = valid["comment_id"].tolist()
    text_hashes = valid["text_hash"].tolist()

    yield 0.02, f"  Prepared {len(texts)} texts for embedding"

    # Run the appropriate backend
    if prov == EmbeddingProvider.LOCAL:
        gen = _embed_local(texts, model_name, batch_size)
    else:
        gen = _embed_openai(texts, model_name, batch_size)

    vectors: list[list[float]] = []
    while True:
        try:
            frac, msg = next(gen)
            yield frac, msg
        except StopIteration as e:
            vectors = e.value
            break

    yield 0.97, "Saving vectors…"
    _save_vectors(project_id, run_id, comment_ids, text_hashes, vectors)

    dim = len(vectors[0]) if vectors else 0
    manifest = EmbeddingRunManifest(
        run_id=run_id,
        provider=prov,
        model_name=model_name,
        batch_size=batch_size,
        row_count=len(vectors),
        embedding_dim=dim,
        params={"provider": provider, "model": model_name, "batch_size": batch_size},
    )
    _save_manifest(project_id, manifest)
    yield 1.0, f"✓ Embeddings saved: {len(vectors)} vectors (dim={dim})"

    return manifest


def load_embedding_vectors(project_id: str, run_id: str) -> tuple[list[str], np.ndarray]:
    """
    Load embedding vectors from disk.

    Returns:
        (comment_ids, matrix) where matrix is shape (n, dim)
    """
    path = embedding_vectors_path(project_id, run_id)
    df = read_table(path)
    if df.empty:
        raise FileNotFoundError(f"No embeddings found for run {run_id}")
    comment_ids = df["comment_id"].tolist()
    vec_cols = sorted([c for c in df.columns if c.startswith("emb_")])
    matrix = df[vec_cols].values.astype(np.float32)
    return comment_ids, matrix
