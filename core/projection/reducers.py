"""
Dimensionality reduction: PCA, UMAP, t-SNE.

Each reducer takes a (n, dim) numpy matrix and returns (n, 2) or (n, 3) coordinates.
Yields progress messages for UI display.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Generator

import numpy as np
import pandas as pd

import config
from core.storage.paths import projection_coords_path, projection_manifest_path
from core.storage.tables import write_table
from shared_models import ProjectionMethod, ProjectionRunManifest


# ── PCA ───────────────────────────────────────────────────────────────────────

def _run_pca(
    matrix: np.ndarray,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(matrix).astype(np.float32)


# ── UMAP ──────────────────────────────────────────────────────────────────────

def _run_umap(
    matrix: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> np.ndarray:
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn not installed. Run: pip install umap-learn")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        low_memory=False,
    )
    return reducer.fit_transform(matrix).astype(np.float32)


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def _run_tsne(
    matrix: np.ndarray,
    n_components: int,
    perplexity: float,
    random_state: int,
) -> np.ndarray:
    from sklearn.manifold import TSNE
    # t-SNE is slow on high-dim; pre-reduce with PCA first
    if matrix.shape[1] > 50:
        from sklearn.decomposition import PCA
        matrix = PCA(n_components=50, random_state=random_state).fit_transform(matrix)
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
    )
    return tsne.fit_transform(matrix).astype(np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def run_projection(
    project_id: str,
    embedding_run_id: str,
    comment_ids: list[str],
    matrix: np.ndarray,
    method: str = "umap",
    dims: int = 2,
    params: dict | None = None,
    run_id: str | None = None,
) -> Generator[tuple[float, str], None, ProjectionRunManifest]:
    """
    Run dimensionality reduction and save coordinates + manifest.

    Yields (progress, message) and returns a ProjectionRunManifest.
    """
    run_id = run_id or str(uuid.uuid4())
    params = params or {}
    meth = ProjectionMethod(method)

    random_state = params.get("random_state", config.DEFAULT_RANDOM_SEED)

    yield 0.0, f"Starting {method.upper()} projection (dims={dims}, n={len(comment_ids)})…"

    coords: np.ndarray

    if meth == ProjectionMethod.PCA:
        yield 0.1, "Running PCA…"
        coords = _run_pca(matrix, dims, random_state)

    elif meth == ProjectionMethod.UMAP:
        n_neighbors = params.get("n_neighbors", config.DEFAULT_UMAP_N_NEIGHBORS)
        min_dist = params.get("min_dist", config.DEFAULT_UMAP_MIN_DIST)
        yield 0.1, f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})…"
        coords = _run_umap(matrix, dims, n_neighbors, min_dist, random_state)

    elif meth == ProjectionMethod.TSNE:
        perplexity = params.get("perplexity", 30.0)
        yield 0.1, f"Running t-SNE (perplexity={perplexity})…"
        coords = _run_tsne(matrix, dims, perplexity, random_state)

    else:
        raise ValueError(f"Unknown projection method: {method!r}")

    yield 0.9, "Saving projection coordinates…"

    # Build DataFrame
    coord_cols = ["x", "y", "z"][:dims]
    rows = []
    for cid, row_coords in zip(comment_ids, coords):
        r = {"comment_id": cid}
        for col, val in zip(coord_cols, row_coords):
            r[col] = float(val)
        rows.append(r)
    coords_df = pd.DataFrame(rows)

    path = projection_coords_path(project_id, run_id)
    write_table(coords_df, path)

    manifest = ProjectionRunManifest(
        run_id=run_id,
        embedding_run_id=embedding_run_id,
        method=meth,
        dims=dims,
        params={**params, "random_state": random_state},
        row_count=len(comment_ids),
    )

    manifest_path = projection_manifest_path(project_id, run_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest.model_dump(), f, indent=2, default=str)

    yield 1.0, f"✓ Projection done: {len(comment_ids)} points in {dims}D"
    return manifest


def load_projection_coords(project_id: str, run_id: str) -> pd.DataFrame:
    """Load projection coordinates DataFrame (comment_id, x, y[, z])."""
    from core.storage.tables import read_table
    path = projection_coords_path(project_id, run_id)
    df = read_table(path)
    if df.empty:
        raise FileNotFoundError(f"No projection found for run {run_id}")
    return df
