"""
Clustering service: HDBSCAN, KMeans, Agglomerative.

Operates on 2D/3D projection coordinates (not raw embeddings).
Yields progress messages for UI display.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Generator

import numpy as np
import pandas as pd

import config
from core.storage.paths import cluster_assignments_path, cluster_manifest_path
from core.storage.tables import write_table
from shared_models import ClusteringMethod, ClusterRunManifest


def run_clustering(
    project_id: str,
    projection_run_id: str,
    coords_df: pd.DataFrame,
    method: str = "hdbscan",
    params: dict | None = None,
    run_id: str | None = None,
) -> Generator[tuple[float, str], None, ClusterRunManifest]:
    """
    Cluster projection coordinates and save assignments.

    Yields (progress, message) and returns a ClusterRunManifest.

    Args:
        coords_df: DataFrame with columns comment_id, x, y[, z]
    """
    run_id = run_id or str(uuid.uuid4())
    params = params or {}
    meth = ClusteringMethod(method)

    coord_cols = [c for c in ["x", "y", "z"] if c in coords_df.columns]
    X = coords_df[coord_cols].values.astype(np.float32)
    comment_ids = coords_df["comment_id"].tolist()

    yield 0.0, f"Clustering {len(comment_ids)} points with {method.upper()}…"

    labels: np.ndarray

    if meth == ClusteringMethod.HDBSCAN:
        min_cluster_size = params.get("min_cluster_size", config.DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE)
        yield 0.1, f"Running HDBSCAN (min_cluster_size={min_cluster_size})…"
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                gen_min_span_tree=False,
                prediction_data=False,
            )
            labels = clusterer.fit_predict(X)
        except ImportError:
            yield 0.1, "⚠️ hdbscan not available, falling back to KMeans"
            meth = ClusteringMethod.KMEANS
            params["n_clusters"] = params.get("n_clusters", config.DEFAULT_KMEANS_N_CLUSTERS)
            from sklearn.cluster import KMeans
            km = KMeans(
                n_clusters=params["n_clusters"],
                random_state=params.get("random_state", config.DEFAULT_RANDOM_SEED),
                n_init="auto",
            )
            labels = km.fit_predict(X)

    elif meth == ClusteringMethod.KMEANS:
        n_clusters = params.get("n_clusters", config.DEFAULT_KMEANS_N_CLUSTERS)
        random_state = params.get("random_state", config.DEFAULT_RANDOM_SEED)
        yield 0.1, f"Running KMeans (k={n_clusters})…"
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)

    elif meth == ClusteringMethod.AGGLOMERATIVE:
        n_clusters = params.get("n_clusters", config.DEFAULT_KMEANS_N_CLUSTERS)
        linkage = params.get("linkage", "ward")
        yield 0.1, f"Running Agglomerative clustering (k={n_clusters}, linkage={linkage})…"
        from sklearn.cluster import AgglomerativeClustering
        ac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = ac.fit_predict(X)

    else:
        raise ValueError(f"Unknown clustering method: {method!r}")

    yield 0.9, "Saving cluster assignments…"

    # Save assignments
    assignments_df = pd.DataFrame({
        "comment_id": comment_ids,
        "cluster_id": labels.astype(int),
    })
    path = cluster_assignments_path(project_id, run_id)
    write_table(assignments_df, path)

    # Compute stats
    unique_labels = set(labels.tolist())
    noise_count = int((labels == -1).sum())  # HDBSCAN noise label
    n_clusters_found = len(unique_labels - {-1})

    manifest = ClusterRunManifest(
        run_id=run_id,
        projection_run_id=projection_run_id,
        method=meth,
        n_clusters=n_clusters_found,
        params=params,
        row_count=len(comment_ids),
        noise_count=noise_count,
    )

    mpath = cluster_manifest_path(project_id, run_id)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with open(mpath, "w") as f:
        json.dump(manifest.model_dump(), f, indent=2, default=str)

    yield 1.0, f"✓ Found {n_clusters_found} clusters ({noise_count} noise points)"
    return manifest


def load_cluster_assignments(project_id: str, run_id: str) -> pd.DataFrame:
    """Load cluster assignment DataFrame (comment_id, cluster_id)."""
    from core.storage.tables import read_table
    path = cluster_assignments_path(project_id, run_id)
    return read_table(path)
