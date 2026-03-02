"""
Canonical artifact paths for a project workspace.

Every artifact written or read by any core module must use
these helpers to ensure consistent directory layout.
"""

from pathlib import Path

import config


def project_root(project_id: str) -> Path:
    return config.PROJECTS_ROOT / project_id


def raw_dir(project_id: str) -> Path:
    return project_root(project_id) / "raw"


def normalized_dir(project_id: str) -> Path:
    return project_root(project_id) / "normalized"


def annotations_dir(project_id: str) -> Path:
    return project_root(project_id) / "annotations"


def embeddings_dir(project_id: str) -> Path:
    return project_root(project_id) / "embeddings"


def projections_dir(project_id: str) -> Path:
    return project_root(project_id) / "projections"


def clusters_dir(project_id: str) -> Path:
    return project_root(project_id) / "clusters"


def enrichments_dir(project_id: str) -> Path:
    return project_root(project_id) / "enrichments"


def classifier_runs_dir(project_id: str) -> Path:
    return project_root(project_id) / "classifier_runs"


def evaluation_dir(project_id: str) -> Path:
    return project_root(project_id) / "evaluation"


def exports_dir(project_id: str) -> Path:
    return project_root(project_id) / "exports"


def manifest_path(project_id: str) -> Path:
    return project_root(project_id) / "manifest.json"


# ── Specific artifact files ───────────────────────────────────────────────────

def normalized_comments_path(project_id: str) -> Path:
    return normalized_dir(project_id) / "comments.parquet"


def annotations_path(project_id: str) -> Path:
    return annotations_dir(project_id) / "annotations.parquet"


def embedding_vectors_path(project_id: str, run_id: str) -> Path:
    return embeddings_dir(project_id) / f"{run_id}_vectors.parquet"


def embedding_manifest_path(project_id: str, run_id: str) -> Path:
    return embeddings_dir(project_id) / f"{run_id}_manifest.json"


def projection_coords_path(project_id: str, run_id: str) -> Path:
    return projections_dir(project_id) / f"{run_id}_coords.parquet"


def projection_manifest_path(project_id: str, run_id: str) -> Path:
    return projections_dir(project_id) / f"{run_id}_manifest.json"


def cluster_assignments_path(project_id: str, run_id: str) -> Path:
    return clusters_dir(project_id) / f"{run_id}_assignments.parquet"


def cluster_manifest_path(project_id: str, run_id: str) -> Path:
    return clusters_dir(project_id) / f"{run_id}_manifest.json"


def enrichment_output_path(project_id: str, run_id: str) -> Path:
    return enrichments_dir(project_id) / f"{run_id}_output.parquet"


def enrichment_manifest_path(project_id: str, run_id: str) -> Path:
    return enrichments_dir(project_id) / f"{run_id}_manifest.json"


def classifier_output_path(project_id: str, run_id: str) -> Path:
    return classifier_runs_dir(project_id) / f"{run_id}_predictions.parquet"


def classifier_manifest_path(project_id: str, run_id: str) -> Path:
    return classifier_runs_dir(project_id) / f"{run_id}_manifest.json"


def evaluation_metrics_path(project_id: str, run_id: str) -> Path:
    return evaluation_dir(project_id) / f"{run_id}_metrics.json"


def evaluation_matched_path(project_id: str, run_id: str) -> Path:
    return evaluation_dir(project_id) / f"{run_id}_matched.parquet"


def evaluation_manifest_path(project_id: str, run_id: str) -> Path:
    return evaluation_dir(project_id) / f"{run_id}_manifest.json"


def ensure_project_dirs(project_id: str) -> None:
    """Create all artifact subdirectories for a new project."""
    for d in [
        raw_dir(project_id),
        normalized_dir(project_id),
        annotations_dir(project_id),
        embeddings_dir(project_id),
        projections_dir(project_id),
        clusters_dir(project_id),
        enrichments_dir(project_id),
        classifier_runs_dir(project_id),
        evaluation_dir(project_id),
        exports_dir(project_id),
    ]:
        d.mkdir(parents=True, exist_ok=True)
