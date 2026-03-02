"""
Project workspace management.

Handles creation, listing, loading, and updating of project manifests.
Each project gets a unique project_id (UUID) and a workspace directory
under projects/<project_id>/.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from shared_models import ProjectManifest
from core.storage.paths import ensure_project_dirs, manifest_path


# ── Serialization helpers ─────────────────────────────────────────────────────


def _serialize(obj: Any) -> Any:
    """Custom JSON serializer for datetime and Pydantic objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_manifest(manifest: ProjectManifest) -> None:
    """Persist a ProjectManifest to manifest.json inside the project workspace."""
    manifest.updated_at = datetime.utcnow()
    path = manifest_path(manifest.project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2, default=_serialize)


def load_manifest(project_id: str) -> ProjectManifest:
    """Load a ProjectManifest from disk.  Raises FileNotFoundError if missing."""
    path = manifest_path(project_id)
    if not path.exists():
        raise FileNotFoundError(f"No manifest found for project {project_id!r} at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ProjectManifest(**data)


def create_project(
    project_name: str,
    source_filename: str,
    dataset_fingerprint: str,
) -> ProjectManifest:
    """
    Create a new project workspace and persist an initial manifest.

    Returns the new ProjectManifest.
    """
    manifest = ProjectManifest(
        project_name=project_name,
        source_filename=source_filename,
        dataset_fingerprint=dataset_fingerprint,
    )
    ensure_project_dirs(manifest.project_id)
    save_manifest(manifest)
    return manifest


def list_projects() -> list[ProjectManifest]:
    """Return all projects found under PROJECTS_ROOT, sorted newest first."""
    manifests: list[ProjectManifest] = []
    for child in config.PROJECTS_ROOT.iterdir():
        if not child.is_dir():
            continue
        mp = child / "manifest.json"
        if mp.exists():
            try:
                manifests.append(load_manifest(child.name))
            except Exception:
                pass  # skip corrupt manifests
    manifests.sort(key=lambda m: m.created_at, reverse=True)
    return manifests


def get_project(project_id: str) -> ProjectManifest | None:
    """Return a ProjectManifest by ID, or None if not found."""
    try:
        return load_manifest(project_id)
    except FileNotFoundError:
        return None


def update_manifest(manifest: ProjectManifest) -> None:
    """Alias for save_manifest — signals an intentional update."""
    save_manifest(manifest)


def delete_project(project_id: str) -> None:
    """
    Permanently delete a project workspace directory and all its contents.

    Raises FileNotFoundError if the project directory does not exist.
    """
    project_dir = config.PROJECTS_ROOT / project_id
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")
    shutil.rmtree(project_dir)
