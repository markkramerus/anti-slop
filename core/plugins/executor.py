"""
Plugin execution engine.

Runs enrichment and classifier plugins against the comment dataset,
persists outputs, and updates the project manifest.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.plugins.base import BasePlugin, PluginContext, PluginRunResult
from core.plugins.manager import PluginRegistry
from core.storage.paths import (
    classifier_manifest_path,
    classifier_output_path,
    enrichment_manifest_path,
    enrichment_output_path,
)
from core.storage.tables import write_table
from shared_models import PluginKind, PluginMetadata, PluginRunManifest


def run_plugin(
    plugin: BasePlugin,
    project_id: str,
    dataset: pd.DataFrame,
    user_params: dict[str, Any] | None = None,
    available_artifacts: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> Generator[tuple[float, str], None, PluginRunManifest]:
    """
    Execute a plugin against the dataset.

    Yields (progress_fraction, log_message) during processing.
    Returns a PluginRunManifest on completion.

    Args:
        plugin: instantiated BasePlugin subclass
        project_id: active project ID
        dataset: normalized comments DataFrame
        user_params: overrides for plugin parameters
        available_artifacts: embeddings, enrichments, etc.
        run_id: optional fixed run ID for reproducibility
    """
    run_id = run_id or str(uuid.uuid4())
    user_params = user_params or {}
    available_artifacts = available_artifacts or {}

    meta = plugin.metadata()
    params = plugin.resolve_params(user_params)

    yield 0.0, f"Starting {meta.kind.value} plugin: {meta.name} v{meta.version}"
    yield 0.02, f"  {len(dataset)} comments to process"

    # Validate inputs
    errors = plugin.validate_inputs(dataset, available_artifacts, params)
    if errors:
        for err in errors:
            yield 0.0, f"❌ Validation error: {err}"
        raise ValueError(f"Plugin validation failed: {errors}")

    yield 0.05, "  Inputs validated ✓"

    # Set up logging callback
    logs: list[str] = []

    def log_fn(msg: str) -> None:
        logs.append(msg)

    context = PluginContext(
        project_id=project_id,
        run_id=run_id,
        params=params,
        embeddings_df=available_artifacts.get("embeddings"),
        enrichments=available_artifacts.get("enrichments", {}),
        _log_fn=log_fn,
    )

    yield 0.1, "  Running plugin…"

    # Execute
    started_at = datetime.utcnow()
    try:
        result: PluginRunResult = plugin.run(dataset, available_artifacts, params, context)
    except Exception as e:
        yield 1.0, f"❌ Plugin crashed: {e}"
        raise

    finished_at = datetime.utcnow()
    elapsed = (finished_at - started_at).total_seconds()

    yield 0.85, f"  Completed in {elapsed:.1f}s ({result.error_count} errors)"

    # Persist outputs
    if meta.kind == PluginKind.ENRICHMENT:
        out_path = enrichment_output_path(project_id, run_id)
        mfst_path = enrichment_manifest_path(project_id, run_id)
    else:
        out_path = classifier_output_path(project_id, run_id)
        mfst_path = classifier_manifest_path(project_id, run_id)

    yield 0.9, "  Saving outputs…"
    write_table(result.per_comment_df, out_path)

    # Build run manifest
    run_manifest = PluginRunManifest(
        run_id=run_id,
        plugin_name=meta.name,
        plugin_version=meta.version,
        plugin_kind=meta.kind,
        params=params,
        provider=result.run_metadata.get("provider"),
        model_name=result.run_metadata.get("model_name"),
        prompt_hash=result.run_metadata.get("prompt_hash"),
        started_at=started_at,
        finished_at=finished_at,
        row_count=len(result.per_comment_df),
        error_count=result.error_count,
        status="complete",
    )

    # Save manifest JSON
    mfst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mfst_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest.model_dump(), f, indent=2, default=str)

    yield 1.0, (
        f"✓ {meta.kind.value.title()} run complete: "
        f"{len(result.per_comment_df)} rows, {result.error_count} errors"
    )
    return run_manifest
