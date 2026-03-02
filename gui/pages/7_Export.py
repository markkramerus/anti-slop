"""
Page 7 — Export

Download annotations, predictions, enrichments, merged tables, and metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json

import pandas as pd
import streamlit as st

from core.export.exporters import (
    export_annotations,
    export_enrichments,
    export_merged,
    export_metrics_json,
    export_predictions,
)
from core.storage.tables import read_table
from core.storage.paths import classifier_runs_dir, enrichments_dir
from gui.utils.state import get_project_id, get_project_manifest, init_defaults
from gui.utils.ui_components import require_project

init_defaults()

st.title("📦 Export")
st.markdown("Download research artifacts in CSV or Parquet format.")

if not require_project():
    st.stop()

project_id = get_project_id()
manifest = get_project_manifest()

fmt = st.radio("Export format", ["CSV", "Parquet"], horizontal=True)
ext = fmt.lower()

st.divider()

# ── Annotations ───────────────────────────────────────────────────────────────
st.subheader("Manual Annotations")
try:
    ann_bytes = export_annotations(project_id, ext)
    st.download_button(
        f"⬇️ Download annotations.{ext}",
        data=ann_bytes,
        file_name=f"annotations.{ext}",
        mime="text/csv" if ext == "csv" else "application/octet-stream",
    )
except Exception as e:
    st.warning(f"Annotations not available: {e}")

st.divider()

# ── Classifier Predictions ────────────────────────────────────────────────────
st.subheader("Classifier Predictions")
cls_runs = [r for r in (manifest.plugin_runs if manifest else []) if r.plugin_kind.value == "classifier"]
if not cls_runs:
    st.info("No classifier runs yet.")
else:
    for run in cls_runs:
        try:
            pred_bytes = export_predictions(project_id, run.run_id, ext)
            st.download_button(
                f"⬇️ {run.plugin_name} v{run.plugin_version} [{run.run_id[:8]}].{ext}",
                data=pred_bytes,
                file_name=f"predictions_{run.plugin_name}_{run.run_id[:8]}.{ext}",
                mime="text/csv" if ext == "csv" else "application/octet-stream",
                key=f"dl_pred_{run.run_id}",
            )
        except Exception as e:
            st.warning(f"Could not export {run.run_id[:8]}: {e}")

st.divider()

# ── Enrichments ───────────────────────────────────────────────────────────────
st.subheader("Enrichment Outputs")
enrich_runs = [r for r in (manifest.plugin_runs if manifest else []) if r.plugin_kind.value == "enrichment"]
if not enrich_runs:
    st.info("No enrichment runs yet.")
else:
    for run in enrich_runs:
        try:
            enr_bytes = export_enrichments(project_id, run.run_id, ext)
            st.download_button(
                f"⬇️ {run.plugin_name} [{run.run_id[:8]}].{ext}",
                data=enr_bytes,
                file_name=f"enrichments_{run.plugin_name}_{run.run_id[:8]}.{ext}",
                mime="text/csv" if ext == "csv" else "application/octet-stream",
                key=f"dl_enr_{run.run_id}",
            )
        except Exception as e:
            st.warning(f"Could not export {run.run_id[:8]}: {e}")

st.divider()

# ── Merged Analysis Table ─────────────────────────────────────────────────────
st.subheader("Merged Analysis Table")
st.caption("Comments + manual labels + available predictions (latest classifier run)")
try:
    merged_bytes = export_merged(project_id, ext)
    st.download_button(
        f"⬇️ Download merged_analysis.{ext}",
        data=merged_bytes,
        file_name=f"merged_analysis.{ext}",
        mime="text/csv" if ext == "csv" else "application/octet-stream",
    )
except Exception as e:
    st.warning(f"Merged table not available: {e}")

st.divider()

# ── Project Manifest ──────────────────────────────────────────────────────────
st.subheader("Project Manifest (JSON)")
if manifest:
    manifest_json = json.dumps(manifest.model_dump(), indent=2, default=str)
    st.download_button(
        "⬇️ Download manifest.json",
        data=manifest_json.encode("utf-8"),
        file_name="manifest.json",
        mime="application/json",
    )
    with st.expander("Preview manifest"):
        st.json(manifest.model_dump())
