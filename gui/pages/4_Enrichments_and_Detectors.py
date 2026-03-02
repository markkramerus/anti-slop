"""
Page 4 — Enrichments and Detectors

Run plugins (enrichment + classifier) against the comment corpus.
Dynamically generates parameter forms from plugin.yaml metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from core.plugins.executor import run_plugin
from core.plugins.manager import get_registry
from core.storage.paths import (
    enrichments_dir,
    normalized_comments_path,
    classifier_runs_dir,
)
from core.storage.project_store import update_manifest
from core.storage.tables import read_table
from gui.utils.state import (
    get_project_id,
    get_project_manifest,
    init_defaults,
    refresh_manifest,
)
from gui.utils.ui_components import require_project
from shared_models import PluginKind

init_defaults()

st.title("🔬 Enrichments & Detectors")
st.markdown("Discover and run analysis plugins against your comment corpus.")

if not require_project():
    st.stop()

project_id = get_project_id()
manifest = get_project_manifest()

# ── Discover plugins ──────────────────────────────────────────────────────────

registry = get_registry(refresh=True)
all_plugins = registry.all_metadata()

if not all_plugins:
    st.warning("No plugins found. Check that builtin plugins are in `core/plugins/builtins/`.")
    st.stop()

# ── Load dataset ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_dataset(pid: str) -> pd.DataFrame:
    return read_table(normalized_comments_path(pid))

dataset = load_dataset(project_id)

# ── Plugin selector ───────────────────────────────────────────────────────────

enrichers = registry.by_kind(PluginKind.ENRICHMENT)
classifiers = registry.by_kind(PluginKind.CLASSIFIER)

tab_enrich, tab_classify, tab_history = st.tabs(["🧪 Enrichment Plugins", "🤖 Classifier Plugins", "📋 Run History"])

# ── Tab: Enrichment Plugins ───────────────────────────────────────────────────

with tab_enrich:
    if not enrichers:
        st.info("No enrichment plugins found.")
    else:
        for meta in enrichers:
            with st.expander(f"**{meta.name}** v{meta.version}", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(meta.description or "")
                    if meta.outputs.get("per_comment"):
                        st.caption(f"Outputs: {', '.join(meta.outputs['per_comment'])}")
                with col2:
                    st.caption(f"Kind: `{meta.kind.value}`")
                    st.caption(f"Author: {meta.author or '—'}")

                # Parameter form
                user_params = {}
                if meta.parameters:
                    st.markdown("**Parameters:**")
                    for param_name, spec in meta.parameters.items():
                        key = f"enrich_{meta.name}_{param_name}"
                        if spec.type == "enum" and spec.values:
                            user_params[param_name] = st.selectbox(
                                param_name, spec.values,
                                index=spec.values.index(spec.default) if spec.default in spec.values else 0,
                                key=key,
                                help=spec.description,
                            )
                        elif spec.type == "float":
                            user_params[param_name] = st.slider(
                                param_name, float(spec.min or 0.0), float(spec.max or 1.0),
                                float(spec.default or 0.5), key=key, help=spec.description,
                            )
                        elif spec.type == "int":
                            user_params[param_name] = st.number_input(
                                param_name, value=int(spec.default or 1), min_value=1, key=key,
                                help=spec.description,
                            )
                        elif spec.type == "bool":
                            user_params[param_name] = st.checkbox(
                                param_name, value=bool(spec.default), key=key, help=spec.description,
                            )
                        else:
                            user_params[param_name] = st.text_input(
                                param_name, value=str(spec.default or ""), key=key, help=spec.description,
                            )

                run_key = f"run_enrich_{meta.name}"
                if st.button(f"▶ Run {meta.name}", key=run_key, type="primary"):
                    plugin = registry.instantiate(meta.name, meta.version)
                    if plugin is None:
                        st.error(f"Failed to instantiate plugin {meta.name}")
                    else:
                        progress_bar = st.progress(0.0)
                        log_area = st.empty()
                        logs = []
                        run_manifest = None
                        try:
                            gen = run_plugin(
                                plugin=plugin,
                                project_id=project_id,
                                dataset=dataset,
                                user_params=user_params,
                            )
                            while True:
                                try:
                                    frac, msg = next(gen)
                                    progress_bar.progress(min(frac, 1.0))
                                    logs.append(msg)
                                    log_area.code("\n".join(logs[-15:]))
                                except StopIteration as e:
                                    run_manifest = e.value
                                    break
                        except Exception as ex:
                            st.error(f"Plugin error: {ex}")

                        if run_manifest:
                            manifest.plugin_runs.append(run_manifest)
                            update_manifest(manifest)
                            refresh_manifest()
                            st.success(
                                f"✅ {meta.name} complete: "
                                f"{run_manifest.row_count} rows, "
                                f"{run_manifest.error_count} errors"
                            )
                            load_dataset.clear()
                            st.rerun()

# ── Tab: Classifier Plugins ───────────────────────────────────────────────────

with tab_classify:
    if not classifiers:
        st.info("No classifier plugins found.")
    else:
        # Load available enrichment outputs for use as inputs
        enrichment_runs = [r for r in (manifest.plugin_runs if manifest else [])
                           if r.plugin_kind.value == "enrichment"]
        available_enrichments: dict[str, pd.DataFrame] = {}
        for run in enrichment_runs:
            path = enrichments_dir(project_id) / f"{run.run_id}_output.parquet"
            if path.exists():
                df = read_table(path)
                run_label = f"{run.plugin_name} v{run.plugin_version} [{run.run_id[:8]}]"
                available_enrichments[run_label] = df

        for meta in classifiers:
            with st.expander(f"**{meta.name}** v{meta.version}", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(meta.description or "")
                with col2:
                    st.caption(f"Author: {meta.author or '—'}")

                # Enrichment prerequisite selector
                selected_enrichments: dict[str, pd.DataFrame] = {}
                if available_enrichments:
                    selected_enrich_keys = st.multiselect(
                        "Use enrichment outputs (optional)",
                        list(available_enrichments.keys()),
                        key=f"enrich_select_{meta.name}",
                    )
                    for k in selected_enrich_keys:
                        selected_enrichments[k] = available_enrichments[k]
                elif enrichment_runs:
                    st.info("Enrichment outputs not loaded — run an enrichment plugin first.")

                # Parameter form
                user_params = {}
                if meta.parameters:
                    st.markdown("**Parameters:**")
                    for param_name, spec in meta.parameters.items():
                        key = f"cls_{meta.name}_{param_name}"
                        if spec.type == "enum" and spec.values:
                            user_params[param_name] = st.selectbox(
                                param_name, spec.values,
                                index=spec.values.index(spec.default) if spec.default in spec.values else 0,
                                key=key, help=spec.description,
                            )
                        elif spec.type == "float":
                            user_params[param_name] = st.slider(
                                param_name, float(spec.min or 0.0), float(spec.max or 1.0),
                                float(spec.default or 0.5), key=key, help=spec.description,
                            )
                        elif spec.type == "int":
                            user_params[param_name] = st.number_input(
                                param_name, value=int(spec.default or 1), key=key,
                                help=spec.description,
                            )
                        elif spec.type == "bool":
                            user_params[param_name] = st.checkbox(
                                param_name, value=bool(spec.default), key=key, help=spec.description,
                            )
                        else:
                            user_params[param_name] = st.text_input(
                                param_name, value=str(spec.default or ""), key=key,
                            )

                run_key = f"run_cls_{meta.name}"
                if st.button(f"▶ Run {meta.name}", key=run_key, type="primary"):
                    plugin = registry.instantiate(meta.name, meta.version)
                    if plugin is None:
                        st.error(f"Failed to instantiate plugin {meta.name}")
                    else:
                        progress_bar = st.progress(0.0)
                        log_area = st.empty()
                        logs = []
                        run_manifest = None
                        artifacts = {"enrichments": selected_enrichments}
                        try:
                            gen = run_plugin(
                                plugin=plugin,
                                project_id=project_id,
                                dataset=dataset,
                                user_params=user_params,
                                available_artifacts=artifacts,
                            )
                            while True:
                                try:
                                    frac, msg = next(gen)
                                    progress_bar.progress(min(frac, 1.0))
                                    logs.append(msg)
                                    log_area.code("\n".join(logs[-15:]))
                                except StopIteration as e:
                                    run_manifest = e.value
                                    break
                        except Exception as ex:
                            st.error(f"Plugin error: {ex}")

                        if run_manifest:
                            manifest.plugin_runs.append(run_manifest)
                            update_manifest(manifest)
                            refresh_manifest()
                            st.success(
                                f"✅ {meta.name} complete: "
                                f"{run_manifest.row_count} rows"
                            )
                            load_dataset.clear()
                            st.rerun()

# ── Tab: Run History ──────────────────────────────────────────────────────────

with tab_history:
    if not manifest or not manifest.plugin_runs:
        st.info("No plugin runs yet.")
    else:
        run_rows = []
        for run in reversed(manifest.plugin_runs):
            run_rows.append({
                "Run ID": run.run_id[:12],
                "Plugin": run.plugin_name,
                "Version": run.plugin_version,
                "Kind": run.plugin_kind.value,
                "Status": run.status,
                "Rows": run.row_count,
                "Errors": run.error_count,
                "Provider": run.provider or "—",
                "Started": str(run.started_at)[:19] if run.started_at else "—",
            })
        st.dataframe(pd.DataFrame(run_rows), use_container_width=True, hide_index=True)

        # Compare classifier runs
        cls_runs = [r for r in manifest.plugin_runs if r.plugin_kind.value == "classifier"]
        if len(cls_runs) >= 2:
            st.subheader("Classifier Run Comparison")
            compare_data = []
            for run in cls_runs:
                path = classifier_runs_dir(project_id) / f"{run.run_id}_predictions.parquet"
                if not path.exists():
                    continue
                df = read_table(path)
                if df.empty:
                    continue
                label_counts = df["pred_label"].value_counts().to_dict()
                compare_data.append({
                    "Run": f"{run.plugin_name} [{run.run_id[:8]}]",
                    "AI Generated": label_counts.get("ai_generated", 0),
                    "Human": label_counts.get("human", 0),
                    "Uncertain": label_counts.get("uncertain", 0),
                    "AI%": f"{100*label_counts.get('ai_generated',0)//max(run.row_count,1)}%",
                })
            if compare_data:
                st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)
