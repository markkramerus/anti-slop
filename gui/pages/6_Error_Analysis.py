"""
Page 6 — Error Analysis

Drill into false positives and false negatives from a classifier run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

import config
from core.eval.truth_loader import join_truth_key
from core.storage.paths import classifier_runs_dir, enrichments_dir, normalized_comments_path
from core.storage.tables import read_table
from gui.utils.state import (
    get_project_id, get_project_manifest, get_truth_key,
    init_defaults, set_open_in_review,
)
from gui.utils.ui_components import require_project, label_badge

init_defaults()

st.title("🔎 Error Analysis")

if not config.DEV_MODE:
    st.warning("Error Analysis requires DEV_MODE=true.")
    st.stop()

if not require_project():
    st.stop()

project_id = get_project_id()
manifest = get_project_manifest()
truth_df = get_truth_key()

if truth_df is None or truth_df.empty:
    st.info("Upload a truth key on the **Evaluation** page first.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_comments(pid: str) -> pd.DataFrame:
    return read_table(normalized_comments_path(pid))

comments_df = load_comments(project_id)
matched_comments, diag = join_truth_key(comments_df, truth_df)

cls_runs = [r for r in (manifest.plugin_runs if manifest else []) if r.plugin_kind.value == "classifier"]
if not cls_runs:
    st.info("No classifier runs found. Run a classifier on page 4 first.")
    st.stop()

run_options = {f"{r.plugin_name} [{r.run_id[:8]}]": r for r in cls_runs}
selected_name = st.selectbox("Classifier run", list(run_options.keys()))
selected_run = run_options[selected_name]

pred_path = classifier_runs_dir(project_id) / f"{selected_run.run_id}_predictions.parquet"
pred_df = read_table(pred_path) if pred_path.exists() else pd.DataFrame()

if pred_df.empty:
    st.error("Predictions not found.")
    st.stop()

# Merge everything
eval_df = matched_comments[["comment_id", "comment_text", "truth_label"]].merge(
    pred_df, on="comment_id", how="inner"
)

# Merge enrichments
enrichment_runs = [r for r in (manifest.plugin_runs if manifest else []) if r.plugin_kind.value == "enrichment"]
if enrichment_runs:
    enrich_path = enrichments_dir(project_id) / f"{enrichment_runs[-1].run_id}_output.parquet"
    if enrich_path.exists():
        enrich_df = read_table(enrich_path)
        merge_cols = ["comment_id"] + [c for c in ["theme_primary", "stance"] if c in enrich_df.columns]
        eval_df = eval_df.merge(enrich_df[merge_cols], on="comment_id", how="left")

# ── Error tables ──────────────────────────────────────────────────────────────

fp_df = eval_df[(eval_df["truth_label"] == "human") & (eval_df["pred_label"] == "ai_generated")].copy()
fn_df = eval_df[(eval_df["truth_label"] == "ai_generated") & (eval_df["pred_label"] == "human")].copy()

# Sort by confidence
if "confidence" in fp_df.columns:
    fp_df = fp_df.sort_values("confidence", ascending=False)
if "confidence" in fn_df.columns:
    fn_df = fn_df.sort_values("confidence", ascending=False)

col1, col2 = st.columns(2)
col1.metric("False Positives (Human → Pred AI)", len(fp_df))
col2.metric("False Negatives (AI → Pred Human)", len(fn_df))

tab_fp, tab_fn = st.tabs([f"🔴 False Positives ({len(fp_df)})", f"🟢 False Negatives ({len(fn_df)})"])

def render_error_table(df: pd.DataFrame, error_type: str) -> None:
    if df.empty:
        st.success(f"No {error_type}!")
        return

    # Grouping options
    group_cols = [c for c in ["theme_primary", "stance"] if c in df.columns]
    group_by = st.selectbox(f"Group {error_type} by", ["(none)"] + group_cols, key=f"group_{error_type}")

    if group_by != "(none)":
        for grp_val, grp_df in df.groupby(group_by):
            with st.expander(f"{group_by}: **{grp_val}** ({len(grp_df)} errors)"):
                _render_error_rows(grp_df)
    else:
        _render_error_rows(df)


def _render_error_rows(df: pd.DataFrame) -> None:
    show_cols = ["comment_id", "comment_text"] + [
        c for c in ["score_ai", "confidence", "theme_primary", "stance"]
        if c in df.columns
    ]
    display_df = df[show_cols].copy()
    if "comment_text" in display_df.columns:
        display_df["comment_text"] = display_df["comment_text"].str[:150] + "…"

    st.dataframe(display_df, use_container_width=True, hide_index=True, height=300)

    selected_id = st.text_input("Open comment ID in Review:", key=f"open_id_{id(df)}")
    if selected_id and st.button("📖 Open in Review", key=f"open_btn_{id(df)}"):
        set_open_in_review(selected_id.strip())
        st.switch_page("pages/2_Review.py")


with tab_fp:
    render_error_table(fp_df, "false_positives")

with tab_fn:
    render_error_table(fn_df, "false_negatives")
