"""
Page 5 — Evaluation (Development/Test Mode)

Upload truth key CSV, join on Document ID, compute metrics
against a selected classifier run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from core.eval.metrics import compute_metrics, compute_pr_curve, compute_roc_curve, stratified_metrics
from core.eval.truth_loader import join_truth_key, load_truth_key
from core.storage.paths import classifier_runs_dir, normalized_comments_path, enrichments_dir
from core.storage.tables import read_table
from gui.utils.state import get_project_id, get_project_manifest, init_defaults, set_truth_key, get_truth_key, get_truth_key_filename
from gui.utils.ui_components import require_project

init_defaults()

st.title("📊 Evaluation")

if not config.DEV_MODE:
    st.warning("Evaluation (truth key mode) is disabled. Set DEV_MODE=true in environment.")
    st.stop()

if not require_project():
    st.stop()

project_id = get_project_id()
manifest = get_project_manifest()

@st.cache_data(ttl=60)
def load_comments(pid: str) -> pd.DataFrame:
    return read_table(normalized_comments_path(pid))

comments_df = load_comments(project_id)

# ── Truth Key Upload ──────────────────────────────────────────────────────────

st.subheader("1. Truth Key")
truth_file = st.file_uploader("Upload truth key CSV", type=["csv"], key="truth_uploader")

if truth_file:
    try:
        raw = truth_file.read()
        truth_df, warnings = load_truth_key(raw)
        set_truth_key(truth_df, truth_file.name)
        st.success(f"✓ Loaded {len(truth_df)} truth records from `{truth_file.name}`")
        for w in warnings:
            st.warning(w)
    except Exception as e:
        st.error(f"Failed to load truth key: {e}")

truth_df = get_truth_key()
truth_filename = get_truth_key_filename()

if truth_df is None or truth_df.empty:
    st.info("Upload a truth key to proceed.")
    st.stop()

# Join diagnostics
matched_comments, diag = join_truth_key(comments_df, truth_df)
st.subheader("Join Diagnostics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Truth Rows", diag.total_truth_rows)
col2.metric("Matched", diag.matched_rows)
col3.metric("Coverage", f"{diag.coverage_pct:.1f}%")
col4.metric("Unmatched Truth", diag.unmatched_truth_rows)
for w in diag.warnings:
    st.warning(w)

# ── Select Classifier Run ─────────────────────────────────────────────────────

st.subheader("2. Select Classifier Run")
cls_runs = [r for r in (manifest.plugin_runs if manifest else []) if r.plugin_kind.value == "classifier"]

if not cls_runs:
    st.info("No classifier runs found. Run a classifier on page 4 first.")
    st.stop()

run_options = {f"{r.plugin_name} v{r.plugin_version} [{r.run_id[:8]}]": r for r in cls_runs}
selected_run_name = st.selectbox("Classifier run", list(run_options.keys()), key="eval_cls_select")
selected_run = run_options[selected_run_name]

pred_path = classifier_runs_dir(project_id) / f"{selected_run.run_id}_predictions.parquet"
if not pred_path.exists():
    st.error(f"Predictions file not found: {pred_path}")
    st.stop()
pred_df = read_table(pred_path)

# ── Evaluation Options ────────────────────────────────────────────────────────

st.subheader("3. Options")
uncertain_handling = st.selectbox(
    "Uncertain label handling",
    ["exclude", "map_to_human", "third_class"],
    help="exclude = drop uncertain predictions from metrics; map_to_human = count as human; third_class = count as-is",
    key="uncertain_handling",
)

# ── Compute Metrics ───────────────────────────────────────────────────────────

if st.button("🔢 Compute Metrics", type="primary"):
    # Merge: matched_comments + predictions
    eval_df = matched_comments[["comment_id", "truth_label"]].merge(
        pred_df[["comment_id", "pred_label"] + (["score_ai"] if "score_ai" in pred_df.columns else [])],
        on="comment_id",
        how="inner",
    )

    if eval_df.empty:
        st.error("No overlapping comments between truth key and classifier output.")
        st.stop()

    try:
        metrics = compute_metrics(
            eval_df,
            pred_col="pred_label",
            score_col="score_ai" if "score_ai" in eval_df.columns else None,
            uncertain_handling=uncertain_handling,
        )
    except Exception as e:
        st.error(f"Metrics computation failed: {e}")
        st.stop()

    # ── Metrics Dashboard ─────────────────────────────────────────────────────
    st.subheader("Metrics")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics.accuracy:.3f}")
    m2.metric("Precision (AI)", f"{metrics.precision_ai:.3f}")
    m3.metric("Recall (AI)", f"{metrics.recall_ai:.3f}")
    m4.metric("F1 (AI)", f"{metrics.f1_ai:.3f}")
    m5.metric("Specificity", f"{metrics.specificity:.3f}")

    if metrics.roc_auc:
        col_a, col_b = st.columns(2)
        col_a.metric("ROC-AUC", f"{metrics.roc_auc:.3f}")
        if metrics.pr_auc:
            col_b.metric("PR-AUC", f"{metrics.pr_auc:.3f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = metrics.confusion_matrix
    cm_df = pd.DataFrame(
        [[cm.tp, cm.fn], [cm.fp, cm.tn]],
        index=["True AI", "True Human"],
        columns=["Pred AI", "Pred Human"],
    )
    fig_cm = go.Figure(data=go.Heatmap(
        z=[[cm.tp, cm.fn], [cm.fp, cm.tn]],
        x=["Pred AI", "Pred Human"],
        y=["True AI", "True Human"],
        text=[[str(cm.tp), str(cm.fn)], [str(cm.fp), str(cm.tn)]],
        texttemplate="%{text}",
        colorscale="Blues",
    ))
    fig_cm.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_cm, width="stretch")

    # ── ROC + PR Curves ───────────────────────────────────────────────────────
    if "score_ai" in eval_df.columns:
        col_roc, col_pr = st.columns(2)

        fpr, tpr, _ = compute_roc_curve(eval_df)
        if fpr:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
            fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=350)
            col_roc.plotly_chart(fig_roc, width="stretch")

        pre, rec, _ = compute_pr_curve(eval_df)
        if pre:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=rec, y=pre, mode="lines", name="PR"))
            fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", height=350)
            col_pr.plotly_chart(fig_pr, width="stretch")

    # ── Stratified Metrics ────────────────────────────────────────────────────
    st.subheader("Stratified Metrics")
    strat_cols = [c for c in eval_df.columns if c.startswith("cluster") or c in ["theme_primary", "stance"]]

    # Merge enrichments if available
    enrichment_runs = [r for r in (manifest.plugin_runs if manifest else []) if r.plugin_kind.value == "enrichment"]
    if enrichment_runs:
        latest_enrich_path = enrichments_dir(project_id) / f"{enrichment_runs[-1].run_id}_output.parquet"
        if latest_enrich_path.exists():
            enrich_df = read_table(latest_enrich_path)
            eval_df = eval_df.merge(enrich_df[["comment_id", "theme_primary", "stance"]], on="comment_id", how="left")
            strat_cols += ["theme_primary", "stance"]

    strat_col = st.selectbox("Stratify by", ["(none)"] + list(set(strat_cols)), key="strat_col")
    if strat_col != "(none)" and strat_col in eval_df.columns:
        strat_df = stratified_metrics(eval_df, group_col=strat_col, uncertain_handling=uncertain_handling)
        if not strat_df.empty:
            st.dataframe(strat_df.sort_values("f1_ai", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data for stratified metrics.")
