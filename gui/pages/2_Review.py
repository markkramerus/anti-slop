"""
Page 2 — Review

Manual read-and-label workflow. One comment at a time.
Supports keyboard shortcuts (h/a/u to label, j/k or arrows to navigate).
Autosaves labels and notes on every interaction.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from core.annotations.label_store import (
    annotated_comment_ids,
    delete_annotation,
    get_annotation,
    label_counts,
    save_annotation,
)
from core.storage.paths import (
    annotations_path,
    classifier_runs_dir,
    enrichments_dir,
    normalized_comments_path,
)
from core.storage.tables import read_table
from gui.utils.state import (
    get_project_id,
    get_project_manifest,
    get_review_filters,
    get_review_index,
    get_review_queue,
    init_defaults,
    pop_open_in_review,
    set_review_filters,
    set_review_index,
    set_review_queue,
)
from gui.utils.ui_components import (
    comment_card,
    derived_features_panel,
    keyboard_shortcuts_js,
    label_badge,
    label_buttons,
    metadata_panel,
    predictions_panel,
    require_project,
)
from shared_models import ManualAnnotation, ManualLabel

init_defaults()

st.title("🔍 Review")

if not require_project():
    st.stop()

project_id = get_project_id()
manifest = get_project_manifest()

# ── Load comments ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_comments(pid: str) -> pd.DataFrame:
    return read_table(normalized_comments_path(pid))


@st.cache_data(ttl=10)
def load_enrichments_for_comment(pid: str, comment_id: str) -> dict:
    """Load all enrichment outputs for a single comment, keyed by run name."""
    features = {}
    manifest_local = get_project_manifest()
    if not manifest_local:
        return features
    for run in manifest_local.plugin_runs:
        if run.plugin_kind.value != "enrichment":
            continue
        path = enrichments_dir(pid) / f"{run.run_id}_output.parquet"
        if not path.exists():
            continue
        df = read_table(path)
        if df.empty or "comment_id" not in df.columns:
            continue
        row = df[df["comment_id"] == comment_id]
        if row.empty:
            continue
        row_dict = row.iloc[0].to_dict()
        row_dict.pop("comment_id", None)
        run_label = f"{run.plugin_name} v{run.plugin_version}"
        features[run_label] = row_dict
    return features


@st.cache_data(ttl=10)
def load_predictions_for_comment(pid: str, comment_id: str) -> dict:
    """Load all classifier predictions for a single comment, keyed by run name."""
    preds = {}
    manifest_local = get_project_manifest()
    if not manifest_local:
        return preds
    for run in manifest_local.plugin_runs:
        if run.plugin_kind.value != "classifier":
            continue
        path = classifier_runs_dir(pid) / f"{run.run_id}_predictions.parquet"
        if not path.exists():
            continue
        df = read_table(path)
        if df.empty or "comment_id" not in df.columns:
            continue
        row = df[df["comment_id"] == comment_id]
        if row.empty:
            continue
        run_label = f"{run.plugin_name} v{run.plugin_version}"
        preds[run_label] = row.iloc[0].to_dict()
    return preds


df_all = load_comments(project_id)
if df_all.empty:
    st.error("No comments found. Did the import complete successfully?")
    st.stop()

# ── Sidebar — Filters ─────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("🔎 Filter Queue")

    labeled_ids = annotated_comment_ids(project_id)
    counts = label_counts(project_id)
    total = len(df_all)
    labeled_count = len(labeled_ids)

    st.caption(f"Labeled: **{labeled_count}/{total}** ({100*labeled_count//max(total,1)}%)")
    st.progress(labeled_count / max(total, 1))

    # Label counts
    for lbl, cnt in counts.items():
        st.caption(f"{label_badge(lbl)}: {cnt}")

    st.divider()

    label_filter = st.selectbox(
        "Show comments",
        ["All", "Unlabeled", "Human", "AI Generated", "Uncertain"],
        key="review_label_filter",
    )

    keyword = st.text_input("Keyword search (comment text)", key="review_keyword")

    # Metadata filters
    meta_cols = [c.replace("meta_", "") for c in df_all.columns if c.startswith("meta_")]
    useful_meta = ["state_province", "organization_name", "category", "city", "country"]
    filter_meta_col = st.selectbox(
        "Filter by metadata",
        ["(none)"] + [c for c in useful_meta if c in meta_cols],
        key="review_meta_col_select",
    )
    filter_meta_val = ""
    if filter_meta_col != "(none)":
        col_vals = sorted(df_all[f"meta_{filter_meta_col}"].dropna().unique().tolist())
        filter_meta_val = st.selectbox(
            f"{filter_meta_col} value",
            ["(all)"] + col_vals,
            key="review_meta_val_select",
        )

    st.divider()
    st.caption("**Keyboard shortcuts:**")
    st.caption("h = Human · a = AI · u = Uncertain")
    st.caption("j / → = Next · k / ← = Prev")

# ── Build filtered queue ──────────────────────────────────────────────────────

def build_queue(df: pd.DataFrame, annotations_df: pd.DataFrame) -> list[str]:
    ids = df["comment_id"].tolist()

    # Check if a specific comment was requested (from map page)
    jump_to = pop_open_in_review()
    if jump_to and jump_to in ids:
        idx = ids.index(jump_to)
        set_review_index(idx)

    # Apply label filter
    if label_filter != "All":
        lbl_map = {
            "Unlabeled": None,
            "Human": "human",
            "AI Generated": "ai_generated",
            "Uncertain": "uncertain",
        }
        target = lbl_map[label_filter]
        if target is None:
            ids = [cid for cid in ids if cid not in labeled_ids]
        else:
            ann_label_map = {}
            if not annotations_df.empty:
                ann_label_map = dict(zip(annotations_df["comment_id"], annotations_df["label"]))
            ids = [cid for cid in ids if ann_label_map.get(cid) == target]

    # Apply keyword filter
    if keyword.strip():
        kw_lower = keyword.lower()
        text_map = dict(zip(df["comment_id"], df["comment_text"].fillna("")))
        ids = [cid for cid in ids if kw_lower in text_map.get(cid, "").lower()]

    # Apply metadata filter
    if filter_meta_col != "(none)" and filter_meta_val and filter_meta_val != "(all)":
        meta_col_name = f"meta_{filter_meta_col}"
        if meta_col_name in df.columns:
            meta_map = dict(zip(df["comment_id"], df[meta_col_name].fillna("")))
            ids = [cid for cid in ids if meta_map.get(cid, "") == filter_meta_val]

    return ids


annotations_df = read_table(annotations_path(project_id))
queue = build_queue(df_all, annotations_df)
set_review_queue(queue)

if not queue:
    st.info("No comments match the current filters.")
    st.stop()

# ── Navigation ────────────────────────────────────────────────────────────────

current_idx = get_review_index()
current_idx = max(0, min(current_idx, len(queue) - 1))
set_review_index(current_idx)

comment_id = queue[current_idx]

# Navigation bar — row 1: prev/next/number
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 3, 1])
with nav_col1:
    if st.button("◀ Prev", key="btn_prev", disabled=current_idx == 0, use_container_width=True):
        set_review_index(current_idx - 1)
        st.rerun()
with nav_col2:
    if st.button("▶ Next", key="btn_next", disabled=current_idx >= len(queue) - 1, use_container_width=True):
        set_review_index(current_idx + 1)
        st.rerun()
with nav_col3:
    jump_idx = st.number_input(
        "Jump to (#)",
        min_value=1,
        max_value=len(queue),
        value=current_idx + 1,
        step=1,
        key="jump_input",
        label_visibility="collapsed",
    )
    if jump_idx - 1 != current_idx:
        set_review_index(int(jump_idx) - 1)
        st.rerun()
with nav_col4:
    st.caption(f"{current_idx + 1} / {len(queue)}")

# Navigation bar — row 2: Document ID (editable, synced with prev/next)
id_label_col, id_input_col = st.columns([1, 5])
with id_label_col:
    st.caption("Document ID:")
with id_input_col:
    # Key includes current_idx so the widget resets to the correct ID after every navigation
    typed_id = st.text_input(
        "Document ID",
        value=comment_id,
        key=f"jump_id_{current_idx}",
        label_visibility="collapsed",
        placeholder="Enter Document ID to jump…",
    )
    if typed_id != comment_id:
        if typed_id in queue:
            set_review_index(queue.index(typed_id))
            st.rerun()
        elif typed_id in df_all["comment_id"].values:
            st.warning(
                f"Comment `{typed_id}` exists but is not in the current filtered queue. "
                "Clear filters to navigate to it."
            )
        elif typed_id.strip():
            st.error(f"Document ID `{typed_id}` not found.")

# ── Main review area ──────────────────────────────────────────────────────────

# Get comment data
comment_row = df_all[df_all["comment_id"] == comment_id]
if comment_row.empty:
    st.error(f"Comment {comment_id!r} not found in normalized data.")
    st.stop()

comment_row = comment_row.iloc[0]
comment_text = str(comment_row.get("comment_text", ""))

# Current annotation
existing_ann = get_annotation(project_id, comment_id)
current_label = existing_ann.label.value if existing_ann else None

# Render comment card
comment_card(
    comment_id=comment_id,
    comment_text=comment_text,
    current_label=current_label,
    index_display=f"#{current_idx + 1} of {len(queue)}",
)

st.divider()

# ── Label buttons ─────────────────────────────────────────────────────────────

st.subheader("Label this comment")
new_label = label_buttons(current_label, key_prefix=f"label_btn_{comment_id}")

if new_label is not None:
    if new_label == "__clear__":
        delete_annotation(project_id, comment_id)
        st.toast("Label cleared", icon="⬜")
    else:
        ann = ManualAnnotation(
            comment_id=comment_id,
            label=ManualLabel(new_label),
            note=existing_ann.note if existing_ann else None,
        )
        save_annotation(project_id, ann)
        st.toast(f"Saved: {label_badge(new_label)}", icon="✅")
    # Invalidate cache and auto-advance to next
    load_comments.clear()
    if new_label != "__clear__" and current_idx < len(queue) - 1:
        set_review_index(current_idx + 1)
    st.rerun()

# ── Notes ─────────────────────────────────────────────────────────────────────

note_val = existing_ann.note or "" if existing_ann else ""
new_note = st.text_area(
    "Analyst note (optional)",
    value=note_val,
    height=80,
    key=f"note_{comment_id}",
    help="Notes are autosaved when you move away or re-label",
    placeholder="Add observations, red flags, reasoning…",
)

if new_note != note_val:
    # Autosave note change
    label = ManualLabel(current_label) if current_label else ManualLabel.UNCERTAIN
    ann = ManualAnnotation(
        comment_id=comment_id,
        label=label,
        note=new_note if new_note.strip() else None,
        annotation_id=existing_ann.annotation_id if existing_ann else None,
    )
    save_annotation(project_id, ann)

st.divider()

# ── Metadata + derived features panels ───────────────────────────────────────

metadata_panel(comment_row, expanded=False)

enrichments = load_enrichments_for_comment(project_id, comment_id)
derived_features_panel(enrichments, expanded=bool(enrichments))

predictions = load_predictions_for_comment(project_id, comment_id)
predictions_panel(predictions, expanded=bool(predictions))

# ── Keyboard shortcuts ────────────────────────────────────────────────────────

keyboard_shortcuts_js()
