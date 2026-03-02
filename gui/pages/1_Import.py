"""
Page 1 — Import

Create a new project by uploading a comment CSV.
Maps Document ID → comment_id and Comment → comment_text.
Validates, normalizes, and saves project artifacts.
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
from core.ingest.csv_loader import load_csv_bytes
from core.ingest.normalizer import normalize_dataframe
from core.ingest.schema_mapper import detect_mapping
from core.ingest.validators import validate_and_summarize
from core.storage.paths import normalized_comments_path, raw_dir
from core.storage.project_store import create_project, delete_project, list_projects, update_manifest
from core.storage.tables import write_table
from gui.utils.state import (
    CURRENT_PROJECT_ID,
    CURRENT_PROJECT_MANIFEST,
    has_project,
    init_defaults,
    set_project,
)

init_defaults()

st.title("📥 Import")
st.markdown("Upload a public comment CSV to create a new analysis project.")

# ── Sidebar — project status banner ──────────────────────────────────────────

with st.sidebar:
    if has_project():
        manifest = st.session_state.get(CURRENT_PROJECT_MANIFEST)
        if manifest:
            st.success(f"📁 **{manifest.project_name}**")
            st.caption(f"ID: `{manifest.project_id[:8]}…`")
            if manifest.ingest_summary:
                s = manifest.ingest_summary
                st.caption(
                    f"{s.total_rows:,} comments · "
                    f"{s.rows_missing_comment} missing"
                )
    else:
        st.info("No project loaded.")

# ── Existing projects ─────────────────────────────────────────────────────────

with st.expander("📂 Open existing project", expanded=not has_project()):
    projects = list_projects()
    if not projects:
        st.info("No projects found. Create one below.")
    else:
        options = {f"{m.project_name} ({m.project_id[:8]}…)": m for m in projects}
        choice = st.selectbox("Select project", list(options.keys()), key="existing_project_select")
        if st.button("Open selected project", key="btn_open_project"):
            manifest = options[choice]
            set_project(manifest)
            st.toast(f"✅ Opened: **{manifest.project_name}**", icon="📁")
            st.rerun()

        st.divider()
        st.markdown("**🗑️ Delete a project**")
        st.caption("Permanently removes the project directory and all associated data. This cannot be undone.")

        delete_options = {f"{m.project_name} ({m.project_id[:8]}…)": m for m in projects}
        delete_choice = st.selectbox(
            "Project to delete",
            list(delete_options.keys()),
            key="delete_project_select",
        )
        confirm_delete = st.checkbox(
            f"Yes, permanently delete this project and all its data",
            key="confirm_delete_checkbox",
        )
        if st.button("🗑️ Delete Project", type="secondary", key="btn_delete_project", disabled=not confirm_delete):
            to_delete = delete_options[delete_choice]
            deleted_name = to_delete.project_name
            deleted_id = to_delete.project_id
            try:
                delete_project(deleted_id)
                # If the deleted project was the active one, clear session state
                if st.session_state.get(CURRENT_PROJECT_ID) == deleted_id:
                    st.session_state[CURRENT_PROJECT_ID] = None
                    st.session_state[CURRENT_PROJECT_MANIFEST] = None
                st.success(f"Project **{deleted_name}** has been deleted.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to delete project: {e}")

st.divider()

# ── New project creation ──────────────────────────────────────────────────────

st.subheader("Create new project")

uploaded_file = st.file_uploader(
    "Upload comment CSV",
    type=["csv"],
    help="Expected columns: Document ID, Comment (plus optional metadata columns)",
    key="csv_uploader",
)

if uploaded_file is not None:
    # Load raw CSV
    try:
        raw_bytes = uploaded_file.read()
        df_raw, fingerprint = load_csv_bytes(raw_bytes, filename=uploaded_file.name)
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        st.stop()

    st.success(f"Loaded **{len(df_raw):,} rows** × {len(df_raw.columns)} columns from `{uploaded_file.name}`")

    # ── Column mapping ────────────────────────────────────────────────────────
    st.subheader("Column Mapping")
    st.caption("Auto-detected defaults. Override if your CSV uses different column names.")

    all_cols = list(df_raw.columns)

    col1, col2 = st.columns(2)
    with col1:
        # Try to auto-detect sensible defaults
        default_id = next((c for c in ["Document ID", "document_id", "id"] if c in all_cols), all_cols[0])
        id_col = st.selectbox("Document ID column", all_cols, index=all_cols.index(default_id), key="id_col_select")
    with col2:
        default_comment = next((c for c in ["Comment", "comment", "text", "body"] if c in all_cols), all_cols[-1])
        comment_col = st.selectbox(
            "Comment text column",
            all_cols,
            index=all_cols.index(default_comment),
            key="comment_col_select",
        )

    # ── Data preview ──────────────────────────────────────────────────────────
    st.subheader("Data Preview (first 10 rows)")
    preview_cols = [id_col, comment_col] + [
        c for c in all_cols if c not in (id_col, comment_col)
    ][:5]
    st.dataframe(
        df_raw[preview_cols].head(10),
        use_container_width=True,
        height=280,
    )

    # ── Project name ──────────────────────────────────────────────────────────
    st.subheader("Project Settings")
    default_name = Path(uploaded_file.name).stem.replace("_", " ").replace("-", " ")
    project_name = st.text_input("Project name", value=default_name, key="project_name_input")

    # ── Create project button ────────────────────────────────────────────────
    if st.button("🚀 Create Project & Ingest", type="primary", key="btn_create_project"):
        if not project_name.strip():
            st.error("Please enter a project name.")
            st.stop()

        with st.spinner("Detecting column mapping…"):
            try:
                mapping = detect_mapping(df_raw, id_col_override=id_col, comment_col_override=comment_col)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        with st.spinner("Normalizing comments…"):
            _records, normalized_df = normalize_dataframe(df_raw, mapping)

        with st.spinner("Validating data…"):
            summary = validate_and_summarize(normalized_df, fingerprint)

        with st.spinner("Creating project workspace…"):
            manifest = create_project(
                project_name=project_name.strip(),
                source_filename=uploaded_file.name,
                dataset_fingerprint=fingerprint,
            )

            # Save raw CSV
            raw_path = raw_dir(manifest.project_id) / uploaded_file.name
            raw_path.write_bytes(raw_bytes)

            # Save normalized table
            write_table(normalized_df, normalized_comments_path(manifest.project_id))

            # Save ingest summary to manifest
            manifest.ingest_summary = summary
            update_manifest(manifest)

        # Activate project in session
        set_project(manifest)

        # ── Ingest summary report ─────────────────────────────────────────
        st.success(f"✅ Project **{project_name}** created successfully!")
        st.subheader("Ingest Summary")

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Rows", f"{summary.total_rows:,}")
        col_b.metric("With Comment", f"{summary.rows_with_comment:,}")
        col_c.metric("Missing Comment", f"{summary.rows_missing_comment:,}")
        col_d.metric("Duplicate IDs", f"{summary.duplicate_comment_ids:,}")

        col_e, col_f = st.columns(2)
        col_e.metric("Duplicate Text Hashes", f"{summary.duplicate_text_hashes:,}")
        col_f.metric("Dataset Fingerprint", summary.dataset_fingerprint[:16] + "…")

        # Length distribution
        st.subheader("Comment Length Distribution (characters)")
        dist = summary.length_distribution
        col_g, col_h, col_i, col_j, col_k = st.columns(5)
        col_g.metric("Min", f"{dist.min:,}")
        col_h.metric("P25", f"{dist.p25:,.0f}")
        col_i.metric("Median", f"{dist.median:,.0f}")
        col_j.metric("P75", f"{dist.p75:,.0f}")
        col_k.metric("Max", f"{dist.max:,}")

        # ── Duplicate text hash groups ────────────────────────────────────
        if summary.duplicate_text_hashes > 0:
            st.subheader("🔁 Duplicate Text Hash Groups")
            st.caption(
                "Each row below is a group of Document IDs that share identical comment text "
                "(after stripping whitespace and lowercasing). These are likely copy-paste submissions."
            )
            dup_groups = (
                normalized_df[normalized_df["text_hash"].duplicated(keep=False)]
                .groupby("text_hash")["comment_id"]
                .apply(list)
                .reset_index(drop=True)
            )
            dup_df = pd.DataFrame(
                {
                    "Group": range(1, len(dup_groups) + 1),
                    "Count": dup_groups.apply(len),
                    "Document IDs": dup_groups.apply(lambda ids: ", ".join(str(i) for i in ids)),
                }
            )
            st.dataframe(
                dup_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Group": st.column_config.NumberColumn("Group", width="small"),
                    "Count": st.column_config.NumberColumn("Count", width="small"),
                    "Document IDs": st.column_config.TextColumn("Document IDs"),
                },
            )

        # Warnings / errors
        if summary.errors:
            for err in summary.errors:
                st.error(f"❌ {err}")
        if summary.warnings:
            for warn in summary.warnings:
                st.warning(f"⚠️ {warn}")

        if not summary.errors and not summary.warnings:
            st.info("No data quality issues detected.")

        st.balloons()
