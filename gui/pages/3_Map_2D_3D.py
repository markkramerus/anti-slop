"""
Page 3 — Map 2D/3D

Semantic visualization of comments.
Supports embedding generation, PCA/UMAP/t-SNE projection, clustering,
and color-by any label, score, cluster, or derived feature column.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st
from streamlit_plotly_events import plotly_events  # type: ignore[import]

import config
from core.features.embeddings import generate_embeddings, load_embedding_vectors
from core.projection.clustering import load_cluster_assignments, run_clustering
from core.projection.reducers import load_projection_coords, run_projection
from core.storage.paths import (
    annotations_path,
    classifier_runs_dir,
    enrichments_dir,
    normalized_comments_path,
)
from core.storage.project_store import update_manifest
from core.storage.tables import read_table
from gui.utils.plot_helpers import build_plot_df, get_color_by_options, scatter_2d, scatter_3d
from gui.utils.state import (
    get_project_id,
    get_project_manifest,
    init_defaults,
    refresh_manifest,
    set_open_in_review,
)
from gui.utils.ui_components import require_project

init_defaults()

st.title("🗺️ Map 2D/3D")
st.markdown("Explore semantic structure, clusters, and patterns across your comment corpus.")

if not require_project():
    st.stop()

project_id = get_project_id()
manifest = get_project_manifest()

# ── Load base data ────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_base(pid: str) -> pd.DataFrame:
    return read_table(normalized_comments_path(pid))


@st.cache_data(ttl=30)
def load_annotations_cached(pid: str) -> pd.DataFrame:
    return read_table(annotations_path(pid))


comments_df = load_base(project_id)
annotations_df = load_annotations_cached(project_id)

# ── Sidebar Controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("⚙️ Map Controls")

    # ── Step 1: Embeddings ───────────────────────────────────────────────────
    st.markdown("**1. Embeddings**")
    embedding_runs = manifest.embedding_runs if manifest else []

    emb_options = {f"{r.model_name} [{r.run_id[:8]}]": r for r in embedding_runs}
    emb_names = list(emb_options.keys())

    if emb_names:
        selected_emb_name = st.selectbox("Embedding run", emb_names, key="map_emb_select")
        selected_emb_run = emb_options[selected_emb_name]
    else:
        selected_emb_run = None
        st.info("No embeddings yet — generate below.")

    with st.expander("Generate new embeddings"):
        emb_provider = st.selectbox("Provider", ["local", "openai"], key="emb_provider_select")
        if emb_provider == "local":
            emb_model = st.text_input("Model", value=config.DEFAULT_EMBEDDING_MODEL, key="emb_model_input")
        else:
            emb_model = st.text_input("Model", value="text-embedding-3-small", key="emb_model_input_oa")
        emb_batch = st.number_input("Batch size", value=config.DEFAULT_EMBEDDING_BATCH_SIZE, min_value=1, max_value=512, key="emb_batch")

        if st.button("▶ Generate Embeddings", key="btn_gen_emb"):
            progress_bar = st.progress(0.0)
            log_area = st.empty()
            logs = []
            try:
                gen = generate_embeddings(
                    project_id=project_id,
                    comments_df=comments_df,
                    provider=emb_provider,
                    model_name=emb_model,
                    batch_size=int(emb_batch),
                )
                emb_manifest = None
                while True:
                    try:
                        frac, msg = next(gen)
                        progress_bar.progress(min(frac, 1.0))
                        logs.append(msg)
                        log_area.code("\n".join(logs[-15:]))
                    except StopIteration as e:
                        emb_manifest = e.value
                        break
                if emb_manifest:
                    manifest.embedding_runs.append(emb_manifest)
                    update_manifest(manifest)
                    refresh_manifest()
                    st.success(f"✓ Embeddings done ({emb_manifest.row_count} vectors)")
                    load_base.clear()
                    st.rerun()
            except Exception as ex:
                st.error(f"Embedding error: {ex}")

    st.divider()

    # ── Step 2: Projection ───────────────────────────────────────────────────
    st.markdown("**2. Projection**")
    projection_runs = manifest.projection_runs if manifest else []

    proj_options = {
        f"{r.method.value.upper()} {r.dims}D [{r.run_id[:8]}]": r
        for r in projection_runs
    }
    proj_names = list(proj_options.keys())

    if proj_names:
        selected_proj_name = st.selectbox("Projection run", proj_names, key="map_proj_select")
        selected_proj_run = proj_options[selected_proj_name]
    else:
        selected_proj_run = None
        st.info("No projections yet — generate below.")

    with st.expander("Generate new projection"):
        if not selected_emb_run:
            st.warning("Generate embeddings first.")
        else:
            proj_method = st.selectbox("Method", ["umap", "pca", "tsne"], key="proj_method")
            proj_dims = st.selectbox("Dimensions", [2, 3], key="proj_dims")
            proj_seed = st.number_input("Random seed", value=config.DEFAULT_RANDOM_SEED, key="proj_seed")

            proj_params: dict = {"random_state": int(proj_seed)}
            if proj_method == "umap":
                proj_params["n_neighbors"] = st.slider("n_neighbors", 5, 50, config.DEFAULT_UMAP_N_NEIGHBORS, key="umap_nn")
                proj_params["min_dist"] = st.slider("min_dist", 0.01, 0.99, config.DEFAULT_UMAP_MIN_DIST, key="umap_md")
            elif proj_method == "tsne":
                proj_params["perplexity"] = st.slider("perplexity", 5, 100, 30, key="tsne_perp")

            if st.button("▶ Run Projection", key="btn_run_proj"):
                progress_bar2 = st.progress(0.0)
                log_area2 = st.empty()
                logs2 = []
                try:
                    cids, matrix = load_embedding_vectors(project_id, selected_emb_run.run_id)
                    gen = run_projection(
                        project_id=project_id,
                        embedding_run_id=selected_emb_run.run_id,
                        comment_ids=cids,
                        matrix=matrix,
                        method=proj_method,
                        dims=int(proj_dims),
                        params=proj_params,
                    )
                    proj_manifest_result = None
                    while True:
                        try:
                            frac, msg = next(gen)
                            progress_bar2.progress(min(frac, 1.0))
                            logs2.append(msg)
                            log_area2.code("\n".join(logs2[-10:]))
                        except StopIteration as e:
                            proj_manifest_result = e.value
                            break
                    if proj_manifest_result:
                        manifest.projection_runs.append(proj_manifest_result)
                        update_manifest(manifest)
                        refresh_manifest()
                        st.success(f"✓ Projection done")
                        st.rerun()
                except Exception as ex:
                    st.error(f"Projection error: {ex}")

    st.divider()

    # ── Step 3: Clustering ───────────────────────────────────────────────────
    st.markdown("**3. Clustering**")
    cluster_runs = manifest.cluster_runs if manifest else []

    clust_options = {
        f"{r.method.value.upper()} [{r.run_id[:8]}] ({r.n_clusters} clusters)": r
        for r in cluster_runs
    }
    clust_names = ["(none)"] + list(clust_options.keys())

    selected_clust_name = st.selectbox("Cluster run (optional)", clust_names, key="map_clust_select")
    selected_clust_run = clust_options.get(selected_clust_name) if selected_clust_name != "(none)" else None

    with st.expander("Run clustering"):
        if not selected_proj_run:
            st.warning("Run projection first.")
        else:
            clust_method = st.selectbox("Method", ["hdbscan", "kmeans", "agglomerative"], key="clust_method")
            clust_params: dict = {}
            if clust_method == "hdbscan":
                clust_params["min_cluster_size"] = st.slider(
                    "min_cluster_size", 3, 30, config.DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE, key="hdbscan_mcs"
                )
            else:
                clust_params["n_clusters"] = st.slider(
                    "n_clusters", 2, 30, config.DEFAULT_KMEANS_N_CLUSTERS, key="clust_k"
                )

            if st.button("▶ Run Clustering", key="btn_run_clust"):
                progress_bar3 = st.progress(0.0)
                log_area3 = st.empty()
                logs3 = []
                try:
                    coords_df = load_projection_coords(project_id, selected_proj_run.run_id)
                    gen = run_clustering(
                        project_id=project_id,
                        projection_run_id=selected_proj_run.run_id,
                        coords_df=coords_df,
                        method=clust_method,
                        params=clust_params,
                    )
                    clust_manifest_result = None
                    while True:
                        try:
                            frac, msg = next(gen)
                            progress_bar3.progress(min(frac, 1.0))
                            logs3.append(msg)
                            log_area3.code("\n".join(logs3[-10:]))
                        except StopIteration as e:
                            clust_manifest_result = e.value
                            break
                    if clust_manifest_result:
                        manifest.cluster_runs.append(clust_manifest_result)
                        update_manifest(manifest)
                        refresh_manifest()
                        st.success(f"✓ Clustering done")
                        st.rerun()
                except Exception as ex:
                    st.error(f"Clustering error: {ex}")

    st.divider()

    # ── Visualization options ────────────────────────────────────────────────
    st.markdown("**4. Display**")
    plot_dims = st.radio("Plot dimensions", ["2D", "3D"], horizontal=True, key="plot_dims_radio")

# ── Load enrichments for color-by ─────────────────────────────────────────────

def load_latest_enrichments(pid: str) -> pd.DataFrame | None:
    if not manifest:
        return None
    enrichment_runs = [r for r in manifest.plugin_runs if r.plugin_kind.value == "enrichment"]
    if not enrichment_runs:
        return None
    latest = enrichment_runs[-1]
    path = enrichments_dir(pid) / f"{latest.run_id}_output.parquet"
    df = read_table(path)
    return df if not df.empty else None


def load_latest_predictions(pid: str) -> pd.DataFrame | None:
    if not manifest:
        return None
    classifier_runs = [r for r in manifest.plugin_runs if r.plugin_kind.value == "classifier"]
    if not classifier_runs:
        return None
    latest = classifier_runs[-1]
    path = classifier_runs_dir(pid) / f"{latest.run_id}_predictions.parquet"
    df = read_table(path)
    return df if not df.empty else None


# ── Main map area ─────────────────────────────────────────────────────────────

if selected_proj_run is None:
    st.info(
        "👈 Use the sidebar to:\n"
        "1. Generate embeddings\n"
        "2. Run a projection\n"
        "3. (Optional) Run clustering\n\n"
        "Then the semantic map will appear here."
    )
    st.stop()

# Load projection coordinates
try:
    coords_df = load_projection_coords(project_id, selected_proj_run.run_id)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Load cluster assignments if selected
clusters_df = None
if selected_clust_run:
    clusters_df = load_cluster_assignments(project_id, selected_clust_run.run_id)

# Load enrichments + predictions for color-by
enrichments_df = load_latest_enrichments(project_id)
predictions_df = load_latest_predictions(project_id)

# Build merged plot DataFrame
plot_df = build_plot_df(
    coords_df=coords_df,
    comments_df=comments_df,
    annotations_df=annotations_df,
    clusters_df=clusters_df,
    predictions_df=predictions_df,
    enrichments_df=enrichments_df,
)

# Color-by selector
color_options = get_color_by_options(plot_df)
color_by = st.selectbox(
    "Color by",
    color_options,
    index=0,
    key="map_color_by_select",
)

# Info bar
col_a, col_b, col_c = st.columns(3)
col_a.metric("Points", f"{len(plot_df):,}")
if "cluster_id" in plot_df.columns:
    n_clusters = plot_df["cluster_id"].nunique()
    col_b.metric("Clusters", n_clusters)
if "manual_label" in plot_df.columns:
    labeled = (plot_df["manual_label"] != "unlabeled").sum()
    col_c.metric("Labeled", f"{labeled:,} / {len(plot_df):,}")

# Render the chart
try:
    is_3d = plot_dims == "3D" and "z" in coords_df.columns
    if is_3d:
        fig = scatter_3d(plot_df, color_by=color_by)
        selected_points = plotly_events(fig, click_event=True, hover_event=False, key="map_3d_events")
    else:
        if plot_dims == "3D" and "z" not in coords_df.columns:
            st.warning("This projection was run in 2D. Switch to 2D display or regenerate in 3D.")
        fig = scatter_2d(plot_df, color_by=color_by)
        selected_points = plotly_events(fig, click_event=True, hover_event=False, key="map_2d_events")
except ImportError:
    st.warning(
        "⚠️ `streamlit-plotly-events` not installed — click-to-review disabled. "
        "Install with: `pip install streamlit-plotly-events`"
    )
    if plot_dims == "3D" and "z" in coords_df.columns:
        fig = scatter_3d(plot_df, color_by=color_by)
    else:
        fig = scatter_2d(plot_df, color_by=color_by)
    st.plotly_chart(fig, use_container_width=True)
    selected_points = []

# ── Point-click → open comment ────────────────────────────────────────────────

if selected_points:
    try:
        point = selected_points[0]
        # plotly_events returns {x, y, curveNumber, pointNumber, pointIndex}
        point_idx = point.get("pointIndex", point.get("pointNumber", 0))
        clicked_id = plot_df.iloc[point_idx]["comment_id"]

        st.info(f"**Selected comment:** `{clicked_id}`")
        comment_row = plot_df[plot_df["comment_id"] == clicked_id]
        if not comment_row.empty:
            text = str(comment_row.iloc[0].get("comment_text", ""))
            label = str(comment_row.iloc[0].get("manual_label", "unlabeled"))
            cluster = str(comment_row.iloc[0].get("cluster_label", "?"))

            st.markdown(
                f"""
                **Label:** `{label}` · **Cluster:** `{cluster}`

                > {text[:600]}{"…" if len(text) > 600 else ""}
                """
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📖 Open in Review", key="btn_open_in_review"):
                    set_open_in_review(clicked_id)
                    st.switch_page("pages/2_Review.py")
    except Exception:
        pass

# ── Cluster summary table ─────────────────────────────────────────────────────

if selected_clust_run and clusters_df is not None and not clusters_df.empty:
    with st.expander("📊 Cluster Summary", expanded=False):
        cluster_summary = (
            clusters_df.merge(annotations_df[["comment_id", "label"]], on="comment_id", how="left")
            .groupby("cluster_id")
            .agg(
                count=("comment_id", "count"),
                human=("label", lambda x: (x == "human").sum()),
                ai_generated=("label", lambda x: (x == "ai_generated").sum()),
                uncertain=("label", lambda x: (x == "uncertain").sum()),
                unlabeled=("label", lambda x: x.isna().sum()),
            )
            .reset_index()
        )
        cluster_summary["cluster"] = cluster_summary["cluster_id"].apply(
            lambda x: f"Cluster {x}" if x >= 0 else "Noise"
        )
        cluster_summary = cluster_summary[["cluster", "count", "human", "ai_generated", "uncertain", "unlabeled"]]
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)
