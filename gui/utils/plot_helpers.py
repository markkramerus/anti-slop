"""
Plotly scatter plot helpers for 2D and 3D semantic maps.

Builds interactive charts with per-point hover, click, and color-by support.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Color palettes ────────────────────────────────────────────────────────────

LABEL_COLOR_MAP = {
    "human": "#28a745",
    "ai_generated": "#dc3545",
    "uncertain": "#ffc107",
    "unlabeled": "#adb5bd",
    None: "#adb5bd",
}

CLUSTER_COLORSCALE = px.colors.qualitative.Set3


def _truncate(text: str, n: int = 120) -> str:
    if not text:
        return ""
    return text[:n] + "…" if len(text) > n else text


# ── Build plot DataFrame ──────────────────────────────────────────────────────

def build_plot_df(
    coords_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    annotations_df: pd.DataFrame | None = None,
    clusters_df: pd.DataFrame | None = None,
    predictions_df: pd.DataFrame | None = None,
    enrichments_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge all data sources into a single plot DataFrame.

    All merges are left joins on comment_id.
    """
    df = coords_df.copy()

    # Merge comment text
    text_cols = ["comment_id", "comment_text"] + [
        c for c in comments_df.columns if c.startswith("meta_")
    ]
    df = df.merge(comments_df[text_cols], on="comment_id", how="left")

    # Manual labels
    if annotations_df is not None and not annotations_df.empty:
        ann_cols = ["comment_id", "label"]
        if "note" in annotations_df.columns:
            ann_cols.append("note")
        df = df.merge(annotations_df[ann_cols], on="comment_id", how="left")
        df["manual_label"] = df["label"].fillna("unlabeled")
        df.drop(columns=["label"], errors="ignore", inplace=True)
    else:
        df["manual_label"] = "unlabeled"

    # Cluster assignments
    if clusters_df is not None and not clusters_df.empty:
        df = df.merge(clusters_df[["comment_id", "cluster_id"]], on="comment_id", how="left")
        df["cluster_id"] = df["cluster_id"].fillna(-1).astype(int)
        df["cluster_label"] = df["cluster_id"].apply(
            lambda x: f"Cluster {x}" if x >= 0 else "Noise"
        )
    else:
        df["cluster_id"] = -1
        df["cluster_label"] = "No clustering"

    # Classifier predictions
    if predictions_df is not None and not predictions_df.empty:
        pred_cols = [c for c in ["comment_id", "pred_label", "score_ai", "confidence"]
                     if c in predictions_df.columns]
        df = df.merge(predictions_df[pred_cols], on="comment_id", how="left")

    # Enrichments (selected columns only)
    if enrichments_df is not None and not enrichments_df.empty:
        enrich_cols = [c for c in enrichments_df.columns if c != "comment_id"]
        df = df.merge(enrichments_df, on="comment_id", how="left")

    # Hover text
    df["hover_text"] = df.apply(
        lambda r: (
            f"<b>{r['comment_id']}</b><br>"
            f"Label: {r.get('manual_label', '?')}<br>"
            f"Cluster: {r.get('cluster_label', '?')}<br><br>"
            f"{_truncate(str(r.get('comment_text', '')))}"
        ),
        axis=1,
    )

    return df


# ── 2D scatter ────────────────────────────────────────────────────────────────

def scatter_2d(
    df: pd.DataFrame,
    color_by: str = "manual_label",
    title: str = "Semantic Map",
    height: int = 650,
) -> go.Figure:
    """Build a 2D Plotly scatter plot."""
    if color_by not in df.columns:
        color_by = "manual_label"

    # Determine color settings
    is_categorical = df[color_by].dtype == object or df[color_by].nunique() < 20

    if color_by == "manual_label" and is_categorical:
        unique_labels = df[color_by].dropna().unique().tolist()
        color_map = {lbl: LABEL_COLOR_MAP.get(lbl, "#888") for lbl in unique_labels}
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color=color_by,
            color_discrete_map=color_map,
            hover_data={"x": False, "y": False, "comment_id": True},
            custom_data=["comment_id"],
            title=title,
            labels={color_by: color_by.replace("_", " ").title()},
        )
    elif is_categorical:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color=color_by,
            color_discrete_sequence=CLUSTER_COLORSCALE,
            hover_data={"x": False, "y": False, "comment_id": True},
            custom_data=["comment_id"],
            title=title,
            labels={color_by: color_by.replace("_", " ").title()},
        )
    else:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color=color_by,
            color_continuous_scale="RdYlGn_r",
            hover_data={"x": False, "y": False, "comment_id": True},
            custom_data=["comment_id"],
            title=title,
            labels={color_by: color_by.replace("_", " ").title()},
        )

    # Hover template
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Click to view comment<extra></extra>"
        ),
        marker=dict(size=6, opacity=0.8),
    )

    fig.update_layout(
        height=height,
        xaxis=dict(showticklabels=False, title=""),
        yaxis=dict(showticklabels=False, title=""),
        legend=dict(itemsizing="constant"),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="#fafafa",
    )

    return fig


# ── 3D scatter ────────────────────────────────────────────────────────────────

def scatter_3d(
    df: pd.DataFrame,
    color_by: str = "manual_label",
    title: str = "Semantic Map (3D)",
    height: int = 700,
) -> go.Figure:
    """Build a 3D Plotly scatter plot."""
    if "z" not in df.columns:
        raise ValueError("DataFrame missing 'z' column for 3D scatter")
    if color_by not in df.columns:
        color_by = "manual_label"

    is_categorical = df[color_by].dtype == object or df[color_by].nunique() < 20

    if color_by == "manual_label":
        unique_labels = df[color_by].dropna().unique().tolist()
        color_map = {lbl: LABEL_COLOR_MAP.get(lbl, "#888") for lbl in unique_labels}
        colors = df[color_by].map(color_map).fillna("#888")
    elif is_categorical:
        # Assign colors from palette by index
        unique_vals = df[color_by].dropna().unique().tolist()
        palette = CLUSTER_COLORSCALE
        val_color_map = {v: palette[i % len(palette)] for i, v in enumerate(unique_vals)}
        colors = df[color_by].map(val_color_map).fillna("#888")
    else:
        colors = df[color_by]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df["x"],
                y=df["y"],
                z=df["z"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=colors,
                    opacity=0.75,
                    colorscale="RdYlGn_r" if not is_categorical else None,
                ),
                customdata=df["comment_id"].tolist(),
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    "Click to view comment<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        height=height,
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


# ── Color-by options ──────────────────────────────────────────────────────────

def get_color_by_options(df: pd.DataFrame) -> list[str]:
    """
    Return candidate columns for the color-by selector.
    Includes: manual_label, cluster_label, pred_label, score_ai, confidence,
    plus any derived feature columns (theme_*, stance, etc.).
    """
    core_options = []
    for col in ["manual_label", "cluster_label", "pred_label", "score_ai", "confidence"]:
        if col in df.columns:
            core_options.append(col)

    # Derived enrichment columns (not coordinate/text/hash cols)
    skip = {
        "x", "y", "z", "comment_id", "comment_text", "text_hash",
        "ingest_status", "ingest_warning", "source_row_index",
        "hover_text", "manual_label", "cluster_id", "cluster_label",
        "pred_label", "score_ai", "confidence",
    }
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    derived_cols = [
        c for c in df.columns
        if c not in skip and c not in meta_cols
        and not c.startswith("emb_")
    ]

    return core_options + derived_cols
