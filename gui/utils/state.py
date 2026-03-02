"""
Centralized Streamlit session state management.

All session state keys are defined here to prevent typos and make
dependencies explicit. Prefer using these helpers over direct st.session_state access.
"""

from __future__ import annotations

import streamlit as st

from shared_models import ProjectManifest

# ── Key constants ─────────────────────────────────────────────────────────────

CURRENT_PROJECT_ID = "current_project_id"
CURRENT_PROJECT_MANIFEST = "current_project_manifest"
REVIEW_QUEUE_IDS = "review_queue_ids"
REVIEW_CURRENT_INDEX = "review_current_index"
REVIEW_FILTERS = "review_filters"
MAP_SELECTED_COMMENT_IDS = "map_selected_comment_ids"
MAP_COLOR_BY = "map_color_by"
MAP_EMBEDDING_RUN_ID = "map_embedding_run_id"
MAP_PROJECTION_RUN_ID = "map_projection_run_id"
MAP_CLUSTER_RUN_ID = "map_cluster_run_id"
EVAL_TRUTH_KEY_DF = "eval_truth_key_df"
EVAL_TRUTH_KEY_FILENAME = "eval_truth_key_filename"
OPEN_IN_REVIEW_COMMENT_ID = "open_in_review_comment_id"


# ── Project state ─────────────────────────────────────────────────────────────

def set_project(manifest: ProjectManifest) -> None:
    st.session_state[CURRENT_PROJECT_ID] = manifest.project_id
    st.session_state[CURRENT_PROJECT_MANIFEST] = manifest


def get_project_id() -> str | None:
    return st.session_state.get(CURRENT_PROJECT_ID)


def get_project_manifest() -> ProjectManifest | None:
    return st.session_state.get(CURRENT_PROJECT_MANIFEST)


def has_project() -> bool:
    return bool(get_project_id())


def refresh_manifest() -> ProjectManifest | None:
    """Reload manifest from disk and update session state."""
    pid = get_project_id()
    if not pid:
        return None
    from core.storage.project_store import get_project
    manifest = get_project(pid)
    if manifest:
        st.session_state[CURRENT_PROJECT_MANIFEST] = manifest
    return manifest


# ── Review queue state ────────────────────────────────────────────────────────

def get_review_index() -> int:
    return int(st.session_state.get(REVIEW_CURRENT_INDEX, 0))


def set_review_index(idx: int) -> None:
    st.session_state[REVIEW_CURRENT_INDEX] = idx


def get_review_queue() -> list[str]:
    return st.session_state.get(REVIEW_QUEUE_IDS, [])


def set_review_queue(ids: list[str]) -> None:
    st.session_state[REVIEW_QUEUE_IDS] = ids


def get_review_filters() -> dict:
    return st.session_state.get(REVIEW_FILTERS, {})


def set_review_filters(filters: dict) -> None:
    st.session_state[REVIEW_FILTERS] = filters


# ── Map state ─────────────────────────────────────────────────────────────────

def get_map_selected_ids() -> list[str]:
    return st.session_state.get(MAP_SELECTED_COMMENT_IDS, [])


def set_map_selected_ids(ids: list[str]) -> None:
    st.session_state[MAP_SELECTED_COMMENT_IDS] = ids


def set_open_in_review(comment_id: str) -> None:
    """Signal the Review page to open a specific comment."""
    st.session_state[OPEN_IN_REVIEW_COMMENT_ID] = comment_id


def pop_open_in_review() -> str | None:
    return st.session_state.pop(OPEN_IN_REVIEW_COMMENT_ID, None)


# ── Evaluation state ──────────────────────────────────────────────────────────

def set_truth_key(df, filename: str) -> None:
    st.session_state[EVAL_TRUTH_KEY_DF] = df
    st.session_state[EVAL_TRUTH_KEY_FILENAME] = filename


def get_truth_key():
    return st.session_state.get(EVAL_TRUTH_KEY_DF)


def get_truth_key_filename() -> str | None:
    return st.session_state.get(EVAL_TRUTH_KEY_FILENAME)


# ── Utility ───────────────────────────────────────────────────────────────────

def init_defaults() -> None:
    """Ensure all required keys exist with sensible defaults."""
    defaults = {
        CURRENT_PROJECT_ID: None,
        CURRENT_PROJECT_MANIFEST: None,
        REVIEW_CURRENT_INDEX: 0,
        REVIEW_QUEUE_IDS: [],
        REVIEW_FILTERS: {},
        MAP_SELECTED_COMMENT_IDS: [],
        MAP_COLOR_BY: "manual_label",
        MAP_EMBEDDING_RUN_ID: None,
        MAP_PROJECTION_RUN_ID: None,
        MAP_CLUSTER_RUN_ID: None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
