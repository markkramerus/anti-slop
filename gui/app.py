"""
Anti-Slop Regulator Platform — Main Streamlit App

Run with:
    streamlit run gui/app.py
or:
    python run_gui.py
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path so all imports work regardless of CWD
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

import config
from gui.utils.state import init_defaults, has_project, get_project_manifest

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

init_defaults()

# ── Sidebar — project context banner ─────────────────────────────────────────

with st.sidebar:
    st.title(f"{config.APP_ICON} Anti-Slop")
    st.caption("Regulator Platform v0.2")
    st.divider()

    if has_project():
        manifest = get_project_manifest()
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
        st.info("No project loaded. Go to **Import** to create one.")

    st.divider()
    if config.DEV_MODE:
        st.caption("🧪 Dev mode (truth key enabled)")

# ── Home content ─────────────────────────────────────────────────────────────

st.title(f"{config.APP_ICON} Anti-Slop Regulator Platform")
st.markdown(
    """
    **A research tool for analysts inspecting public comment corpora for synthetic (AI-generated) submissions.**

    ---

    ### Getting Started

    1. **Import** — Upload your comment CSV and create a project
    2. **Review** — Manually label comments as human / AI / uncertain
    3. **Map 2D/3D** — Visualize semantic structure, find clusters
    4. **Enrichments & Detectors** — Run detection algorithms
    5. **Evaluation** — Benchmark against a truth key
    6. **Error Analysis** — Drill into false positives / negatives
    7. **Export** — Download labeled data, metrics, and research artifacts

    ---

    > ⚠️ This platform is for **research and review only** — it does not make
    > final enforcement decisions automatically.
    """
)

if not has_project():
    st.page_link("pages/1_Import.py", label="→ Go to Import to create a project", icon="📥")
