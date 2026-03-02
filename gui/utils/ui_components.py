"""
Reusable Streamlit UI components shared across pages.

Includes:
- Label button group with keyboard shortcut support
- Metadata panel
- Derived features panel (dynamic field rendering)
- Predictions panel
- Comment card
- Project guard (redirect if no project loaded)
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from shared_models import ManualLabel

# ── Label colors ──────────────────────────────────────────────────────────────

LABEL_COLORS = {
    "human": "🟢",
    "ai_generated": "🔴",
    "uncertain": "🟡",
    None: "⬜",
}

LABEL_DISPLAY = {
    "human": "Human",
    "ai_generated": "AI Generated",
    "uncertain": "Uncertain",
}


def label_badge(label: str | None) -> str:
    icon = LABEL_COLORS.get(label, "⬜")
    text = LABEL_DISPLAY.get(label, "Unlabeled")
    return f"{icon} {text}"


# ── Project guard ─────────────────────────────────────────────────────────────

def require_project() -> bool:
    """
    Show a warning and return False if no project is loaded.
    Pages should call this at the top and stop if it returns False.
    """
    from gui.utils.state import has_project
    if not has_project():
        st.warning(
            "⚠️ No project loaded. Please go to **Import** to create or open a project.",
            icon="📥",
        )
        st.page_link("pages/1_Import.py", label="→ Go to Import", icon="📥")
        return False
    return True


# ── Label button group ────────────────────────────────────────────────────────

def label_buttons(
    current_label: str | None,
    key_prefix: str = "label_btn",
) -> str | None:
    """
    Render three label buttons (Human / AI / Uncertain).
    Returns the new label string if clicked, or None if not clicked this render.
    """
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    new_label = None

    with col1:
        is_active = current_label == "human"
        if st.button(
            "🟢 Human" + (" ✓" if is_active else ""),
            key=f"{key_prefix}_human",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            new_label = "human"

    with col2:
        is_active = current_label == "ai_generated"
        if st.button(
            "🔴 AI Generated" + (" ✓" if is_active else ""),
            key=f"{key_prefix}_ai",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            new_label = "ai_generated"

    with col3:
        is_active = current_label == "uncertain"
        if st.button(
            "🟡 Uncertain" + (" ✓" if is_active else ""),
            key=f"{key_prefix}_uncertain",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            new_label = "uncertain"

    with col4:
        if current_label is not None:
            if st.button(
                "⬜ Clear",
                key=f"{key_prefix}_clear",
                type="secondary",
                use_container_width=True,
            ):
                new_label = "__clear__"

    return new_label


def keyboard_shortcuts_js(container_key: str = "shortcuts") -> None:
    """
    Inject JavaScript for keyboard shortcuts in the review page.
    h = Human, a = AI, u = Uncertain, j = Next, k = Prev
    """
    st.components.v1.html(
        """
        <script>
        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            const keyMap = {
                'h': 'label_btn_human',
                'a': 'label_btn_ai',
                'u': 'label_btn_uncertain',
            };
            const btnId = keyMap[e.key];
            if (btnId) {
                // Find button by its data-testid pattern and click it
                const buttons = window.parent.document.querySelectorAll('button');
                for (const btn of buttons) {
                    if (btn.innerText.includes('Human') && e.key === 'h') { btn.click(); break; }
                    if (btn.innerText.includes('AI Generated') && e.key === 'a') { btn.click(); break; }
                    if (btn.innerText.includes('Uncertain') && e.key === 'u') { btn.click(); break; }
                }
            }
            // Navigation
            if (e.key === 'j' || e.key === 'ArrowRight') {
                const nextBtns = window.parent.document.querySelectorAll('button');
                for (const btn of nextBtns) {
                    if (btn.innerText.trim() === '▶ Next') { btn.click(); break; }
                }
            }
            if (e.key === 'k' || e.key === 'ArrowLeft') {
                const prevBtns = window.parent.document.querySelectorAll('button');
                for (const btn of prevBtns) {
                    if (btn.innerText.trim() === '◀ Prev') { btn.click(); break; }
                }
            }
        });
        </script>
        """,
        height=0,
    )


# ── Metadata panel ────────────────────────────────────────────────────────────

def metadata_panel(row: pd.Series, expanded: bool = False) -> None:
    """Render a collapsible metadata panel for a comment row."""
    meta_cols = [c for c in row.index if c.startswith("meta_") and pd.notna(row[c]) and row[c] != ""]
    if not meta_cols:
        return

    with st.expander("📋 Metadata", expanded=expanded):
        display_items = []
        for col in meta_cols:
            field = col.replace("meta_", "").replace("_", " ").title()
            val = str(row[col])
            display_items.append((field, val))

        # Display in two columns
        n = len(display_items)
        half = (n + 1) // 2
        left, right = display_items[:half], display_items[half:]

        col1, col2 = st.columns(2)
        with col1:
            for field, val in left:
                st.markdown(f"**{field}:** {val}")
        with col2:
            for field, val in right:
                st.markdown(f"**{field}:** {val}")


# ── Derived features panel ────────────────────────────────────────────────────

def derived_features_panel(
    features_by_run: dict[str, dict[str, Any]],
    expanded: bool = True,
) -> None:
    """
    Render plugin-derived features grouped by plugin run.

    Args:
        features_by_run: {run_label: {field_name: value}}
        expanded: whether expanders start open
    """
    if not features_by_run:
        return

    with st.expander("🔬 Derived Features", expanded=expanded):
        for run_label, fields in features_by_run.items():
            st.markdown(f"**{run_label}**")
            _render_feature_fields(fields)
            st.divider()


def _render_feature_fields(fields: dict[str, Any]) -> None:
    """Dynamically render feature fields based on their type."""
    for field_name, value in fields.items():
        if value is None or value == "" or value == []:
            continue

        display_name = field_name.replace("_", " ").title()

        if isinstance(value, float):
            # Scalar numeric — render as metric or progress bar
            if 0.0 <= value <= 1.0:
                col1, col2 = st.columns([2, 3])
                col1.caption(display_name)
                col2.progress(value, text=f"{value:.2f}")
            else:
                st.metric(display_name, f"{value:.4f}")

        elif isinstance(value, int):
            st.metric(display_name, str(value))

        elif isinstance(value, bool):
            icon = "✅" if value else "❌"
            st.markdown(f"**{display_name}:** {icon}")

        elif isinstance(value, list):
            # List of strings — render as chips
            if value:
                st.markdown(f"**{display_name}:**")
                chips = " ".join([f"`{v}`" for v in value if v])
                st.markdown(chips)

        elif isinstance(value, dict):
            # JSON object — expandable
            with st.expander(f"📦 {display_name}"):
                st.json(value)

        elif isinstance(value, str):
            if len(value) > 200:
                # Long text — collapsible
                with st.expander(f"📝 {display_name}"):
                    st.markdown(value)
            else:
                # Short text — inline badge
                st.markdown(f"**{display_name}:** `{value}`")


# ── Predictions panel ─────────────────────────────────────────────────────────

def predictions_panel(
    predictions_by_run: dict[str, dict[str, Any]],
    expanded: bool = False,
) -> None:
    """
    Render classifier predictions grouped by run.

    Args:
        predictions_by_run: {run_label: {pred_label, score_ai, confidence, ...}}
    """
    if not predictions_by_run:
        return

    with st.expander("🤖 Classifier Predictions", expanded=expanded):
        for run_label, pred in predictions_by_run.items():
            st.markdown(f"**{run_label}**")
            pred_label = pred.get("pred_label")
            score_ai = pred.get("score_ai")
            confidence = pred.get("confidence")

            cols = st.columns(3)
            cols[0].metric("Prediction", label_badge(pred_label))
            if score_ai is not None:
                cols[1].metric("AI Score", f"{score_ai:.3f}")
            if confidence is not None:
                cols[2].metric("Confidence", f"{confidence:.3f}")

            explanation = pred.get("explanation_json")
            if explanation:
                with st.expander("Explanation"):
                    st.json(explanation)
            st.divider()


# ── Comment card ──────────────────────────────────────────────────────────────

def comment_card(
    comment_id: str,
    comment_text: str,
    current_label: str | None = None,
    index_display: str = "",
) -> None:
    """Render the main comment reading card."""
    st.markdown(
        f"""
        <div style="
            background: #f8f9fa;
            border-left: 4px solid {'#28a745' if current_label == 'human' else '#dc3545' if current_label == 'ai_generated' else '#ffc107' if current_label == 'uncertain' else '#6c757d'};
            padding: 1rem 1.5rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 0.75rem; color: #6c757d; margin-bottom: 0.5rem;">
                {index_display} · <code>{comment_id}</code> · {label_badge(current_label)}
            </div>
            <div style="font-size: 1rem; line-height: 1.6; white-space: pre-wrap;">{comment_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
