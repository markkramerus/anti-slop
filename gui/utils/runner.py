"""
Helpers for running long-running backend tasks from Streamlit pages
with progress bars and live log output.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import Any, Callable

import streamlit as st


def run_with_progress(
    task_fn: Callable[..., Generator[tuple[float, str], None, Any]],
    *args,
    button_label: str = "Run",
    **kwargs,
) -> Any:
    """
    Run a generator-based task function that yields (progress_fraction, log_message)
    tuples and displays a progress bar + log area.

    The task_fn should be a generator that:
    - yields (float 0..1, str message) for each progress update
    - returns the final result (accessible via StopIteration.value)

    Returns the final result of the task, or None if not yet run.
    """
    result_key = f"_runner_result_{task_fn.__name__}"
    running_key = f"_runner_running_{task_fn.__name__}"

    if st.button(button_label, type="primary"):
        st.session_state[running_key] = True
        st.session_state.pop(result_key, None)

    if not st.session_state.get(running_key):
        return st.session_state.get(result_key)

    progress_bar = st.progress(0.0)
    log_area = st.empty()
    logs: list[str] = []

    result = None
    try:
        gen = task_fn(*args, **kwargs)
        while True:
            try:
                frac, msg = next(gen)
                progress_bar.progress(min(float(frac), 1.0))
                logs.append(msg)
                log_area.code("\n".join(logs[-20:]))  # show last 20 lines
            except StopIteration as e:
                result = e.value
                break
    except Exception as exc:
        st.error(f"Task failed: {exc}")
        st.session_state[running_key] = False
        return None

    progress_bar.progress(1.0)
    logs.append("✓ Done")
    log_area.code("\n".join(logs[-20:]))
    st.session_state[running_key] = False
    st.session_state[result_key] = result
    return result


def stream_logs(
    task_fn: Callable[..., Generator[str, None, Any]],
    *args,
    **kwargs,
) -> Any:
    """
    Run a generator task that yields log lines (strings only).
    Simpler alternative to run_with_progress when no % progress is available.
    """
    log_area = st.empty()
    logs: list[str] = []
    result = None
    try:
        gen = task_fn(*args, **kwargs)
        while True:
            try:
                line = next(gen)
                logs.append(line)
                log_area.code("\n".join(logs[-30:]))
            except StopIteration as e:
                result = e.value
                break
    except Exception as exc:
        st.error(f"Error: {exc}")
    return result
