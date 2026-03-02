"""
Abstract base class for all Anti-Slop plugins.

Every plugin (enrichment or classifier) must subclass BasePlugin
and implement metadata(), validate_inputs(), and run().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from shared_models import PluginKind, PluginMetadata


@dataclass
class PluginContext:
    """Runtime context passed to every plugin.run() call."""
    project_id: str
    run_id: str
    params: dict[str, Any]
    # Available artifact DataFrames (may be None if not yet computed)
    embeddings_df: pd.DataFrame | None = None
    enrichments: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Logging callback: call context.log("message") for progress output
    _log_fn: Any = None

    def log(self, msg: str) -> None:
        if self._log_fn:
            self._log_fn(msg)


@dataclass
class PluginRunResult:
    """Standardized output from a plugin execution."""
    kind: PluginKind
    per_comment_df: pd.DataFrame          # must include 'comment_id'
    run_metadata: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error_count: int = 0


class BasePlugin(ABC):
    """
    Abstract base for all plugins.

    Subclasses must implement:
      - metadata(): return PluginMetadata
      - validate_inputs(): return list of error strings (empty = ok)
      - run(): return PluginRunResult
    """

    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata (name, version, kind, params, etc.)."""
        ...

    @abstractmethod
    def validate_inputs(
        self,
        dataset: pd.DataFrame,
        available_artifacts: dict[str, Any],
        params: dict[str, Any],
    ) -> list[str]:
        """
        Validate that required inputs are present.
        Returns a list of error messages (empty list = all good).
        """
        ...

    @abstractmethod
    def run(
        self,
        dataset: pd.DataFrame,
        available_artifacts: dict[str, Any],
        params: dict[str, Any],
        context: PluginContext,
    ) -> PluginRunResult:
        """
        Execute the plugin on the dataset.

        Args:
            dataset: normalized comments DataFrame
            available_artifacts: dict with 'embeddings', 'enrichments', etc.
            params: resolved plugin parameters
            context: runtime context (project_id, run_id, log fn, etc.)

        Returns:
            PluginRunResult with per_comment_df and metadata
        """
        ...

    def default_params(self) -> dict[str, Any]:
        """Return default parameter values from plugin metadata."""
        meta = self.metadata()
        return {
            name: spec.default
            for name, spec in meta.parameters.items()
        }

    def resolve_params(self, user_params: dict[str, Any]) -> dict[str, Any]:
        """Merge user-supplied params with defaults."""
        resolved = self.default_params()
        resolved.update(user_params)
        return resolved
