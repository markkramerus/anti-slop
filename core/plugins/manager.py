"""
Plugin discovery and registry.

Scans built-in and local plugin directories for plugin.yaml files,
loads metadata, and returns a registry of available plugins.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml

import config
from core.plugins.base import BasePlugin
from shared_models import PluginKind, PluginMetadata, PluginParameterSpec


def _load_plugin_metadata(plugin_dir: Path) -> PluginMetadata | None:
    """Parse plugin.yaml from a plugin directory. Returns None if invalid."""
    yaml_path = plugin_dir / "plugin.yaml"
    if not yaml_path.exists():
        return None
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data:
            return None

        # Parse parameters
        params = {}
        for name, spec_data in (data.get("parameters") or {}).items():
            params[name] = PluginParameterSpec(**spec_data)

        meta = PluginMetadata(
            name=data["name"],
            version=data["version"],
            kind=PluginKind(data["kind"]),
            author=data.get("author"),
            description=data.get("description"),
            entrypoint=data.get("entrypoint", "plugin.py:Plugin"),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            parameters=params,
            plugin_dir=str(plugin_dir.resolve()),
        )
        return meta
    except Exception as e:
        print(f"Warning: could not load plugin.yaml from {plugin_dir}: {e}")
        return None


def _load_plugin_class(meta: PluginMetadata) -> type[BasePlugin] | None:
    """Dynamically import and return the plugin class from its entrypoint."""
    try:
        entrypoint = meta.entrypoint  # e.g. "plugin.py:Plugin"
        if ":" not in entrypoint:
            return None
        module_file, class_name = entrypoint.split(":", 1)
        plugin_dir = Path(meta.plugin_dir)
        module_path = plugin_dir / module_file

        if not module_path.exists():
            print(f"Warning: plugin module not found: {module_path}")
            return None

        # Load module dynamically
        spec = importlib.util.spec_from_file_location(
            f"plugin_{meta.name}_{meta.version}",
            module_path,
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        # Add plugin dir to path so relative imports work
        plugin_dir_str = str(plugin_dir)
        if plugin_dir_str not in sys.path:
            sys.path.insert(0, plugin_dir_str)
        spec.loader.exec_module(module)  # type: ignore

        cls = getattr(module, class_name, None)
        if cls is None:
            print(f"Warning: class {class_name!r} not found in {module_path}")
            return None
        return cls
    except Exception as e:
        print(f"Warning: failed to load plugin {meta.name}: {e}")
        return None


class PluginRegistry:
    """In-memory registry of discovered plugins."""

    def __init__(self) -> None:
        self._metadata: dict[str, PluginMetadata] = {}
        self._classes: dict[str, type[BasePlugin]] = {}

    def register(self, meta: PluginMetadata, cls: type[BasePlugin]) -> None:
        key = f"{meta.name}@{meta.version}"
        self._metadata[key] = meta
        self._classes[key] = cls

    def all_metadata(self) -> list[PluginMetadata]:
        return list(self._metadata.values())

    def get_metadata(self, name: str, version: str | None = None) -> PluginMetadata | None:
        if version:
            return self._metadata.get(f"{name}@{version}")
        # Return latest if version not specified
        matches = [m for k, m in self._metadata.items() if m.name == name]
        return matches[-1] if matches else None

    def instantiate(self, name: str, version: str | None = None) -> BasePlugin | None:
        meta = self.get_metadata(name, version)
        if not meta:
            return None
        key = f"{meta.name}@{meta.version}"
        cls = self._classes.get(key)
        if cls is None:
            return None
        return cls()

    def by_kind(self, kind: PluginKind) -> list[PluginMetadata]:
        return [m for m in self._metadata.values() if m.kind == kind]


def discover_plugins() -> PluginRegistry:
    """
    Scan built-in and local plugin directories and return a PluginRegistry.

    Search order:
    1. core/plugins/builtins/  (each subdirectory is one plugin)
    2. plugins/                 (local analyst plugins)
    """
    registry = PluginRegistry()
    dirs_to_scan = [
        config.BUILTIN_PLUGINS_DIR,
        config.LOCAL_PLUGINS_DIR,
    ]

    for search_dir in dirs_to_scan:
        if not search_dir.exists():
            continue
        for plugin_dir in sorted(search_dir.iterdir()):
            if not plugin_dir.is_dir():
                continue
            meta = _load_plugin_metadata(plugin_dir)
            if meta is None:
                continue
            cls = _load_plugin_class(meta)
            if cls is None:
                continue
            registry.register(meta, cls)

    return registry


# Module-level singleton (lazy initialized)
_registry: PluginRegistry | None = None


def get_registry(refresh: bool = False) -> PluginRegistry:
    """Return the global plugin registry, discovering if not yet done."""
    global _registry
    if _registry is None or refresh:
        _registry = discover_plugins()
    return _registry
