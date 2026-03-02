"""
Tests for the label_store annotation backend.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from shared_models import ManualAnnotation, ManualLabel


# Patch config to use a temp projects root
@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Create a temporary project workspace and patch config."""
    import config
    monkeypatch.setattr(config, "PROJECTS_ROOT", tmp_path)
    project_id = "test_project_001"
    from core.storage.paths import ensure_project_dirs
    ensure_project_dirs(project_id)
    return project_id


class TestLabelStore:
    def test_save_and_retrieve(self, tmp_project):
        from core.annotations.label_store import get_annotation, save_annotation

        ann = ManualAnnotation(
            comment_id="ABC-001",
            label=ManualLabel.HUMAN,
            note="Clearly personal",
        )
        save_annotation(tmp_project, ann)

        retrieved = get_annotation(tmp_project, "ABC-001")
        assert retrieved is not None
        assert retrieved.label == ManualLabel.HUMAN
        assert retrieved.note == "Clearly personal"

    def test_update_annotation(self, tmp_project):
        from core.annotations.label_store import get_annotation, save_annotation

        ann = ManualAnnotation(comment_id="ABC-002", label=ManualLabel.UNCERTAIN)
        save_annotation(tmp_project, ann)

        ann2 = ManualAnnotation(comment_id="ABC-002", label=ManualLabel.AI_GENERATED)
        save_annotation(tmp_project, ann2)

        retrieved = get_annotation(tmp_project, "ABC-002")
        assert retrieved.label == ManualLabel.AI_GENERATED

    def test_delete_annotation(self, tmp_project):
        from core.annotations.label_store import (
            delete_annotation,
            get_annotation,
            save_annotation,
        )

        ann = ManualAnnotation(comment_id="ABC-003", label=ManualLabel.HUMAN)
        save_annotation(tmp_project, ann)
        delete_annotation(tmp_project, "ABC-003")

        retrieved = get_annotation(tmp_project, "ABC-003")
        assert retrieved is None

    def test_annotated_ids(self, tmp_project):
        from core.annotations.label_store import annotated_comment_ids, save_annotation

        for i, label in enumerate([ManualLabel.HUMAN, ManualLabel.AI_GENERATED]):
            save_annotation(tmp_project, ManualAnnotation(comment_id=f"C-{i}", label=label))

        ids = annotated_comment_ids(tmp_project)
        assert "C-0" in ids
        assert "C-1" in ids

    def test_label_counts(self, tmp_project):
        from core.annotations.label_store import label_counts, save_annotation

        for i in range(3):
            save_annotation(tmp_project, ManualAnnotation(comment_id=f"H-{i}", label=ManualLabel.HUMAN))
        for i in range(2):
            save_annotation(tmp_project, ManualAnnotation(comment_id=f"A-{i}", label=ManualLabel.AI_GENERATED))

        counts = label_counts(tmp_project)
        assert counts.get("human") == 3
        assert counts.get("ai_generated") == 2

    def test_get_missing_returns_none(self, tmp_project):
        from core.annotations.label_store import get_annotation

        result = get_annotation(tmp_project, "DOES_NOT_EXIST")
        assert result is None
