"""
Canonical Pydantic data models for the Anti-Slop Regulator Platform.
All layers share these schemas; never use raw dicts across module boundaries.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────


class ManualLabel(str, Enum):
    HUMAN = "human"
    AI_GENERATED = "ai_generated"
    UNCERTAIN = "uncertain"


class PredLabel(str, Enum):
    HUMAN = "human"
    AI_GENERATED = "ai_generated"
    UNCERTAIN = "uncertain"


class TruthLabel(str, Enum):
    """Labels that appear in the truth key (normalized)."""
    HUMAN = "human"
    AI_GENERATED = "ai_generated"


class IngestStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


class PluginKind(str, Enum):
    ENRICHMENT = "enrichment"
    CLASSIFIER = "classifier"


class ProjectionMethod(str, Enum):
    PCA = "pca"
    UMAP = "umap"
    TSNE = "tsne"


class ClusteringMethod(str, Enum):
    HDBSCAN = "hdbscan"
    KMEANS = "kmeans"
    AGGLOMERATIVE = "agglomerative"


class EmbeddingProvider(str, Enum):
    LOCAL = "local"
    OPENAI = "openai"


# ── Core Comment Records ──────────────────────────────────────────────────────


class CommentMetadata(BaseModel):
    """Optional metadata fields ingested from source CSV."""
    agency_id: str | None = None
    docket_id: str | None = None
    tracking_number: str | None = None
    document_type: str | None = None
    posted_date: str | None = None
    is_withdrawn: str | None = None
    title: str | None = None
    topics: str | None = None
    duplicate_comments: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    city: str | None = None
    state_province: str | None = None
    country: str | None = None
    organization_name: str | None = None
    government_agency: str | None = None
    category: str | None = None
    attachment_files: str | None = None

    class Config:
        extra = "allow"  # absorb any extra source columns


class CommentRecord(BaseModel):
    """Normalized internal representation of a single public comment."""
    comment_id: str
    comment_text: str
    source_row_index: int
    metadata: CommentMetadata = Field(default_factory=CommentMetadata)
    raw_row: dict[str, Any] = Field(default_factory=dict)
    text_hash: str = ""
    ingest_status: IngestStatus = IngestStatus.OK
    ingest_warning: str | None = None

    @model_validator(mode="after")
    def compute_text_hash(self) -> "CommentRecord":
        if not self.text_hash and self.comment_text:
            normalized = self.comment_text.strip().lower()
            self.text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return self


# ── Annotation Records ────────────────────────────────────────────────────────


class ManualAnnotation(BaseModel):
    """Analyst-assigned label for a comment."""
    annotation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    comment_id: str
    label: ManualLabel
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    note: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Plugin Records ────────────────────────────────────────────────────────────


class ClassifierPrediction(BaseModel):
    """Output from a classifier plugin for a single comment."""
    comment_id: str
    pred_label: PredLabel
    score_ai: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    explanation_json: dict[str, Any] = Field(default_factory=dict)


class DerivedFeatureRecord(BaseModel):
    """Output from an enrichment plugin for a single comment."""
    comment_id: str
    theme_primary: str | None = None
    theme_labels: list[str] = Field(default_factory=list)
    stance: str | None = None
    stance_confidence: float | None = None
    template_similarity_score: float | None = None
    density_score: float | None = None
    explanation_json: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"  # arbitrary additional enrichment columns


# ── Plugin Metadata ───────────────────────────────────────────────────────────


class PluginParameterSpec(BaseModel):
    type: str  # string | int | float | bool | enum
    default: Any = None
    values: list[str] | None = None   # for enum type
    min: float | None = None
    max: float | None = None
    description: str | None = None


class PluginMetadata(BaseModel):
    """Parsed from plugin.yaml."""
    name: str
    version: str
    kind: PluginKind
    author: str | None = None
    description: str | None = None
    entrypoint: str  # "plugin.py:Plugin"
    inputs: dict[str, list[str]] = Field(default_factory=dict)  # required/optional
    outputs: dict[str, list[str]] = Field(default_factory=dict)  # per_comment
    parameters: dict[str, PluginParameterSpec] = Field(default_factory=dict)
    plugin_dir: str = ""  # absolute path, set at discovery time


class PluginRunManifest(BaseModel):
    """Persisted metadata for a completed plugin execution."""
    model_config = ConfigDict(protected_namespaces=())
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plugin_name: str
    plugin_version: str
    plugin_kind: PluginKind
    params: dict[str, Any] = Field(default_factory=dict)
    provider: str | None = None
    model_name: str | None = None
    prompt_hash: str | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    row_count: int = 0
    error_count: int = 0
    status: str = "pending"  # pending | running | complete | failed


# ── Ingestion Summary ────────────────────────────────────────────────────────


class LengthDistribution(BaseModel):
    min: int
    max: int
    mean: float
    median: float
    p25: float
    p75: float


class IngestSummary(BaseModel):
    """Summary report from a CSV ingest operation."""
    total_rows: int
    rows_with_comment: int
    rows_missing_comment: int
    duplicate_comment_ids: int
    duplicate_text_hashes: int
    length_distribution: LengthDistribution
    dataset_fingerprint: str
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ── Embedding Records ────────────────────────────────────────────────────────


class EmbeddingRunManifest(BaseModel):
    """Metadata for a completed embedding generation run."""
    model_config = ConfigDict(protected_namespaces=())
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    provider: EmbeddingProvider
    model_name: str
    batch_size: int
    row_count: int
    embedding_dim: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    params: dict[str, Any] = Field(default_factory=dict)


class ProjectionRunManifest(BaseModel):
    """Metadata for a completed projection run."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding_run_id: str
    method: ProjectionMethod
    dims: int
    params: dict[str, Any] = Field(default_factory=dict)
    row_count: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ClusterRunManifest(BaseModel):
    """Metadata for a completed clustering run."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    projection_run_id: str
    method: ClusteringMethod
    n_clusters: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    row_count: int
    noise_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Project Manifest ──────────────────────────────────────────────────────────


class ProjectManifest(BaseModel):
    """Top-level project descriptor persisted to manifest.json."""
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    source_filename: str
    dataset_fingerprint: str
    ingest_summary: IngestSummary | None = None
    embedding_runs: list[EmbeddingRunManifest] = Field(default_factory=list)
    projection_runs: list[ProjectionRunManifest] = Field(default_factory=list)
    cluster_runs: list[ClusterRunManifest] = Field(default_factory=list)
    plugin_runs: list[PluginRunManifest] = Field(default_factory=list)
    app_version: str = "0.2.0"


# ── Evaluation Records ────────────────────────────────────────────────────────


class TruthRecord(BaseModel):
    """A single row from the truth key CSV (normalized)."""
    comment_id: str
    truth_label: TruthLabel


class TruthJoinDiagnostics(BaseModel):
    total_truth_rows: int
    matched_rows: int
    unmatched_truth_rows: int
    unmatched_dataset_rows: int
    duplicate_truth_keys: int
    coverage_pct: float
    warnings: list[str] = Field(default_factory=list)


class ConfusionMatrix(BaseModel):
    tp: int  # AI predicted, AI true
    tn: int  # Human predicted, Human true
    fp: int  # AI predicted, Human true
    fn: int  # Human predicted, AI true


class ClassificationMetrics(BaseModel):
    confusion_matrix: ConfusionMatrix
    accuracy: float
    precision_ai: float
    recall_ai: float
    f1_ai: float
    specificity: float
    support_ai: int
    support_human: int
    roc_auc: float | None = None
    pr_auc: float | None = None
    uncertain_handling: str = "exclude"


class EvaluationRunManifest(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    classifier_run_id: str
    truth_key_filename: str
    uncertain_handling: str = "exclude"
    metrics: ClassificationMetrics | None = None
    join_diagnostics: TruthJoinDiagnostics | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
