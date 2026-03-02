"""
Global configuration for the Anti-Slop Regulator Platform.
All secrets (API keys) must come from environment variables — never stored in files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────

# Root of the repo
REPO_ROOT = Path(__file__).parent

# Where project workspaces are stored (gitignored)
PROJECTS_ROOT = REPO_ROOT / "projects"
PROJECTS_ROOT.mkdir(exist_ok=True)

# Built-in plugins directory
BUILTIN_PLUGINS_DIR = REPO_ROOT / "core" / "plugins" / "builtins"

# Analyst / local plugins directory
LOCAL_PLUGINS_DIR = REPO_ROOT / "plugins"
LOCAL_PLUGINS_DIR.mkdir(exist_ok=True)

# Disk cache directory
CACHE_DIR = REPO_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Embedding Defaults ────────────────────────────────────────────────────────

DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# ── LLM / API Settings ────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini")
OPENAI_DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_DEFAULT_TEMPERATURE", "0.0"))
OPENAI_DEFAULT_TIMEOUT = int(os.getenv("OPENAI_DEFAULT_TIMEOUT", "60"))
OPENAI_DEFAULT_RETRIES = int(os.getenv("OPENAI_DEFAULT_RETRIES", "3"))

# ── Feature Flags ─────────────────────────────────────────────────────────────

# Enable development/test mode features (truth key evaluation)
DEV_MODE = os.getenv("DEV_MODE", "true").lower() in ("1", "true", "yes")

# ── Projection Defaults ───────────────────────────────────────────────────────

DEFAULT_PROJECTION_METHOD = "umap"      # pca | umap | tsne
DEFAULT_PROJECTION_DIMS = 2             # 2 | 3
DEFAULT_UMAP_N_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_RANDOM_SEED = 42

# ── Clustering Defaults ───────────────────────────────────────────────────────

DEFAULT_CLUSTERING_METHOD = "hdbscan"   # hdbscan | kmeans | agglomerative
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 5
DEFAULT_KMEANS_N_CLUSTERS = 10

# ── Storage ───────────────────────────────────────────────────────────────────

# Parquet engine
PARQUET_ENGINE = "pyarrow"

# Whether to also save CSV mirrors of key tables for portability
SAVE_CSV_MIRRORS = False

# ── UI ────────────────────────────────────────────────────────────────────────

APP_TITLE = "Anti-Slop Regulator Platform"
APP_ICON = "🔍"
REVIEW_PAGE_COMMENTS_PER_PAGE = 1       # one-at-a-time review

# Comment text truncation in tables/hover
COMMENT_PREVIEW_LEN = 200
