import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
SPIDER2_DIR = PROJECT_ROOT / "spider2-lite"
JSONL_PATH = SPIDER2_DIR / "spider2-lite.jsonl"
LOCAL_MAP_PATH = PROJECT_ROOT / "local_sqlite" / "local-map.jsonl"
DATABASES_DIR = SPIDER2_DIR / "resource" / "databases"
SQLITE_DDL_DIR = DATABASES_DIR / "sqlite"
BIGQUERY_DDL_DIR = DATABASES_DIR / "bigquery"
SNOWFLAKE_DDL_DIR = DATABASES_DIR / "snowflake"
GOLD_SQL_DIR = SPIDER2_DIR / "evaluation_suite" / "gold" / "sql"
DOCUMENTS_DIR = SPIDER2_DIR / "resource" / "documents"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"

# OpenRouter API
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-sonnet-4"

# Agent settings (baseline BM25)
MAX_AGENT_ITERATIONS = 10
SEARCH_TOP_K_TABLES = 10
SEARCH_TOP_K_COLUMNS = 20

# AutoLink settings
SQLITE_DB_DIR = PROJECT_ROOT / "local_sqlite"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_CACHE_DIR = PROJECT_ROOT / "cache" / "embeddings"
VS_INITIAL_TOP_N = 20
VS_RETRIEVE_TOP_M = 3
SQLITE_TIMEOUT = 5.0
SQLITE_MAX_ROWS = 50
AUTOLINK_MAX_TURNS = 10
