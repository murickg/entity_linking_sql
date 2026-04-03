import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
SPIDER2_DIR = PROJECT_ROOT / "spider2-lite"
JSONL_PATH = SPIDER2_DIR / "spider2-lite.jsonl"
LOCAL_MAP_PATH = PROJECT_ROOT / "local_sqlite" / "local-map.jsonl"
SQLITE_DDL_DIR = SPIDER2_DIR / "resource" / "databases" / "sqlite"
GOLD_SQL_DIR = SPIDER2_DIR / "evaluation_suite" / "gold" / "sql"
DOCUMENTS_DIR = SPIDER2_DIR / "resource" / "documents"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"

# OpenRouter API
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-sonnet-4"

# Agent settings
MAX_AGENT_ITERATIONS = 10
SEARCH_TOP_K_TABLES = 10
SEARCH_TOP_K_COLUMNS = 20
