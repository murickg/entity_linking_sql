"""BigQuery executor — supports both live execution and metadata-only fallback.

Modes:
  - Live (default if credentials available): real BQ queries via google-cloud-bigquery
  - Metadata-only (if no credentials): sample values from DDL JSON, stub for execute()
"""

import json
import re
import threading
import time
from pathlib import Path

from src.config import PROJECT_ROOT


_UNSAFE_PATTERN = re.compile(
    r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|REPLACE|TRUNCATE|GRANT|REVOKE|MERGE)\b',
    re.IGNORECASE,
)


def _find_credential_file() -> str | None:
    """Look for a service account JSON in the project root."""
    root = PROJECT_ROOT
    for f in root.iterdir():
        if f.is_file() and f.suffix == ".json":
            try:
                data = json.load(open(f))
                if data.get("type") == "service_account" and "project_id" in data:
                    return str(f)
            except Exception:
                continue
    return None


class BigQueryExecutor:
    """BigQuery executor with optional live query execution."""

    _client = None
    _client_lock = threading.Lock()
    _credential_path: str | None = None

    def __init__(
        self,
        sample_rows: dict | None = None,
        max_rows: int = 50,
        timeout: float = 60.0,
        live: bool | None = None,
    ):
        self.sample_rows = sample_rows or {}
        self.max_rows = max_rows
        self.timeout = timeout

        # Auto-detect live mode unless explicitly disabled
        if live is None:
            live = _find_credential_file() is not None
        self.live = live

    @classmethod
    def _get_client(cls):
        if cls._client is not None:
            return cls._client
        with cls._client_lock:
            if cls._client is None:
                from google.oauth2 import service_account
                from google.cloud import bigquery

                cls._credential_path = _find_credential_file()
                if cls._credential_path is None:
                    raise RuntimeError(
                        "No BigQuery service account JSON found in project root. "
                        "Place a service account key (.json) in the project root, "
                        "or use BigQueryExecutor(live=False) for metadata-only mode."
                    )
                creds = service_account.Credentials.from_service_account_file(
                    cls._credential_path
                )
                cls._client = bigquery.Client(credentials=creds)
        return cls._client

    def execute(self, sql: str) -> str:
        """Execute a read-only SQL query and return formatted results."""
        sql = sql.strip().rstrip(";")
        if not sql:
            return "[ERROR: Empty query]"
        if _UNSAFE_PATTERN.search(sql):
            return "[ERROR: Only read-only queries are allowed]"

        if not self.live:
            return ("[INFO: Live SQL execution disabled. "
                    "Use `verify_schema` to commit SQL — schema will be auto-extracted.]")

        try:
            client = self._get_client()
            start = time.time()
            query_job = client.query(sql)
            rows = list(query_job.result(timeout=self.timeout, max_results=self.max_rows + 1))
            elapsed = time.time() - start

            if not rows:
                return f"[Execution time: {elapsed:.2f}s, 0 rows]"

            truncated = len(rows) > self.max_rows
            rows = rows[: self.max_rows]
            columns = list(rows[0].keys()) if rows else []
            return self._format(columns, rows, elapsed, truncated)
        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {str(e)[:300]}]"

    def get_sample_values(self, table: str, column: str, limit: int = 10) -> list[str]:
        """Look up sample values from pre-loaded DDL JSON (cheaper than live query)."""
        if table in self.sample_rows:
            vals = self.sample_rows[table].get(column, [])
            return [str(v) for v in vals if v is not None][:limit]
        for t, cols in self.sample_rows.items():
            if t.lower() == table.lower():
                vals = cols.get(column, [])
                if not vals:
                    for c, vs in cols.items():
                        if c.lower() == column.lower():
                            vals = vs
                            break
                return [str(v) for v in vals if v is not None][:limit]
        return []

    def get_view_definition(self, name: str) -> str | None:
        """Could query INFORMATION_SCHEMA.VIEWS, but skipped for now."""
        return None

    def _format(self, columns, rows, elapsed, truncated):
        str_rows = [[str(r[c]) for c in columns] for r in rows]
        widths = [max(len(c), *(len(sr[i]) for sr in str_rows)) for i, c in enumerate(columns)]
        widths = [min(w, 40) for w in widths]
        header = " | ".join(c.ljust(w)[:w] for c, w in zip(columns, widths))
        sep = "-+-".join("-" * w for w in widths)
        lines = [f"[Rows: {len(rows)}, Execution time: {elapsed:.2f}s]", header, sep]
        for r in rows:
            line = " | ".join(str(r[c]).ljust(w)[:w] for c, w in zip(columns, widths))
            lines.append(line)
        if truncated:
            lines.append(f"... (truncated to {self.max_rows} rows)")
        return "\n".join(lines)
