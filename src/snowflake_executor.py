"""Snowflake SQL execution wrapper (mirror of SQLiteExecutor)."""

import json
import re
import threading
import time
from pathlib import Path

import snowflake.connector

from src.config import PROJECT_ROOT


_UNSAFE_PATTERN = re.compile(
    r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|REPLACE|TRUNCATE|GRANT|REVOKE)\b',
    re.IGNORECASE,
)


class SnowflakeExecutor:
    """Read-only Snowflake executor with shared connection."""

    _connection = None
    _connection_lock = threading.Lock()

    def __init__(
        self,
        timeout: float = 30.0,
        max_rows: int = 50,
        default_db: str | None = None,
        default_schema: str | None = None,
    ):
        self.timeout = timeout
        self.max_rows = max_rows
        self.default_db = default_db
        self.default_schema = default_schema

    @classmethod
    def _get_connection(cls):
        with cls._connection_lock:
            if cls._connection is None:
                cred_path = PROJECT_ROOT / "snowflake_credential" / "snowflake_credential.json"
                cred = json.load(open(cred_path))
                cls._connection = snowflake.connector.connect(**cred)
            return cls._connection

    def _use_db(self, cursor, db_name: str | None, schema: str | None = None):
        """Switch the active DATABASE (and optionally SCHEMA) on the cursor's session."""
        if db_name:
            try:
                cursor.execute(f'USE DATABASE "{db_name}"')
            except Exception:
                pass
        schema = schema or self.default_schema
        if schema:
            try:
                cursor.execute(f'USE SCHEMA "{schema}"')
            except Exception:
                pass

    def execute(self, sql: str, db_name: str | None = None) -> str:
        """Execute a read-only SQL query and return formatted results."""
        sql = sql.strip().rstrip(";")
        if not sql:
            return "[ERROR: Empty query]"
        if _UNSAFE_PATTERN.search(sql):
            return "[ERROR: Only read-only queries are allowed]"

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            self._use_db(cur, db_name or self.default_db)
            start = time.time()
            cur.execute(sql)
            elapsed = time.time() - start

            if cur.description is None:
                cur.close()
                return f"[Query OK, {elapsed:.2f}s, no result set]"

            columns = [d[0] for d in cur.description]
            rows = cur.fetchmany(self.max_rows + 1)
            truncated = len(rows) > self.max_rows
            rows = rows[:self.max_rows]
            cur.close()
            return self._format(columns, rows, elapsed, truncated)
        except Exception as e:
            return f"[ERROR: {e}]"

    def get_sample_values(self, table_fullname: str, column: str, limit: int = 10) -> list[str]:
        """Get distinct sample values for a column. table_fullname is DB.SCHEMA.TABLE."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = (
                f'SELECT DISTINCT "{column}" FROM {table_fullname} '
                f'WHERE "{column}" IS NOT NULL LIMIT {limit}'
            )
            cur.execute(sql)
            values = [str(r[0]) for r in cur.fetchall() if r[0] is not None]
            cur.close()
            return values
        except Exception:
            return []

    def get_view_definition(self, table_fullname: str) -> str | None:
        """Return view SQL if it's a view. Snowflake: query INFORMATION_SCHEMA.VIEWS."""
        try:
            parts = table_fullname.split(".")
            if len(parts) != 3:
                return None
            db, schema, name = parts
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(
                f'SELECT VIEW_DEFINITION FROM "{db}".INFORMATION_SCHEMA.VIEWS '
                f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{name}'"
            )
            row = cur.fetchone()
            cur.close()
            return row[0] if row else None
        except Exception:
            return None

    def _format(self, columns, rows, elapsed, truncated):
        if not rows:
            return f"[Execution time: {elapsed:.2f}s, 0 rows]"
        str_rows = [[str(v) for v in row] for row in rows]
        widths = [max(len(c), *(len(r[i]) for r in str_rows)) for i, c in enumerate(columns)]
        widths = [min(w, 40) for w in widths]
        header = " | ".join(c.ljust(w)[:w] for c, w in zip(columns, widths))
        sep = "-+-".join("-" * w for w in widths)
        lines = [f"[Rows: {len(rows)}, Execution time: {elapsed:.2f}s]", header, sep]
        for row in rows:
            line = " | ".join(str(v).ljust(w)[:w] for v, w in zip(row, widths))
            lines.append(line)
        if truncated:
            lines.append(f"... (truncated to {self.max_rows} rows)")
        return "\n".join(lines)

    @classmethod
    def close(cls):
        if cls._connection:
            cls._connection.close()
            cls._connection = None
