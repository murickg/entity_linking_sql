import re
import sqlite3
import time
from pathlib import Path

from src.config import SQLITE_TIMEOUT, SQLITE_MAX_ROWS

# Disallowed SQL patterns (DDL/DML)
_UNSAFE_PATTERN = re.compile(
    r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|REPLACE|ATTACH|DETACH|VACUUM|REINDEX)\b',
    re.IGNORECASE,
)


class SQLiteExecutor:
    """Safe read-only SQL execution against a SQLite database (E_DB)."""

    def __init__(
        self,
        db_path: Path,
        timeout: float = SQLITE_TIMEOUT,
        max_rows: int = SQLITE_MAX_ROWS,
    ):
        self.db_path = db_path
        self.timeout = timeout
        self.max_rows = max_rows

    def execute(self, sql: str) -> str:
        """Execute a read-only SQL query and return formatted results or error."""
        sql = sql.strip().rstrip(";")
        if not sql:
            return "[ERROR: Empty query]"

        if _UNSAFE_PATTERN.search(sql):
            return "[ERROR: Only read-only queries (SELECT, PRAGMA) are allowed]"

        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                timeout=self.timeout,
            )
            conn.execute("PRAGMA query_only = ON")
            cursor = conn.cursor()

            start = time.time()
            cursor.execute(sql)
            elapsed = time.time() - start

            if cursor.description is None:
                conn.close()
                return f"[Query executed successfully, Execution time: {elapsed:.2f}s, No result set]"

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchmany(self.max_rows + 1)
            total_truncated = len(rows) > self.max_rows
            rows = rows[:self.max_rows]
            conn.close()

            return self._format_results(columns, rows, elapsed, total_truncated)

        except sqlite3.OperationalError as e:
            return f"[ERROR: {e}]"
        except sqlite3.DatabaseError as e:
            return f"[ERROR: {e}]"
        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def get_sample_values(self, table: str, column: str, limit: int = 10) -> list[str]:
        """Get distinct sample values for a column."""
        sql = f'SELECT DISTINCT "{column}" FROM "{table}" LIMIT {limit}'
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=self.timeout
            )
            cursor = conn.cursor()
            cursor.execute(sql)
            values = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]
            conn.close()
            return values
        except Exception:
            return []

    def get_table_info(self, table: str) -> list[dict]:
        """Return PRAGMA table_info as list of dicts."""
        sql = f'PRAGMA table_info("{table}")'
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=self.timeout
            )
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            conn.close()
            return [
                {"cid": r[0], "name": r[1], "type": r[2], "notnull": r[3], "default": r[4], "pk": r[5]}
                for r in rows
            ]
        except Exception:
            return []

    def _format_results(
        self,
        columns: list[str],
        rows: list[tuple],
        elapsed: float,
        truncated: bool,
    ) -> str:
        if not rows:
            return f"[Execution time: {elapsed:.2f}s, Query returned 0 rows]"

        # Compute column widths
        str_rows = [[str(v) for v in row] for row in rows]
        widths = [max(len(c), *(len(r[i]) for r in str_rows)) for i, c in enumerate(columns)]
        # Cap column width
        widths = [min(w, 40) for w in widths]

        header = " | ".join(c.ljust(w)[:w] for c, w in zip(columns, widths))
        sep = "-+-".join("-" * w for w in widths)

        lines = [f"[Rows: {len(rows)}, Execution time: {elapsed:.2f}s]", header, sep]
        for row in rows[:self.max_rows]:
            line = " | ".join(str(v).ljust(w)[:w] for v, w in zip(row, widths))
            lines.append(line)

        if truncated:
            lines.append(f"... (truncated to {self.max_rows} rows)")

        return "\n".join(lines)
