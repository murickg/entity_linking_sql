import csv
import json
import re
from pathlib import Path

from src.config import (
    JSONL_PATH,
    LOCAL_MAP_PATH,
    SQLITE_DDL_DIR,
    GOLD_SQL_DIR,
    DOCUMENTS_DIR,
)


def load_instances(only_local: bool = True) -> list[dict]:
    """Load instances from spider2-lite.jsonl, optionally filtering to local* only."""
    instances = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inst = json.loads(line)
            if only_local and not inst["instance_id"].startswith("local"):
                continue
            instances.append(inst)
    return instances


def load_local_map() -> dict[str, str]:
    """Load local-map.jsonl -> dict mapping instance_id to db_name."""
    with open(LOCAL_MAP_PATH, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def parse_columns_from_ddl(ddl: str) -> list[tuple[str, str]]:
    """Extract (column_name, column_type) pairs from a CREATE TABLE DDL string."""
    columns = []
    # Find content between first ( and last )
    match = re.search(r'\((.*)\)', ddl, re.DOTALL)
    if not match:
        return columns

    body = match.group(1)
    # Split by commas that are not inside parentheses
    depth = 0
    current = []
    parts = []
    for char in body:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append(''.join(current).strip())

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Skip constraints
        upper = part.upper().lstrip()
        if upper.startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK', 'CONSTRAINT', 'INDEX')):
            continue
        # Extract column name and type
        tokens = part.split()
        if len(tokens) >= 2:
            col_name = tokens[0].strip('"').strip('`').strip("'")
            col_type = tokens[1].strip('"').strip('`')
            columns.append((col_name, col_type))
        elif len(tokens) == 1:
            col_name = tokens[0].strip('"').strip('`').strip("'")
            columns.append((col_name, ""))

    return columns


def _resolve_ddl_dir(db_name: str) -> Path | None:
    """Find the DDL directory, handling case/punctuation mismatches."""
    direct = SQLITE_DDL_DIR / db_name
    if direct.exists():
        return direct
    # Case-insensitive + normalize hyphens/underscores
    normalized = db_name.lower().replace("-", "_")
    for d in SQLITE_DDL_DIR.iterdir():
        if d.is_dir() and d.name.lower().replace("-", "_") == normalized:
            return d
    return None


def load_ddl(db_name: str) -> dict[str, dict]:
    """Load and parse DDL.csv for a SQLite database.

    Returns: {table_name: {"ddl": str, "columns": [(name, type)]}}
    """
    ddl_dir = _resolve_ddl_dir(db_name)
    if not ddl_dir:
        return {}
    ddl_path = ddl_dir / "DDL.csv"
    if not ddl_path.exists():
        return {}

    result = {}
    with open(ddl_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return {}

        for row in reader:
            if len(row) < 2:
                continue
            table_name = row[0].strip()
            ddl_text = row[1].strip()
            columns = parse_columns_from_ddl(ddl_text)
            result[table_name] = {
                "ddl": ddl_text,
                "columns": columns,
            }

    return result


def load_gold_sql(instance_id: str) -> str | None:
    """Load gold SQL for an instance. Returns None if not found."""
    sql_path = GOLD_SQL_DIR / f"{instance_id}.sql"
    if not sql_path.exists():
        return None
    return sql_path.read_text(encoding="utf-8")


def load_external_knowledge(filename: str) -> str | None:
    """Load an external knowledge markdown file."""
    if not filename:
        return None
    doc_path = DOCUMENTS_DIR / filename
    if not doc_path.exists():
        return None
    return doc_path.read_text(encoding="utf-8")


def get_instances_with_gold_sql(only_local: bool = True) -> list[dict]:
    """Return only instances that have a corresponding gold SQL file."""
    instances = load_instances(only_local=only_local)
    return [inst for inst in instances if (GOLD_SQL_DIR / f"{inst['instance_id']}.sql").exists()]
