import csv
import json
import re
from pathlib import Path

from src.config import (
    JSONL_PATH,
    LOCAL_MAP_PATH,
    SQLITE_DDL_DIR,
    BIGQUERY_DDL_DIR,
    SNOWFLAKE_DDL_DIR,
    GOLD_SQL_DIR,
    DOCUMENTS_DIR,
)


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def detect_platform(instance_id: str) -> str:
    """Detect the platform (sqlite, bigquery, snowflake) from instance_id prefix."""
    if instance_id.startswith("local"):
        return "sqlite"
    elif instance_id.startswith("sf_bq") or instance_id.startswith("sf0"):
        return "snowflake"
    else:
        return "bigquery"


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

def load_instances(platform: str | None = None) -> list[dict]:
    """Load instances from spider2-lite.jsonl.

    Args:
        platform: Filter to a specific platform ('sqlite', 'bigquery', 'snowflake')
                  or None for all instances.
    """
    instances = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inst = json.loads(line)
            if platform is not None:
                inst_platform = detect_platform(inst["instance_id"])
                if inst_platform != platform:
                    continue
            instances.append(inst)
    return instances


def load_local_map() -> dict[str, str]:
    """Load local-map.jsonl -> dict mapping instance_id to db_name."""
    with open(LOCAL_MAP_PATH, "r", encoding="utf-8") as f:
        return json.loads(f.read())


# ---------------------------------------------------------------------------
# DDL column parsing
# ---------------------------------------------------------------------------

def parse_columns_from_ddl(ddl: str, platform: str = "sqlite") -> list[tuple[str, str]]:
    """Extract (column_name, column_type) pairs from a CREATE TABLE DDL string.

    For BigQuery, also flattens STRUCT fields into dot-path columns.
    """
    columns = []
    # Find content between first ( and last )
    match = re.search(r'\((.*)\)', ddl, re.DOTALL)
    if not match:
        return columns

    body = match.group(1)
    # Split by commas respecting nested parentheses and angle brackets
    parts = _split_top_level(body)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Skip constraints
        upper = part.upper().lstrip()
        if upper.startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK',
                             'CONSTRAINT', 'INDEX', 'CLUSTER BY')):
            continue
        # Extract column name and type
        tokens = part.split(None, 1)
        if len(tokens) >= 2:
            col_name = tokens[0].strip('"').strip('`').strip("'")
            col_type_raw = tokens[1].strip()
            # Remove trailing NOT NULL, OPTIONS(...), etc.
            col_type = _clean_type(col_type_raw)
            columns.append((col_name, col_type))

            # For BigQuery: flatten STRUCT fields into dot-path columns
            if platform == "bigquery" and ("STRUCT" in col_type.upper() or "ARRAY" in col_type.upper()):
                nested = _flatten_struct(col_name, col_type)
                columns.extend(nested)
        elif len(tokens) == 1:
            col_name = tokens[0].strip('"').strip('`').strip("'")
            columns.append((col_name, ""))

    return columns


def _split_top_level(body: str) -> list[str]:
    """Split a string by commas at top level (not inside <>, (), or quotes)."""
    depth_paren = 0
    depth_angle = 0
    current = []
    parts = []
    i = 0
    while i < len(body):
        char = body[i]
        if char in ('"', "'"):
            # Skip quoted strings
            quote = char
            current.append(char)
            i += 1
            while i < len(body) and body[i] != quote:
                current.append(body[i])
                i += 1
            if i < len(body):
                current.append(body[i])
        elif char == '(':
            depth_paren += 1
            current.append(char)
        elif char == ')':
            depth_paren -= 1
            current.append(char)
        elif char == '<':
            depth_angle += 1
            current.append(char)
        elif char == '>':
            depth_angle -= 1
            current.append(char)
        elif char == ',' and depth_paren == 0 and depth_angle == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
        i += 1
    if current:
        parts.append(''.join(current).strip())
    return parts


def _clean_type(type_str: str) -> str:
    """Remove trailing NOT NULL, OPTIONS(...), DEFAULT, etc. from type string."""
    # Remove OPTIONS(...) block
    type_str = re.sub(r'\s*OPTIONS\s*\(.*?\)\s*$', '', type_str, flags=re.DOTALL | re.IGNORECASE)
    # Remove trailing NOT NULL
    type_str = re.sub(r'\s+NOT\s+NULL\s*$', '', type_str, flags=re.IGNORECASE)
    # Remove trailing DEFAULT ...
    type_str = re.sub(r'\s+DEFAULT\s+.*$', '', type_str, flags=re.IGNORECASE)
    return type_str.strip()


def _flatten_struct(prefix: str, type_str: str) -> list[tuple[str, str]]:
    """Recursively flatten STRUCT<...> and ARRAY<STRUCT<...>> into dot-path columns."""
    result = []
    # Extract inner STRUCT fields
    # Match STRUCT<...> or ARRAY<STRUCT<...>>
    struct_match = re.search(r'STRUCT\s*<(.+)>', type_str, re.DOTALL | re.IGNORECASE)
    if not struct_match:
        return result

    inner = struct_match.group(1)
    fields = _split_top_level(inner)
    for field in fields:
        field = field.strip()
        if not field:
            continue
        tokens = field.split(None, 1)
        if len(tokens) >= 2:
            field_name = tokens[0].strip('"').strip('`')
            field_type = _clean_type(tokens[1])
            full_path = f"{prefix}.{field_name}"
            result.append((full_path, field_type))
            # Recurse for nested STRUCTs
            if "STRUCT" in field_type.upper():
                result.extend(_flatten_struct(full_path, field_type))
        elif len(tokens) == 1:
            field_name = tokens[0].strip('"').strip('`')
            result.append((f"{prefix}.{field_name}", ""))

    return result


# ---------------------------------------------------------------------------
# DDL loading per platform
# ---------------------------------------------------------------------------

def _resolve_sqlite_dir(db_name: str) -> Path | None:
    """Find the SQLite DDL directory, handling case/punctuation mismatches."""
    direct = SQLITE_DDL_DIR / db_name
    if direct.exists():
        return direct
    normalized = db_name.lower().replace("-", "_")
    for d in SQLITE_DDL_DIR.iterdir():
        if d.is_dir() and d.name.lower().replace("-", "_") == normalized:
            return d
    return None


def _read_ddl_csv(ddl_path: Path, platform: str) -> dict[str, dict]:
    """Read and parse a single DDL.csv file."""
    result = {}
    with open(ddl_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return {}

        # Detect column layout: Snowflake has 3 cols (table_name, description, DDL)
        # BigQuery/SQLite have 2 cols (table_name, ddl)
        has_description = len(header) >= 3

        for row in reader:
            if len(row) < 2:
                continue
            table_name = row[0].strip()
            if has_description:
                description = row[1].strip() if len(row) > 1 else ""
                ddl_text = row[2].strip() if len(row) > 2 else ""
            else:
                description = ""
                ddl_text = row[1].strip()

            columns = parse_columns_from_ddl(ddl_text, platform=platform)
            result[table_name] = {
                "ddl": ddl_text,
                "columns": columns,
                "description": description,
            }

    return result


def load_ddl(db_name: str, platform: str = "sqlite") -> dict[str, dict]:
    """Load and parse DDL.csv for a database.

    Returns: {table_name: {"ddl": str, "columns": [(name, type)], "description": str}}
    """
    if platform == "sqlite":
        ddl_dir = _resolve_sqlite_dir(db_name)
        if not ddl_dir:
            return {}
        ddl_path = ddl_dir / "DDL.csv"
        if not ddl_path.exists():
            return {}
        return _read_ddl_csv(ddl_path, platform)

    elif platform == "bigquery":
        db_dir = BIGQUERY_DDL_DIR / db_name
        if not db_dir.exists():
            return {}
        # BQ has nested dataset dirs: bigquery/{db}/{project.dataset}/DDL.csv
        result = {}
        for dataset_dir in db_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            ddl_path = dataset_dir / "DDL.csv"
            if ddl_path.exists():
                tables = _read_ddl_csv(ddl_path, platform)
                # Deduplicate date-suffixed tables (e.g., ga_sessions_20170101 -> ga_sessions)
                tables = _dedup_bq_tables(tables)
                result.update(tables)
        return result

    elif platform == "snowflake":
        db_dir = SNOWFLAKE_DDL_DIR / db_name
        if not db_dir.exists():
            return {}
        # SF has nested schema dirs: snowflake/{DB}/{SCHEMA}/DDL.csv
        result = {}
        for schema_dir in db_dir.iterdir():
            if not schema_dir.is_dir():
                continue
            ddl_path = schema_dir / "DDL.csv"
            if ddl_path.exists():
                tables = _read_ddl_csv(ddl_path, platform)
                # Store tables under both full name (db.schema.table) and short name (table)
                normalized = {}
                for name, data in tables.items():
                    normalized[name] = data
                    # Add short name (last part after dots)
                    short = name.rsplit(".", 1)[-1] if "." in name else name
                    if short != name:
                        normalized[short] = data
                result.update(normalized)
        return result

    return {}


def _dedup_bq_tables(tables: dict[str, dict]) -> dict[str, dict]:
    """Deduplicate BigQuery date-suffixed tables (ga_sessions_20170101, etc.).

    Keeps one representative table per unique base name.
    """
    # Detect date-suffix patterns: table_YYYYMMDD, table_YYYY, table_YYYYQ1, etc.
    date_suffix = re.compile(r'^(.+?)_(\d{4,8}(?:_q\d)?)$', re.IGNORECASE)
    base_names: dict[str, str] = {}  # base_name -> first full name seen

    result = {}
    for name, data in tables.items():
        m = date_suffix.match(name)
        if m:
            base = m.group(1)
            if base not in base_names:
                base_names[base] = name
                result[base] = data  # Store under base name
        else:
            result[name] = data

    return result


# ---------------------------------------------------------------------------
# Instance -> DB resolution
# ---------------------------------------------------------------------------

def resolve_db_name(instance: dict, local_map: dict[str, str] | None = None) -> str | None:
    """Resolve an instance to its database name.

    For local instances, uses local_map. For BQ/SF, uses the 'db' field directly.
    """
    instance_id = instance["instance_id"]
    platform = detect_platform(instance_id)

    if platform == "sqlite":
        if local_map is None:
            local_map = load_local_map()
        return local_map.get(instance_id)
    else:
        return instance.get("db")


# ---------------------------------------------------------------------------
# Gold SQL and external knowledge
# ---------------------------------------------------------------------------

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


def get_instances_with_gold_sql(platform: str | None = None) -> list[dict]:
    """Return only instances that have a corresponding gold SQL file.

    Args:
        platform: Filter to a specific platform, or None for all.
    """
    instances = load_instances(platform=platform)
    return [inst for inst in instances if (GOLD_SQL_DIR / f"{inst['instance_id']}.sql").exists()]
