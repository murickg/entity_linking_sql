"""Schema prompt formatters for end-to-end EX evaluation.

Three formats:
- format_full_schema(db_name): all tables/columns + sample values
- format_autolink_schema(instance_id): read official AutoLink linking_results
- format_our_schema(linked_schema, ...): format our LinkedSchema with sample values
"""

import os
from pathlib import Path

from src.sqlite_executor import SQLiteExecutor


OFFICIAL_AUTOLINK_RESULTS = Path("/Users/muradgamzatov/Desktop/AutoLink/linking_results")


def _format_sample_values(values: list, limit: int = 5) -> str:
    """Format sample values list as compact string."""
    truncated = []
    for v in values[:limit]:
        s = str(v)
        if len(s) > 50:
            s = s[:50] + "..."
        truncated.append(s)
    return f"[{', '.join(repr(v) for v in truncated)}]"


def format_full_schema(
    db_name: str,
    ddl_data: dict,
    executor=None,
    external_samples: dict | None = None,
    bq_fullnames: dict | None = None,
) -> str:
    """Build a full-schema prompt: every table with all columns + sample values.

    For SQLite, pass an executor (live queries).
    For Snowflake/BQ, pass external_samples = {table: {col: [values]}}.
    """
    parts = []
    seen_tables = set()
    for table_name, table_info in ddl_data.items():
        # Dedup: snowflake DDL stores both full path and short name
        canonical = table_name.lower()
        if canonical in seen_tables:
            continue
        # If full path exists, prefer it
        if "." not in table_name:
            full_variants = [k for k in ddl_data if "." in k and k.lower().endswith("." + canonical)]
            if full_variants:
                continue
        seen_tables.add(canonical)

        columns = table_info.get("columns", [])
        if not columns:
            continue
        # For BQ — use fully-qualified wildcard name if available
        display_name = table_name
        if bq_fullnames and table_name in bq_fullnames:
            display_name = bq_fullnames[table_name]
        parts.append(f"###Table full name: `{display_name}`\n[")
        for col_name, col_type in columns:
            if executor is not None:
                samples = executor.get_sample_values(table_name, col_name, limit=5)
            elif external_samples and table_name in external_samples:
                raw = external_samples[table_name].get(col_name, [])
                if not raw and "." in col_name:
                    # Walk nested path for BQ STRUCTs
                    parts_path = col_name.split(".")
                    cur = external_samples[table_name].get(parts_path[0], [])
                    nested_vals = []
                    for v in cur:
                        if isinstance(v, dict):
                            node = v
                            for p in parts_path[1:]:
                                if isinstance(node, dict):
                                    node = node.get(p)
                                else:
                                    node = None
                                    break
                            if node is not None:
                                nested_vals.append(node)
                    raw = nested_vals
                samples = [str(v) for v in raw if v is not None][:5]
            else:
                samples = []
            sample_str = _format_sample_values(samples)
            parts.append(f"    {col_name} (Type: {col_type}; Sample values: {sample_str})")
        parts.append("]\n\n--------------------------------------------------\n")
    return "\n".join(parts)


def format_autolink_schema(instance_id: str) -> str | None:
    """Read official AutoLink's pre-computed schema linking result."""
    path = OFFICIAL_AUTOLINK_RESULTS / f"{instance_id}.txt"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # Strip the "External knowledge" section if present — we add that separately
    if "External knowledge that might be helpful:" in content:
        content = content.split("External knowledge that might be helpful:")[0]
    return content.strip()


def format_our_schema(
    pred_tables: list[str],
    pred_columns: list[str],
    ddl_data: dict,
    executor=None,
    external_samples: dict | None = None,
    bq_fullnames: dict | None = None,
) -> str:
    """Format our agent's predicted schema in AutoLink-style for SQL generation."""
    # Group columns by table (handles nested like "ga_sessions.trafficSource.source")
    table_cols: dict[str, list[str]] = {}
    for col in pred_columns:
        if "." in col:
            table, c = col.split(".", 1)
            table_cols.setdefault(table, []).append(c)

    # Build prompt
    parts = []
    for table in sorted(pred_tables):
        # Find canonical case from DDL
        canonical = table
        for ddl_t in ddl_data.keys():
            if ddl_t.lower() == table.lower():
                canonical = ddl_t
                break

        table_info = ddl_data.get(canonical, {})
        all_columns = table_info.get("columns", [])
        # Include both flat and nested column types
        col_types = {c.lower(): t for c, t in all_columns}

        # For BigQuery — use the fully-qualified wildcard name
        display_name = canonical
        if bq_fullnames and canonical in bq_fullnames:
            display_name = bq_fullnames[canonical]
        parts.append(f"###Table full name: `{display_name}`\n[")
        cols = table_cols.get(table, [])
        if not cols:
            cols = [c for c, _ in all_columns]

        for c in cols:
            c_lower = c.lower()
            col_type = col_types.get(c_lower, "")
            if executor is not None:
                samples = executor.get_sample_values(canonical, c, limit=5)
            elif external_samples and canonical in external_samples:
                raw = external_samples[canonical].get(c, [])
                if not raw and "." in c:
                    # Walk nested path for BQ STRUCTs
                    parts_path = c.split(".")
                    cur = external_samples[canonical].get(parts_path[0], [])
                    nested_vals = []
                    for v in cur:
                        if isinstance(v, dict):
                            node = v
                            for p in parts_path[1:]:
                                if isinstance(node, dict):
                                    node = node.get(p)
                                else:
                                    node = None
                                    break
                            if node is not None:
                                nested_vals.append(node)
                    raw = nested_vals
                samples = [str(v) for v in raw if v is not None][:5]
            else:
                samples = []
            sample_str = _format_sample_values(samples)
            parts.append(f"    {c} (Type: {col_type}; Sample values: {sample_str})")
        parts.append("]\n\n--------------------------------------------------\n")
    return "\n".join(parts)
