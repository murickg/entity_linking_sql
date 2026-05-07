import re

import sqlglot
from sqlglot import exp


# Map our platform names to sqlglot dialect names
DIALECT_MAP = {
    "sqlite": "sqlite",
    "bigquery": "bigquery",
    "snowflake": "snowflake",
}


def extract_tables_columns(sql: str, platform: str = "sqlite") -> tuple[set[str], set[str]]:
    """Extract table names and column references from a SQL query.

    Returns: (set of table names, set of table.column references)
    Only includes columns that reference real tables (not CTE aliases or computed names).
    """
    tables = set()
    columns = set()
    cte_names = set()
    table_aliases = {}  # alias -> real table name
    unnest_aliases = set()  # aliases created by UNNEST (BigQuery)
    select_aliases = set()  # AS aliases from SELECT/CTE (not real columns)

    dialect = DIALECT_MAP.get(platform, platform)

    # Strip DECLARE statements (BigQuery-specific, not handled by sqlglot)
    cleaned_sql = _strip_declares(sql)

    try:
        parsed = sqlglot.parse(cleaned_sql, read=dialect)
    except sqlglot.errors.ParseError:
        try:
            parsed = sqlglot.parse(cleaned_sql)
        except sqlglot.errors.ParseError:
            return set(), set()

    for statement in parsed:
        if statement is None:
            continue

        # Collect CTE names
        for cte in statement.find_all(exp.CTE):
            alias = cte.alias
            if alias:
                cte_names.add(alias.lower())

        # Collect SELECT aliases (AS names) — these are computed, not real columns
        for alias_node in statement.find_all(exp.Alias):
            alias_name = alias_node.alias
            if alias_name:
                select_aliases.add(alias_name.lower())

        # Collect UNNEST aliases (BigQuery)
        if platform == "bigquery":
            for unnest in statement.find_all(exp.Unnest):
                parent = unnest.parent
                if hasattr(parent, 'alias') and parent.alias:
                    unnest_aliases.add(parent.alias.lower())

        # Collect real table names and their aliases
        for table in statement.find_all(exp.Table):
            name = table.name
            if not name:
                continue
            name_lower = name.lower()

            # Skip CTE references
            if name_lower in cte_names:
                continue

            # For BigQuery: strip date suffixes and wildcards
            if platform == "bigquery":
                name_lower = _normalize_bq_table(name_lower)

            tables.add(name_lower)

            # Track alias -> table mapping
            alias = table.alias
            if alias:
                table_aliases[alias.lower()] = name_lower
            table_aliases[name_lower] = name_lower

        # Collect column references — only those tied to real tables
        for col in statement.find_all(exp.Column):
            col_name = col.name
            if not col_name:
                continue
            col_name_lower = col_name.lower()

            # For Snowflake: strip quotes
            col_name_lower = col_name_lower.strip('"')

            table_ref = col.table
            if table_ref:
                table_ref_lower = table_ref.lower().strip('"')
                real_table = table_aliases.get(table_ref_lower)
                if real_table:
                    columns.add(f"{real_table}.{col_name_lower}")
                elif table_ref_lower in unnest_aliases:
                    # UNNEST alias — column is a nested field
                    columns.add(f"{table_ref_lower}.{col_name_lower}")
                elif table_ref_lower not in cte_names:
                    # Unknown table ref, not a CTE — include it
                    columns.add(f"{table_ref_lower}.{col_name_lower}")
                # If table_ref is a CTE name, skip — it's a CTE-computed column
            else:
                # Unqualified column — only include if NOT a known SELECT alias
                # (aliases like "total_spent", "rank", "games_played" are computed, not real)
                if col_name_lower not in select_aliases:
                    columns.add(col_name_lower)

    return tables, columns


def _strip_declares(sql: str) -> str:
    """Remove DECLARE statements from BigQuery SQL."""
    lines = sql.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip().upper()
        if stripped.startswith('DECLARE '):
            continue
        filtered.append(line)
    return '\n'.join(filtered)


def _normalize_bq_table(name: str) -> str:
    """Normalize BigQuery table names: strip date suffixes, wildcards."""
    # Strip wildcard: ga_sessions_* -> ga_sessions
    name = name.rstrip('*').rstrip('_')
    # Strip full date suffix: ga_sessions_20170101 -> ga_sessions
    name = re.sub(r'_\d{8}$', '', name)
    # Strip partial date suffix (YYYYMM or YYYY): ga_sessions_201707, ga_sessions_2017
    name = re.sub(r'_\d{4,6}$', '', name)
    # Clean up trailing underscores
    name = name.rstrip('_')
    return name


def normalize_columns(columns: set[str], tables: set[str]) -> set[str]:
    """Normalize column references: remove CTE prefixes, keep only real table prefixes."""
    normalized = set()
    for col in columns:
        if "." in col:
            prefix, name = col.split(".", 1)
            if prefix in tables:
                normalized.add(col)
            else:
                # Prefix is not a known table (might be CTE alias), keep just column name
                normalized.add(name)
        else:
            normalized.add(col)
    return normalized


def validate_columns_against_schema(
    columns: set[str],
    tables: set[str],
    schema_columns: dict[str, set[str]],
) -> set[str]:
    """Filter ground truth columns to only include those that exist in the actual DB schema.

    Args:
        columns: Ground truth columns (from SQL parsing), may include aliases.
        tables: Ground truth tables.
        schema_columns: dict of {table_name_lower: {col_name_lower, ...}} from DDL/PRAGMA.

    Returns: filtered set of columns that exist in the schema.
    """
    validated = set()

    # Build a flat set of all known columns for quick lookup
    all_known_qualified = set()  # "table.column"
    all_known_bare = {}  # "column" -> set of tables containing it
    for table, cols in schema_columns.items():
        for col in cols:
            all_known_qualified.add(f"{table}.{col}")
            if col not in all_known_bare:
                all_known_bare[col] = set()
            all_known_bare[col].add(table)

    for col in columns:
        col_lower = col.lower()
        if "." in col_lower:
            # table.column format — check if it exists in schema
            if col_lower in all_known_qualified:
                validated.add(col_lower)
        else:
            # Bare column name — try to resolve to a GT table
            if col_lower in all_known_bare:
                # Find which GT tables contain this column
                matching_tables = all_known_bare[col_lower] & tables
                if matching_tables:
                    # Qualify with the table name(s)
                    for t in matching_tables:
                        validated.add(f"{t}.{col_lower}")
                else:
                    # Column exists in schema but not in GT tables — still valid
                    # Take any table that has it
                    for t in all_known_bare[col_lower]:
                        validated.add(f"{t}.{col_lower}")
                        break  # just one match is enough
            # If not found in schema at all, it's an alias — drop it

    return validated
