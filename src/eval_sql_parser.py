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
    """
    tables = set()
    columns = set()
    cte_names = set()
    table_aliases = {}  # alias -> real table name
    unnest_aliases = set()  # aliases created by UNNEST (BigQuery)

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

        # Collect column references
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
                else:
                    columns.add(f"{table_ref_lower}.{col_name_lower}")
            else:
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
