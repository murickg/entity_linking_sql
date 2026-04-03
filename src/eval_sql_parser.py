import sqlglot
from sqlglot import exp


def extract_tables_columns(sql: str) -> tuple[set[str], set[str]]:
    """Extract table names and column references from a SQL query.

    Returns: (set of table names, set of table.column references)
    """
    tables = set()
    columns = set()
    cte_names = set()
    table_aliases = {}  # alias -> real table name

    try:
        parsed = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        # Fallback: try without dialect
        try:
            parsed = sqlglot.parse(sql)
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

        # Collect real table names and their aliases
        for table in statement.find_all(exp.Table):
            name = table.name
            if not name:
                continue
            name_lower = name.lower()
            # Skip CTE references
            if name_lower in cte_names:
                continue
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

            # Determine the table this column belongs to
            table_ref = col.table
            if table_ref:
                table_ref_lower = table_ref.lower()
                # Resolve alias to real table name
                real_table = table_aliases.get(table_ref_lower)
                if real_table:
                    columns.add(f"{real_table}.{col_name_lower}")
                else:
                    # Could be a CTE reference or unresolved
                    columns.add(f"{table_ref_lower}.{col_name_lower}")
            else:
                # Unqualified column - try to attribute later or leave unqualified
                columns.add(col_name_lower)

    return tables, columns


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
