import json
import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from openai import OpenAI

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL,
    AUTOLINK_MAX_TURNS,
    VS_RETRIEVE_TOP_M,
    ENABLE_IDENTIFIER_INJECTION,
    ENABLE_FAMILY_EXPANSION,
)
from src.sqlite_executor import SQLiteExecutor
from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# LinkedSchema — tracks the current S_linked
# ---------------------------------------------------------------------------

@dataclass
class LinkedSchema:
    """Tracks the current set of linked schema elements."""
    tables: set[str] = field(default_factory=set)
    columns: dict[str, set[str]] = field(default_factory=dict)  # table -> {columns}

    def add(self, specs: str) -> str:
        """Parse 'table.column; table.column; ...' and add to linked set."""
        added = []
        for spec in specs.split(";"):
            spec = spec.strip()
            if not spec:
                continue
            if "." in spec:
                table, col = spec.split(".", 1)
                table = table.strip()
                col = col.strip()
                self.tables.add(table)
                if table not in self.columns:
                    self.columns[table] = set()
                self.columns[table].add(col)
                added.append(f"{table}.{col}")
            else:
                self.tables.add(spec.strip())
                added.append(spec.strip())
        if added:
            return f"Added to linked schema: {', '.join(added)}"
        return "No valid schema elements to add."

    def remove(self, specs: str) -> str:
        """Parse 'table.column; table; ...' and remove from linked set."""
        removed = []
        for spec in specs.split(";"):
            spec = spec.strip()
            if not spec:
                continue
            if "." in spec:
                table, col = spec.split(".", 1)
                table = table.strip()
                col = col.strip()
                if table in self.columns and col in self.columns[table]:
                    self.columns[table].discard(col)
                    removed.append(f"{table}.{col}")
                    if not self.columns[table]:
                        del self.columns[table]
                        self.tables.discard(table)
            else:
                table = spec.strip()
                if table in self.tables:
                    self.tables.discard(table)
                    self.columns.pop(table, None)
                    removed.append(table)
        if removed:
            return f"Removed from linked schema: {', '.join(removed)}"
        return "No matching elements found to remove."

    def to_mschema(self, ddl_data: dict) -> str:
        """Format current linked schema in M-Schema format."""
        if not self.tables:
            return "(empty schema)"

        parts = []
        for table in sorted(self.tables):
            cols = self.columns.get(table, set())
            table_ddl = ddl_data.get(table, {})
            all_columns = table_ddl.get("columns", [])

            linked_cols = []
            for col_name, col_type in all_columns:
                if col_name in cols:
                    linked_cols.append(f"  ({col_name}, {col_type})")

            if linked_cols:
                parts.append(f"# Table: {table} [\n" + "\n".join(linked_cols) + "\n]")
            else:
                parts.append(f"# Table: {table} (no columns linked yet)")

        return "\n".join(parts)

    def to_result(self) -> dict:
        """Convert to evaluation-compatible format."""
        all_columns = []
        for table, cols in self.columns.items():
            for col in cols:
                all_columns.append(f"{table}.{col}")
        return {
            "tables": sorted(self.tables),
            "columns": sorted(all_columns),
        }


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an autonomous database schema linking expert. Your task is to identify \
which tables and columns from a SQLite database are needed to answer a given question.

You do NOT see the full database schema. Instead, you explore it iteratively using tools.

## Available Actions

1. **explore_schema** — Execute a read-only SQL query on the database to discover schema structure.
   Examples: `PRAGMA table_info("table_name")`, `SELECT DISTINCT col FROM table LIMIT 5`, \
`SELECT * FROM pragma_table_info('table') WHERE name LIKE '%date%'`

2. **retrieve_schema** — Semantic search for columns matching a natural language description.
   The query should describe what column you're looking for.
   Examples: "column storing the total price of an order", "date when the customer was created"

3. **verify_schema** — Execute a draft SQL query to test if the current linked schema is sufficient.
   If it fails with "no such column" or "no such table", that tells you exactly what's missing.
   Write a minimal query that uses the columns you think are needed.

4. **add_schema** — Add table.column pairs to the linked schema. Format: "table.col1; table.col2"

5. **match_literal** — Search for a literal value (e.g. "Fresno", "EUR", "2024") across all text \
columns. Returns which columns contain that value. Great for finding the right filter column.

6. **remove_schema** — Remove table.column pairs from the linked schema if they are not needed.
   Format: "table.col1; table.col2". Use "table" (without column) to remove entire table.

7. **stop_action** — Finish the schema linking process. Use this when you believe the linked \
schema contains all necessary tables and columns.

## Strategy
1. Review the candidate columns retrieved below — they are suggestions, NOT confirmed
2. Think about what tables and columns might be needed for the question
3. Use add_schema to add elements you are confident about
4. Use explore_schema to inspect table structures (PRAGMA table_info)
5. Use retrieve_schema to find semantically related columns you might be missing
6. Use verify_schema to write a draft SQL query — it will auto-extract all referenced tables and \
columns into the linked schema, AND execute the query to verify correctness. THIS IS THE MOST \
POWERFUL TOOL — write SQL that uses all the columns you think are needed!
7. Use match_literal if the question mentions specific values (names, codes, statuses)
8. Use remove_schema to prune elements that turned out to be irrelevant
9. Use stop_action when confident the schema is complete

IMPORTANT: Be selective. Only add tables/columns that are actually needed for the query. \
Quality over quantity — precision matters as much as recall.

⚠️ CRITICAL RULES (failure to follow = empty result, evaluation = 0):
- You MUST commit elements to the linked schema before stopping. Use either:
  - `add_schema(...)` to explicitly add table.column pairs, OR
  - `verify_schema(...)` which auto-extracts tables and columns from your SQL
- DO NOT call `stop_action` if the linked schema is empty.
- BEFORE calling `stop_action`, glance at "Current Linked Schema" below — if it says \
"(empty schema)", you MUST first call `add_schema` or `verify_schema`.
- If the database has views (virtual tables), they will be auto-expanded to base tables \
when added — don't worry about them.

📋 PATTERNS TO WATCH FOR (these are common sources of missed schema):

1. **LINKAGE / JOIN tables** — If the question requires combining 2 entities (e.g. \
"players in matches"), look for a separate junction table like `player_match`, \
`order_items`, `*_xref`. The query will FAIL without it.

2. **IDENTIFIER columns** — For each table you commit, also include its identifier \
column(s). Look for names like `*_id`, `*_number`, `*_code`, `*_uuid`. Even if the \
question doesn't mention them explicitly, they are needed for joins and output.

3. **ALL columns of a name-family** — If a table has indexed columns like \
`home_player_1`, `home_player_2`, … `home_player_11`, the question likely needs ALL of them. \
Don't just commit one — check `PRAGMA table_info` or `INFORMATION_SCHEMA.COLUMNS` \
and add the full family.

4. **TIME / DATE columns** — Questions with "when", "duration", "lifespan", \
"recent", "before/after" need explicit date columns. Search for them via \
`retrieve_schema("date column for X")` if not obvious.

5. **AGGREGATE-input columns** — If the question says "average sales", "total spend", \
"max games", you need both the numeric column (`price`, `amount`, `g`) AND \
the grouping/key column (e.g. `customer_id`).

6. **Multi-table queries** — If `gt_tables` are more than 3, you likely need \
linkage tables. Be EXHAUSTIVE in `retrieve_schema` calls before stopping.

## Database: {db_name} (SQLite)
## All tables in this database: {table_list}

## Candidate columns from initial retrieval (NOT yet in linked schema):
{initial_candidates}

## Current Linked Schema (M-Schema format):
{linked_schema}
"""

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI format)
# ---------------------------------------------------------------------------

AUTOLINK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "explore_schema",
            "description": "Execute a read-only SQL query on the database for schema exploration. "
                           "Use PRAGMA table_info, SELECT DISTINCT, sample queries, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "SQL query to execute (read-only)",
                    }
                },
                "required": ["sql_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_schema",
            "description": "Semantic search for database columns matching a natural language description. "
                           "Returns the most relevant columns with their table, type, and sample values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nl_query": {
                        "type": "string",
                        "description": "Natural language description of the column(s) you're looking for",
                    }
                },
                "required": ["nl_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_schema",
            "description": "Execute a draft SQL query to test if the current schema is sufficient. "
                           "Error messages reveal missing tables/columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "A minimal SQL query using the columns you think are needed",
                    }
                },
                "required": ["sql_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_schema",
            "description": "Add table.column pairs to the linked schema. "
                           "Must be paired with another action or stop_action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "schemas": {
                        "type": "string",
                        "description": "Semicolon-separated table.column pairs, e.g. 'orders.order_id; customers.name'",
                    }
                },
                "required": ["schemas"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "match_literal",
            "description": "Search for a literal value across all text columns in the database. "
                           "Returns which columns contain the value. "
                           "Use when the question mentions specific names, codes, or statuses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "literal": {
                        "type": "string",
                        "description": "The literal value to search for, e.g. 'Fresno', 'EUR', 'delivered'",
                    }
                },
                "required": ["literal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_schema",
            "description": "Remove table.column pairs from the linked schema that are not needed. "
                           "Use 'table' (without column) to remove an entire table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "schemas": {
                        "type": "string",
                        "description": "Semicolon-separated table.column pairs to remove, e.g. 'orders.shipping_date; payments'",
                    }
                },
                "required": ["schemas"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_action",
            "description": "Finish the schema linking process. "
                           "Use when the linked schema contains all necessary tables and columns.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


# ---------------------------------------------------------------------------
# View expansion (Fix 3)
# ---------------------------------------------------------------------------

def _expand_views(specs: str, executor: SQLiteExecutor, ddl_data: dict) -> tuple[str, list[str]]:
    """If any spec references a SQL view, replace it with underlying base tables.

    Returns: (expanded_specs, info_messages)
    """
    info = []
    parts = []
    known_tables_lower = {t.lower(): t for t in ddl_data.keys()}

    for spec in specs.split(";"):
        spec = spec.strip()
        if not spec:
            continue

        if "." in spec:
            table, col = spec.split(".", 1)
            table = table.strip()
        else:
            table = spec.strip()
            col = None

        view_sql = executor.get_view_definition(table)
        if view_sql is None:
            parts.append(spec)
            continue

        # Parse the view definition to find base tables
        try:
            parsed = sqlglot.parse(view_sql, read="sqlite")
        except sqlglot.errors.ParseError:
            parts.append(spec)
            continue

        base_tables = set()
        for stmt in parsed:
            if stmt is None:
                continue
            for t in stmt.find_all(exp.Table):
                name = t.name
                if not name:
                    continue
                name_lower = name.lower()
                # Skip the view itself
                if name_lower == table.lower():
                    continue
                # Skip if this is also a view (nested views — recurse one level)
                if executor.get_view_definition(name) is not None:
                    continue
                if name_lower in known_tables_lower:
                    base_tables.add(known_tables_lower[name_lower])

        if base_tables:
            for bt in base_tables:
                parts.append(bt)
            info.append(f"Expanded view '{table}' → {sorted(base_tables)}")
        else:
            parts.append(spec)

    return "; ".join(parts), info


# ---------------------------------------------------------------------------
# SQL → schema extraction (task alignment from AT&T paper)
# ---------------------------------------------------------------------------

def _extract_schema_from_sql(sql: str, known_tables: set[str], dialect: str = "sqlite") -> list[str]:
    """Parse a draft SQL and extract table.column pairs that exist in the DB.

    Returns list of 'table.column' strings for auto-addition to LinkedSchema.
    Catches all sqlglot errors (ParseError, TokenError) — returns [] gracefully.
    """
    # Try platform-native dialect first, fall back to sqlite, then None
    for read_dialect in (dialect, "sqlite", None):
        try:
            parsed = sqlglot.parse(sql, read=read_dialect)
            break
        except Exception:
            continue
    else:
        return []

    table_aliases: dict[str, str] = {}  # alias -> real table
    cte_names: set[str] = set()
    results: list[str] = []

    for stmt in parsed:
        if stmt is None:
            continue

        # Collect CTE names
        for cte in stmt.find_all(exp.CTE):
            if cte.alias:
                cte_names.add(cte.alias.lower())

        # Collect tables + aliases
        for table in stmt.find_all(exp.Table):
            name = table.name
            if not name:
                continue
            name_lower = name.lower()
            if name_lower in cte_names:
                continue
            # Resolve to known table (case-insensitive)
            resolved = None
            for kt in known_tables:
                if kt.lower() == name_lower:
                    resolved = kt
                    break
            if resolved:
                alias = table.alias
                if alias:
                    table_aliases[alias.lower()] = resolved
                table_aliases[name_lower] = resolved

        # Collect columns
        for col in stmt.find_all(exp.Column):
            col_name = col.name
            if not col_name:
                continue
            table_ref = col.table
            if table_ref:
                real_table = table_aliases.get(table_ref.lower())
                if real_table:
                    results.append(f"{real_table}.{col_name}")
            # No table ref — skip (ambiguous)

    return results


# ---------------------------------------------------------------------------
# Auto-Identifier Injection
# ---------------------------------------------------------------------------

# Regex for identifier-looking columns: ends in _id, _number, _code, _uuid,
# or is exactly "id" / "code", or like "team_1" / "*_1..9" (family columns).
_IDENTIFIER_PATTERN = re.compile(
    r"(?:^id$|^code$|_id$|_number$|_code$|_uuid$|_key$|_pk$|_fk$)",
    re.IGNORECASE,
)


def _is_identifier_col(col_name: str) -> bool:
    return bool(_IDENTIFIER_PATTERN.search(col_name))


def _inject_identifiers(
    linked_schema: "LinkedSchema",
    ddl_data: dict,
    vector_store,
) -> list[str]:
    """For every table in linked_schema, auto-add its identifier columns.

    Adds directly to `columns` (full schema, visible to SQL gen).
    Returns list of added 'table.col' strings.
    """
    added = []
    for table in list(linked_schema.tables):
        canonical = table
        for t in ddl_data.keys():
            if t.lower() == table.lower():
                canonical = t
                break
        table_info = ddl_data.get(canonical, {})
        for col_name, _ in table_info.get("columns", []):
            if "." in col_name:
                continue
            if not _is_identifier_col(col_name):
                continue
            cur_cols = linked_schema.columns.get(table, set())
            if col_name in cur_cols:
                continue
            linked_schema.tables.add(table)
            linked_schema.columns.setdefault(table, set()).add(col_name)
            vector_store.mark_excluded(table, col_name)
            added.append(f"{table}.{col_name}")
    return added


# ---------------------------------------------------------------------------
# Column Family Expansion
# ---------------------------------------------------------------------------

# Detects a numeric-suffix family: "home_player_1" -> ("home_player_", "1")
# Also catches "col1" / "team_a" — we only expand numeric.
_FAMILY_PATTERN = re.compile(r"^(.+?)([_]?)(\d+)$")


def _family_prefix(col_name: str) -> str | None:
    """Return the prefix (without numeric suffix) if column looks like family member.

    'home_player_1' -> 'home_player_'
    'team_1'        -> 'team_'
    'col1'          -> 'col'
    'colA'          -> None  (not numeric)
    """
    m = _FAMILY_PATTERN.match(col_name)
    if not m:
        return None
    return m.group(1) + m.group(2)


def _expand_column_families(
    linked_schema: "LinkedSchema",
    ddl_data: dict,
    vector_store,
    proactive: bool = True,
    family_min_size: int = 3,
) -> list[str]:
    """For columns with numeric suffix, expand to the full family.

    Two modes:
    - Reactive: if `match.home_player_1` was committed → add `home_player_2..11`.
    - Proactive (default): for every table in linked_schema, discover ALL families
      (≥`family_min_size` members) in its DDL and add the whole group.

    Returns list of added 'table.col' strings.
    """
    added = []
    for table in list(linked_schema.tables):
        # Find canonical case from DDL
        canonical = table
        for t in ddl_data.keys():
            if t.lower() == table.lower():
                canonical = t
                break
        table_info = ddl_data.get(canonical, {})
        all_table_cols = {c for c, _ in table_info.get("columns", []) if "." not in c}

        committed = linked_schema.columns.get(table, set())

        # ---- Reactive: prefixes from currently-committed columns ----
        reactive_prefixes: set[str] = set()
        for col in committed:
            p = _family_prefix(col)
            if p is not None:
                reactive_prefixes.add(p)

        # ---- Proactive: discover families from DDL columns alone ----
        proactive_prefixes: set[str] = set()
        if proactive:
            prefix_counts: dict[str, int] = {}
            for c in all_table_cols:
                p = _family_prefix(c)
                if p is not None:
                    prefix_counts[p] = prefix_counts.get(p, 0) + 1
            for p, count in prefix_counts.items():
                if count >= family_min_size:
                    proactive_prefixes.add(p)

        all_prefixes = reactive_prefixes | proactive_prefixes

        # For each prefix, find ALL members in the table
        for prefix in all_prefixes:
            family = []
            for c in all_table_cols:
                m = _FAMILY_PATTERN.match(c)
                if m and m.group(1) + m.group(2) == prefix:
                    family.append(c)
            # Reactive families: need 2+ members; proactive: family_min_size
            min_size = 2 if prefix in reactive_prefixes else family_min_size
            if len(family) < min_size:
                continue
            for c in family:
                if c in committed:
                    continue
                linked_schema.columns.setdefault(table, set()).add(c)
                vector_store.mark_excluded(table, c)
                added.append(f"{table}.{c}")
    return added


# ---------------------------------------------------------------------------
# Literal matching (from AT&T paper)
# ---------------------------------------------------------------------------

def _find_literal_in_db(literal: str, executor: SQLiteExecutor, ddl_data: dict) -> str:
    """Search for a literal value across all text columns in the database.

    Returns formatted results showing which columns contain the literal.
    """
    matches = []
    literal_escaped = literal.replace("'", "''")

    for table_name, table_info in ddl_data.items():
        columns = table_info.get("columns", [])
        text_cols = [
            col_name for col_name, col_type in columns
            if "." not in col_name  # skip nested
            and col_type.upper() in ("TEXT", "VARCHAR", "CHAR", "NVARCHAR", "")
        ]
        if not text_cols:
            continue

        # Build a single query checking all text columns at once
        conditions = " OR ".join(
            f'"{c}" LIKE \'%{literal_escaped}%\'' for c in text_cols
        )
        count_exprs = ", ".join(
            f'SUM(CASE WHEN "{c}" LIKE \'%{literal_escaped}%\' THEN 1 ELSE 0 END) AS "{c}"'
            for c in text_cols
        )
        sql = f'SELECT {count_exprs} FROM "{table_name}" WHERE {conditions} LIMIT 1'
        result = executor.execute(sql)

        # Parse result to find which columns matched
        if "[ERROR" in result or "0 rows" in result:
            continue
        # Result has column headers + one row of counts
        for col_name in text_cols:
            # Simple check: look for non-zero count
            # The formatted result contains column names and values
            if f"{col_name}" in result:
                # Re-query individual column for accuracy
                check_sql = (
                    f'SELECT COUNT(*) AS cnt FROM "{table_name}" '
                    f'WHERE "{col_name}" LIKE \'%{literal_escaped}%\''
                )
                check_result = executor.execute(check_sql)
                if "[ERROR" not in check_result and "0 rows" not in check_result:
                    # Extract count
                    lines = check_result.strip().split("\n")
                    if len(lines) >= 3:
                        cnt_str = lines[-1].strip()
                        try:
                            cnt = int(cnt_str.split("|")[0].strip())
                            if cnt > 0:
                                matches.append(f"{table_name}.{col_name} ({cnt} matches)")
                        except (ValueError, IndexError):
                            if cnt_str and cnt_str != "0":
                                matches.append(f"{table_name}.{col_name}")

    if matches:
        return f"Literal '{literal}' found in:\n" + "\n".join(f"  - {m}" for m in matches)
    return f"Literal '{literal}' not found in any text column."


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def execute_tool(
    name: str,
    args: dict,
    executor,  # SQLiteExecutor or SnowflakeExecutor — duck-typed
    vector_store: VectorStore,
    linked_schema: LinkedSchema,
    ddl_data: dict | None = None,
) -> str:
    """Execute an AutoLink tool call and return observation."""
    if name == "explore_schema":
        sql = args.get("sql_query", "")
        if not sql:
            return "[ERROR: explore_schema requires `sql_query` argument]"
        return executor.execute(sql)

    elif name == "retrieve_schema":
        nl_query = args.get("nl_query", "")
        if not nl_query:
            return "[ERROR: retrieve_schema requires `nl_query` argument]"
        results = vector_store.retrieve(nl_query, top_m=VS_RETRIEVE_TOP_M)
        if not results:
            return "No matching columns found."
        lines = []
        for doc in results:
            lines.append(doc.to_mschema())
        return "Retrieved columns:\n" + "\n".join(lines)

    elif name == "verify_schema":
        sql = args.get("sql_query", "")
        if not sql:
            return "[ERROR: verify_schema requires `sql_query` argument]"
        exec_result = executor.execute(sql)

        # Auto-extract schema from the draft SQL (task alignment)
        known_tables = set(ddl_data.keys()) if ddl_data else set()
        # Pick SQL dialect from executor class name
        ex_cls = type(executor).__name__.lower()
        if "snowflake" in ex_cls:
            sql_dialect = "snowflake"
        elif "bigquery" in ex_cls:
            sql_dialect = "bigquery"
        else:
            sql_dialect = "sqlite"
        extracted = _extract_schema_from_sql(sql, known_tables, dialect=sql_dialect)
        auto_added = []
        expansion_info = []
        if extracted:
            specs = "; ".join(extracted)
            # Fix 3: expand views into base tables
            if ddl_data is not None:
                specs, expansion_info = _expand_views(specs, executor, ddl_data)
            linked_schema.add(specs)
            for spec in specs.split(";"):
                spec = spec.strip()
                if "." in spec:
                    t, c = spec.split(".", 1)
                    vector_store.mark_excluded(t.strip(), c.strip())
            auto_added = [s.strip() for s in specs.split(";") if "." in s]

        # Auto-inject identifier columns + expand column families (optional)
        injected = []
        family_added = []
        if ddl_data is not None and auto_added:
            if ENABLE_IDENTIFIER_INJECTION:
                injected = _inject_identifiers(linked_schema, ddl_data, vector_store)
            if ENABLE_FAMILY_EXPANSION:
                family_added = _expand_column_families(linked_schema, ddl_data, vector_store)

        result = exec_result
        if auto_added:
            result += f"\n\n[Auto-linked from SQL: {', '.join(auto_added[:15])}]"
        if expansion_info:
            result += "\n" + "\n".join(expansion_info)
        if injected:
            result += f"\n[Auto-injected identifier cols: {', '.join(injected[:15])}]"
        if family_added:
            result += f"\n[Auto-expanded column family: {', '.join(family_added[:15])}]"
        return result

    elif name == "add_schema":
        specs = args.get("schemas", "")
        if not specs:
            return "[ERROR: add_schema requires `schemas` argument (e.g. 'table.col1; table.col2')]"
        expansion_info = []
        if ddl_data is not None:
            specs, expansion_info = _expand_views(specs, executor, ddl_data)
        result = linked_schema.add(specs)
        # Mark added columns as excluded from future retrieval
        for spec in specs.split(";"):
            spec = spec.strip()
            if "." in spec:
                table, col = spec.split(".", 1)
                vector_store.mark_excluded(table.strip(), col.strip())
        # Auto-inject identifier columns for newly added tables (optional)
        injected = []
        family_added = []
        if ddl_data is not None:
            if ENABLE_IDENTIFIER_INJECTION:
                injected = _inject_identifiers(linked_schema, ddl_data, vector_store)
            if ENABLE_FAMILY_EXPANSION:
                family_added = _expand_column_families(linked_schema, ddl_data, vector_store)
        if expansion_info:
            result += "\n" + "\n".join(expansion_info)
        if injected:
            result += f"\n[Auto-injected identifier cols: {', '.join(injected[:15])}]"
        if family_added:
            result += f"\n[Auto-expanded column family: {', '.join(family_added[:15])}]"
        return result

    elif name == "match_literal":
        if ddl_data is None:
            return "[ERROR: No DDL data available]"
        literal = args.get("literal", "")
        if not literal:
            return "[ERROR: match_literal requires `literal` argument (a value to search for, e.g. 'Fresno')]"
        return _find_literal_in_db(literal, executor, ddl_data)

    elif name == "remove_schema":
        specs = args.get("schemas", "")
        if not specs:
            return "[ERROR: remove_schema requires `schemas` argument]"
        return linked_schema.remove(specs)

    elif name == "stop_action":
        # Block stopping if LinkedSchema is empty — force the agent to commit first
        if not linked_schema.tables:
            return ("[ERROR: Cannot stop with empty linked schema. "
                    "You must call `add_schema` or `verify_schema` first to commit "
                    "at least one table/column. Review the candidate columns above "
                    "and the current empty schema, then add the relevant elements.]")
        return "__STOP__"

    return f"Unknown action: {name}"


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_autolink_agent(
    question: str,
    db_name: str,
    ddl_data: dict,
    executor,  # SQLiteExecutor or SnowflakeExecutor — duck-typed
    vector_store: VectorStore,
    external_knowledge: str | None = None,
    max_turns: int = AUTOLINK_MAX_TURNS,
) -> dict:
    """Run the AutoLink agent on a question.

    Returns: {"tables": [...], "columns": [...], "iterations": int, "tool_calls": [...]}
    """
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    # Step 1: Initial schema retrieval — as candidates, not auto-added
    initial_docs = vector_store.retrieve_initial(question, top_n=20)
    linked_schema = LinkedSchema()

    # Format initial candidates for context (agent decides what to add)
    initial_candidates = "\n".join(doc.to_mschema() for doc in initial_docs) if initial_docs else "(none)"

    # Build messages
    table_list = ", ".join(sorted(ddl_data.keys()))
    system_msg = SYSTEM_PROMPT.format(
        db_name=db_name,
        table_list=table_list,
        initial_candidates=initial_candidates,
        linked_schema=linked_schema.to_mschema(ddl_data),
    )

    user_content = f"Question: {question}"
    if external_knowledge:
        user_content += f"\n\nAdditional context:\n{external_knowledge[:3000]}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]

    all_tool_calls = []
    prompt_tokens_total = 0
    completion_tokens_total = 0

    for iteration in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=AUTOLINK_TOOLS,
            tool_choice="auto",
            temperature=0.0,
        )

        # Track token usage
        if response.usage:
            prompt_tokens_total += response.usage.prompt_tokens or 0
            completion_tokens_total += response.usage.completion_tokens or 0

        msg = response.choices[0].message

        if not msg.tool_calls:
            # No tool calls — treat as implicit stop
            break

        messages.append(msg)
        stop = False

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            try:
                result = execute_tool(fn_name, fn_args, executor, vector_store, linked_schema, ddl_data)
            except Exception as e:
                result = f"[ERROR: tool `{fn_name}` raised {type(e).__name__}: {str(e)[:200]}]"

            all_tool_calls.append({
                "name": fn_name,
                "args": fn_args,
                "result_preview": result[:300] if result != "__STOP__" else "STOP",
            })

            if result == "__STOP__":
                # Force-review: if agent stops early AND linked schema is sparse,
                # nudge it to do at least one more retrieval pass.
                early = iteration < max_turns // 3
                sparse = (len(linked_schema.tables) < 2
                          or sum(len(c) for c in linked_schema.columns.values()) < 4)
                already_nudged = any(c.get("name") == "__nudge__" for c in all_tool_calls)
                if early and sparse and not already_nudged:
                    nudge_text = (
                        f"[NUDGE] You called stop_action after only {iteration + 1} turns "
                        f"with a sparse linked schema (tables={len(linked_schema.tables)}, "
                        f"columns={sum(len(c) for c in linked_schema.columns.values())}). "
                        "Before stopping, please do at least one more `retrieve_schema` "
                        "for any missing aspect of the question (identifier columns, "
                        "join/linkage tables, date columns, aggregate inputs)."
                    )
                    all_tool_calls.append({"name": "__nudge__", "args": {},
                                           "result_preview": "force-review issued"})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": nudge_text,
                    })
                    # Don't stop — let the agent process the nudge
                else:
                    stop = True
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Schema linking process completed.",
                    })
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        if stop:
            break

        # Update system message with current linked schema
        system_msg = SYSTEM_PROMPT.format(
            db_name=db_name,
            table_list=table_list,
            initial_candidates=initial_candidates,
            linked_schema=linked_schema.to_mschema(ddl_data),
        )
        messages[0] = {"role": "system", "content": system_msg}

    # Fix 1: Safety net — if LinkedSchema is empty, fall back to initial top-N candidates
    if not linked_schema.tables and initial_docs:
        print(f"  [WARN] Empty LinkedSchema after agent loop — falling back to initial top-{len(initial_docs)} candidates")
        for doc in initial_docs:
            linked_schema.tables.add(doc.table_name)
            if doc.table_name not in linked_schema.columns:
                linked_schema.columns[doc.table_name] = set()
            linked_schema.columns[doc.table_name].add(doc.column_name)
        all_tool_calls.append({
            "name": "__fallback__",
            "args": {"reason": "empty linked schema"},
            "result_preview": f"Used initial top-{len(initial_docs)} candidates as fallback",
        })

    output = linked_schema.to_result()
    output["iterations"] = iteration + 1 if 'iteration' in dir() else 0
    output["tool_calls"] = all_tool_calls
    output["tokens"] = {
        "prompt": prompt_tokens_total,
        "completion": completion_tokens_total,
        "total": prompt_tokens_total + completion_tokens_total,
    }
    return output
