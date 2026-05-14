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
                if "." in col_name:
                    continue
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

def _extract_schema_from_sql(sql: str, known_tables: set[str]) -> list[str]:
    """Parse a draft SQL and extract table.column pairs that exist in the DB.

    Returns list of 'table.column' strings for auto-addition to LinkedSchema.
    """
    try:
        parsed = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.ParseError:
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
    executor: SQLiteExecutor,
    vector_store: VectorStore,
    linked_schema: LinkedSchema,
    ddl_data: dict | None = None,
) -> str:
    """Execute an AutoLink tool call and return observation."""
    if name == "explore_schema":
        return executor.execute(args["sql_query"])

    elif name == "retrieve_schema":
        results = vector_store.retrieve(args["nl_query"], top_m=VS_RETRIEVE_TOP_M)
        if not results:
            return "No matching columns found."
        lines = []
        for doc in results:
            lines.append(doc.to_mschema())
        return "Retrieved columns:\n" + "\n".join(lines)

    elif name == "verify_schema":
        sql = args["sql_query"]
        exec_result = executor.execute(sql)

        # Auto-extract schema from the draft SQL (task alignment)
        known_tables = set(ddl_data.keys()) if ddl_data else set()
        extracted = _extract_schema_from_sql(sql, known_tables)
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

        result = exec_result
        if auto_added:
            result += f"\n\n[Auto-linked from SQL: {', '.join(auto_added[:15])}]"
        if expansion_info:
            result += "\n" + "\n".join(expansion_info)
        return result

    elif name == "add_schema":
        # Fix 3: expand views into their base tables
        specs = args["schemas"]
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
        if expansion_info:
            result += "\n" + "\n".join(expansion_info)
        return result

    elif name == "match_literal":
        if ddl_data is None:
            return "[ERROR: No DDL data available]"
        return _find_literal_in_db(args["literal"], executor, ddl_data)

    elif name == "remove_schema":
        return linked_schema.remove(args["schemas"])

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
    executor: SQLiteExecutor,
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

    for iteration in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=AUTOLINK_TOOLS,
            tool_choice="auto",
        )

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

            result = execute_tool(fn_name, fn_args, executor, vector_store, linked_schema, ddl_data)

            all_tool_calls.append({
                "name": fn_name,
                "args": fn_args,
                "result_preview": result[:300] if result != "__STOP__" else "STOP",
            })

            if result == "__STOP__":
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
    return output
