import json
import re

from openai import OpenAI

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL,
    MAX_AGENT_ITERATIONS,
    SEARCH_TOP_K_TABLES,
    SEARCH_TOP_K_COLUMNS,
)
from src.data_loader import load_external_knowledge
from src.schema_index import SchemaIndex


SYSTEM_PROMPT = """\
You are a database schema expert. Your task is to identify which tables and columns \
from a database are needed to answer a given SQL question.

You have access to the following tools to explore the database schema:
- search_tables: Search for relevant tables by keyword query
- search_columns: Search for relevant columns within a specific table
- get_table_schema: Get the full CREATE TABLE DDL for a specific table

Strategy:
1. First, use search_tables to find tables relevant to the question
2. For promising tables, use get_table_schema to see their full schema
3. Use search_columns to find specific columns if needed
4. When confident, output your final answer

Database: {db_name} (SQLite)
Available tables: {table_list}

When you are ready to give your final answer, respond with a JSON object (and nothing else) in this exact format:
{{"tables": ["table1", "table2"], "columns": ["table1.col1", "table1.col2", "table2.col3"]}}

Use the format table_name.column_name for columns. Only include tables and columns that are directly needed to answer the question.\
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_tables",
            "description": "Search for database tables relevant to a query. Returns table names with relevance scores and column previews.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing what tables you're looking for",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_columns",
            "description": "Search for columns within a specific table. Returns column names, types, and relevance scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to search columns in",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query describing what columns you're looking for",
                    },
                },
                "required": ["table_name", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_schema",
            "description": "Get the full CREATE TABLE DDL statement for a specific table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table",
                    }
                },
                "required": ["table_name"],
            },
        },
    },
]


def execute_tool(index: SchemaIndex, name: str, args: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "search_tables":
        results = index.search_tables(args["query"], top_k=SEARCH_TOP_K_TABLES)
        return json.dumps(results, indent=2)
    elif name == "search_columns":
        results = index.search_columns(args["table_name"], args["query"], top_k=SEARCH_TOP_K_COLUMNS)
        if not results:
            return f"Table '{args['table_name']}' not found or has no columns."
        return json.dumps(results, indent=2)
    elif name == "get_table_schema":
        ddl = index.get_table_schema(args["table_name"])
        if ddl is None:
            return f"Table '{args['table_name']}' not found."
        return ddl
    else:
        return f"Unknown tool: {name}"


def parse_agent_response(text: str) -> dict | None:
    """Try to parse JSON from the agent's final response."""
    # Try to find JSON in the text
    match = re.search(r'\{[^{}]*"tables"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Try parsing the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def run_agent(
    question: str,
    index: SchemaIndex,
    external_knowledge: str | None = None,
) -> dict:
    """Run the entity linking agent on a question.

    Returns: {"tables": [...], "columns": [...], "iterations": int, "tool_calls": [...]}
    """
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    table_list = ", ".join(sorted(index.tables.keys()))
    system_msg = SYSTEM_PROMPT.format(db_name=index.db_name, table_list=table_list)

    user_content = f"Question: {question}"
    if external_knowledge:
        user_content += f"\n\nAdditional context:\n{external_knowledge}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]

    all_tool_calls = []

    for iteration in range(MAX_AGENT_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            # Final answer
            text = msg.content or ""
            parsed = parse_agent_response(text)
            if parsed:
                return {
                    "tables": parsed.get("tables", []),
                    "columns": parsed.get("columns", []),
                    "iterations": iteration + 1,
                    "tool_calls": all_tool_calls,
                }
            # If we couldn't parse, add assistant message and ask to format properly
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": "Please provide your answer as a JSON object with 'tables' and 'columns' keys.",
            })
            continue

        # Process tool calls
        messages.append(msg)
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            result = execute_tool(index, fn_name, fn_args)
            all_tool_calls.append({
                "name": fn_name,
                "args": fn_args,
                "result_preview": result[:200],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # Max iterations reached
    return {
        "tables": [],
        "columns": [],
        "iterations": MAX_AGENT_ITERATIONS,
        "tool_calls": all_tool_calls,
        "error": "Max iterations reached without final answer",
    }
