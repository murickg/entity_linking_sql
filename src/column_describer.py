"""LLM-based column description generator (from AT&T metadata extraction paper).

Sends profiling stats for all columns of a table to the LLM in a single call,
receives short semantic descriptions. Descriptions are cached per database.
"""

import json
from pathlib import Path

from openai import OpenAI

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL,
    DESCRIPTION_CACHE_DIR,
)

_SYSTEM_PROMPT = """\
You are a database documentation expert. Given a table name, its columns with \
types, profiling statistics, and sample values, generate a short (1-2 sentence) \
description for each column explaining what it stores and its practical meaning.

Rules:
- Be concise: 1-2 sentences max per column.
- Focus on the MEANING, not just restate the column name.
- Use profiling clues: distinct count, null %, value range, sample values.
- If a column looks like a foreign key (ends with _id, low distinct count), say so.
- If sample values reveal a format (dates, codes, JSON), mention it.
- Return ONLY valid JSON: {"column_name": "description", ...}
- Keys must match the column names exactly (case-sensitive).
- No markdown, no extra text — just the JSON object.
"""


def _build_column_profile_text(
    table_name: str,
    columns: list[dict],
) -> str:
    """Format column profiles for the LLM prompt."""
    lines = [f"Table: {table_name}", ""]
    for col in columns:
        parts = [f"Column: {col['name']}, Type: {col['type']}"]
        if col.get("distinct_count") is not None and col.get("total_count"):
            null_pct = round(100 * (col.get("null_count", 0)) / col["total_count"])
            parts.append(
                f"  Stats: {col['distinct_count']} distinct / {col['total_count']} rows, "
                f"{null_pct}% null"
            )
        if col.get("min_value") is not None:
            parts.append(f"  Range: [{col['min_value']} .. {col['max_value']}]")
        if col.get("sample_values"):
            vals = ", ".join(f'"{v}"' for v in col["sample_values"][:7])
            parts.append(f"  Sample values: [{vals}]")
        lines.append("\n".join(parts))
        lines.append("")
    return "\n".join(lines)


def generate_descriptions_for_table(
    table_name: str,
    columns: list[dict],
) -> dict[str, str]:
    """Call LLM to generate descriptions for all columns of one table.

    Args:
        table_name: Name of the table.
        columns: List of dicts with keys: name, type, distinct_count, null_count,
                 total_count, min_value, max_value, sample_values.

    Returns: {column_name: description_string}
    """
    if not OPENROUTER_API_KEY:
        return {}

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    profile_text = _build_column_profile_text(table_name, columns)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": profile_text},
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON — handle potential markdown wrapping
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]

        return json.loads(content)
    except Exception as e:
        print(f"  [WARN] Failed to generate descriptions for {table_name}: {e}")
        return {}


def generate_all_descriptions(
    db_name: str,
    documents: list,  # list[ColumnDocument]
) -> dict[str, str]:
    """Generate LLM descriptions for all columns in a database, with caching.

    Args:
        db_name: Database name (used for cache key).
        documents: List of ColumnDocument objects (already profiled).

    Returns: {table.column: description} dict.
    """
    cache_path = DESCRIPTION_CACHE_DIR / f"{db_name}.json"

    # Check cache
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # Validate: all current columns should be in cache
            current_keys = {f"{d.table_name}.{d.column_name}" for d in documents}
            if current_keys.issubset(set(cached.keys())):
                return cached
        except (json.JSONDecodeError, Exception):
            pass

    # Group documents by table
    tables: dict[str, list[dict]] = {}
    for doc in documents:
        if doc.table_name not in tables:
            tables[doc.table_name] = []
        tables[doc.table_name].append({
            "name": doc.column_name,
            "type": doc.col_type,
            "distinct_count": doc.distinct_count,
            "null_count": doc.null_count,
            "total_count": doc.total_count,
            "min_value": doc.min_value,
            "max_value": doc.max_value,
            "sample_values": doc.sample_values,
        })

    all_descriptions: dict[str, str] = {}
    total_tables = len(tables)

    for i, (table_name, cols) in enumerate(tables.items()):
        print(f"  Generating descriptions for {table_name} ({i+1}/{total_tables})...")
        descs = generate_descriptions_for_table(table_name, cols)
        for col_name, desc in descs.items():
            all_descriptions[f"{table_name}.{col_name}"] = desc

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_descriptions, f, indent=2, ensure_ascii=False)

    print(f"  Cached {len(all_descriptions)} descriptions to {cache_path}")
    return all_descriptions
