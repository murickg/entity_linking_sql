"""SQL Generator — multi-candidate pipeline (mirrors official AutoLink).

Approach (per question):
  1. Generate N SQL candidates with DeepSeek-R1 (reasoning model)
  2. Execute each candidate; if failed, revise with the error message (loop)
  3. Cluster surviving candidates by execution result (compare_pandas_table)
  4. Select largest cluster's representative; if ties, use LLM pairwise vote
"""

import math
import re
from typing import Any

import pandas as pd
from openai import OpenAI

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL, SQL_GEN_MODEL


# ===========================================================================
# Prompts (adapted from official AutoLink run/config.py)
# ===========================================================================

SQLITE_OPTIMIZATION = """\
SQLite Optimization Strategies:

- Decimal Precision:
    - If user does not specify the precision, use `ROUND(value, 4)` to round to four decimal places.
- Aggregation:
    - When using `ORDER BY xxx DESC`, add `NULLS LAST` to exclude null records: `ORDER BY xxx DESC NULLS LAST`.
- String Matching:
    - Don't directly match strings if uncertain. Use LOWER for fuzzy queries: `WHERE LOWER(str) LIKE LOWER('%target%')`.
- Date Handling:
    - For time-related queries, given the variety of formats, avoid time-converting functions unless certain of the format.
- Performance Tips:
    - Materialize complex expressions in CTEs to avoid recomputation.
    - Quote all table and column names in double quotes when needed.
    - Filter early using WHERE clauses before applying aggregations.
"""

BIGQUERY_OPTIMIZATION = """\
BigQuery Optimization Strategies:

- ⚠️ MANDATORY: Use the EXACT fully-qualified table name shown in the schema below \
(e.g. `bigquery-public-data.google_analytics_sample.ga_sessions_*`).
- ⛔ NEVER use placeholders like `your_project.your_dataset.X` or `project.dataset.X` — \
they will fail at runtime with "Invalid project ID" or "Access Denied".
- Copy the table name VERBATIM from the schema context, including the wildcard `*` suffix.

- Wildcard / Partitioned Tables:
    - For partitioned tables like `ga_sessions_*`, use wildcards with _TABLE_SUFFIX filter:
      ```
      FROM `project.dataset.ga_sessions_*`
      WHERE _TABLE_SUFFIX BETWEEN '20170101' AND '20171231'
      ```
    - Avoid full scans on wildcard tables without _TABLE_SUFFIX filtering.

- Nested / STRUCT / ARRAY columns:
    - Use UNNEST for arrays: SELECT t.*, h FROM `table` t, UNNEST(t.hits) h
    - Access STRUCT fields: t.totals.totalTransactionRevenue
    - For repeated STRUCT: UNNEST(hits) AS h, then h.product.productSKU

- Date Handling:
    - Use EXTRACT(YEAR FROM date), FORMAT_DATE('%Y-%m', date), DATE_DIFF.
    - DO NOT use strftime, julianday — SQLite-specific.

- String Matching:
    - Use LOWER(str) LIKE LOWER('%target%') for fuzzy queries.
    - REGEXP_CONTAINS(col, r'regex') for complex patterns.

- Decimal Precision:
    - If user does not specify precision, use ROUND(value, 4).

- Performance:
    - Always include _TABLE_SUFFIX filter on wildcard tables.
    - Filter early with WHERE before aggregations.
    - Materialize complex expressions in CTEs.

- BigQuery dialect uses BACKTICKS for identifiers (not double quotes).
"""


SNOWFLAKE_OPTIMIZATION = """\
Snowflake Optimization Strategies:

- Column Naming:
    - Snowflake folds unquoted column names to UPPERCASE.
    - To preserve casing, quote column names in double quotes, e.g. p."user_id".
    - Use full table name: "DATABASE"."SCHEMA"."TABLE" or DATABASE.SCHEMA.TABLE.
- Partitioned Tables:
    - For tables differing only by date suffix with same structure, use UNION ALL (no wildcards).
- VARIANT / JSON columns:
    - Use colon notation: t."col":"field"::STRING
    - For arrays: SELECT f.value FROM "T", LATERAL FLATTEN(input => t."json_col") f
    - DO NOT use JSON_EXTRACT or JSON_EACH (these are SQLite).
- Decimal Precision:
    - If user does not specify precision, use ROUND(value, 4).
- String Matching:
    - Use LOWER(str) LIKE LOWER('%target%') for fuzzy queries.
    - For complex patterns: REGEXP_LIKE(col, 'regex').
- Date Handling:
    - Use TO_DATE / DATE_PART / DATEADD / DATEDIFF.
    - DO NOT use strftime, julianday — SQLite-specific.
- String Functions:
    - Use UPPER/LOWER, SUBSTR, SPLIT_PART, REGEXP_LIKE.
    - DO NOT use PRINTF — use TO_CHAR or LPAD/RPAD.
- Hexadecimal:
    - For hex amount, use LTRIM(amount_hex, '0') then concatenate '0x' prefix for TRY_CAST.
- Geospatial:
    - ST_GEOMPOINT(lng, lat), ST_DISTANCE(g1, g2), ST_WITHIN, ST_CONTAINS, ST_GEOGFROMWKB.
- Performance:
    - Materialize complex expressions in CTEs.
    - Quote all table/column names in double quotes.
    - Filter early with WHERE before aggregations.
"""


SQL_GENERATION_PROMPT = """\
You are a professional data engineer skilled in translating complex natural language \
questions into accurate and efficient SQL queries. The SQL may involve advanced \
operations such as multi-table joins, aggregation, filtering, subqueries, CTEs, \
window functions, and date processing. You must complete this task and generate SQL \
in {DIALECT} dialect.

Question:
{QUESTION}

Database Schema and External Knowledge:
{PROMPT}

🔍 Step-by-Step Reasoning

**Step 1: Deeply Understand the Question Intent**
1. Clearly summarize the core objective of the question.
2. Decompose the question into well-defined sub-problems.
3. Explicitly list out all operations required: aggregation, filtering, sorting, joins, \
date manipulations, ranking, window functions, etc.

**Step 2: Identify Relevant Tables and Columns**
1. Precisely identify relevant tables and columns required to answer the question based \
on clear evidence.
2. Clearly specify any explicit constraints from the question (dates, numerical \
thresholds, text patterns).
3. Highlight any implicit constraints or potential ambiguities that need verification.

**Step 3: Design the SQL Query Structure**
Clearly outline the planned SQL structure:
* Specify if CTEs (WITH clause) are required. Follow syntax rigorously (`name AS (SELECT ...)`).
* Clearly define SELECT, FROM, JOIN conditions, WHERE filters, GROUP BY/HAVING \
conditions, ORDER BY/LIMIT operations.
* Specify exact operations (UNNEST, LATERAL FLATTEN, ST_DISTANCE, window functions, etc.) needed.

**Step 4: Logical Validation (Critical)**
* Before generating the final SQL, explicitly verify that your designed SQL fully meets \
every constraint (explicit and implicit) mentioned in the original question.
* Clearly explain why your SQL logic is correct and how it satisfies the user's intent \
comprehensively.

**Step 5: Write the Final SQL Query**
* Ensure accurate parentheses pairing and commas placement.
* Annotate your SQL clearly using comments to explain each part.

⚙️ Apply Optimization Strategies
When writing the SQL query, consider the following optimization strategies:
{DIALECT_OPTIMIZATION}

- Execution result content:
    - When asked something without stating name or id, return both of them. \
e.g. Which products ...? The answer should include product_name and product_id.
    - Make sure the query result definitely includes what needs to be involved in the \
question. The result may have more than required, but it must not have less.

📤 Output Format
In addition to outputting other information, you also need to return the generated SQL \
query in the following format:
```sql
Your sql query
```
Make sure that all SQLs are contained within ```sql``` blocks and the LAST ```sql``` \
contains the final complete SQL in your output.
"""


REVISE_ERROR_PROMPT = """\
You are a professional data engineer skilled in translating complex natural language \
questions into accurate and efficient SQL queries.
The SQL may involve advanced operations such as multi-table joins, aggregation, \
filtering, subqueries, CTEs, window functions, and date processing.
You must complete this task through **multiple reasoning rounds** and generate SQL \
in {DIALECT} dialect.

Database Schema and External Knowledge:
{PROMPT}

Question:
{QUESTION}

SQL Query:
{SQL}

❌ The SQL you generated encountered an error during execution.

**Error Message:**
{ERROR_MESSAGE}

Please help analyze the SQL and identify the root cause of the failure by following \
this structured checklist:

🔍 [1] Error Type Detection
- Based on the error message, determine the type of issue:
- Syntax error (e.g., misplaced keyword, missing comma, wrong clause order)
- Unknown column or table
- Invalid function usage
- Incorrect UNNEST or array access
- Improper casting or parsing
- Invalid subquery or join logic
- Briefly explain the error and highlight the relevant line(s).

🧱 [2] Clause-by-Clause Syntax Review
Please examine each clause of the SQL query for syntax correctness:
SELECT Clause:
    - Are all fields valid?
    - Are nested fields accessed correctly (e.g., col.key, value:int_value)?
    - Are aliases and expressions properly defined?
FROM Clause:
    - Is the table name correct (including full path for Snowflake)?
    - If wildcard or partitioned tables are used, are they handled properly?
    - Are commas or joins misplaced?
WHERE Clause:
    - Are boolean conditions well-formed?
    - Is the logic clear (no dangling AND/OR)?
    - Are fields used here actually defined in the schema?
JOINs or LATERAL FLATTENs (if any):
    - Are all array fields unnested/flattened before access?
    - Are join conditions properly specified?
GROUP BY / HAVING / ORDER BY:
    - Are aggregation fields valid?
    - Does SELECT contain only grouped or aggregated expressions?

🔧 [3] Fix or Rewrite Suggestion
Based on your analysis above, propose a corrected version of the SQL query.
Or, describe how the query can be restructured to fix the issue.

⚙️ Apply Optimization Strategies
When writing the SQL query, consider the following optimization strategies:
{DIALECT_OPTIMIZATION}

### Output Format:
```sql
Your fixed sql query
```
Make sure that the LAST ```sql``` contains the final complete SQL in your output.
"""


SQL_SELECTION_PROMPT = """\
### {DIALECT} SQL tables, with their properties:
{Database_Schema}
### Answer the question by {DIALECT} SQL query only and with no explanation.
### Question: {Question}
### Two SQLs, the results of execution and time of execution will be given.
### It is unreasonable if all rows are null.
### Select the best SQL query to answer the question correctly from the given two SQLs:
### SQL1:
{sql1}
### Execution result of the SQL1 (First 1000 rows limit 10,000 characters):
{re1}

### SQL2:
{sql2}
### Execution result of the SQL2 (First 1000 rows limit 10,000 characters):
{re2}

Output format:
Just output tag "SQL1" OR "SQL2", don't contain any external explanation.
"""


# ===========================================================================
# Helper functions
# ===========================================================================

def _dialect_opts(platform: str) -> tuple[str, str]:
    if platform == "snowflake":
        return "Snowflake", SNOWFLAKE_OPTIMIZATION
    if platform == "bigquery":
        return "BigQuery", BIGQUERY_OPTIMIZATION
    return "SQLite", SQLITE_OPTIMIZATION


def extract_sql(text: str) -> str:
    """Extract SQL from LLM response. Robust to multiple formats."""
    if not text:
        return ""
    # Try LAST ```sql ... ``` block (preferred)
    matches = re.findall(r"```sql\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    # Try LAST ``` ... ``` block (any language)
    matches = re.findall(r"```\s*(?:\w+)?\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        candidate = matches[-1].strip()
        # Make sure it looks like SQL
        if re.search(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b", candidate, re.IGNORECASE):
            return candidate
    # Fallback: search for SELECT/WITH ... up to ; or EOF
    m = re.search(r"((?:WITH|SELECT)\b[\s\S]+?)(?:\n\s*\n|;\s*$|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(";").strip()
    # Nothing found — return original (will fail at exec, but with logged content)
    return text.strip()


def _client() -> OpenAI:
    return OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)


# ===========================================================================
# Single-shot SQL generation
# ===========================================================================

def generate_sql(
    question: str,
    schema_prompt: str,
    external_knowledge: str | None = None,
    temperature: float = 0.0,
    platform: str = "sqlite",
    model: str | None = None,
) -> dict:
    """Generate one SQL candidate for a question."""
    dialect, opt_block = _dialect_opts(platform)
    full_schema = schema_prompt
    if external_knowledge:
        full_schema += f"\n\nExternal knowledge:\n{external_knowledge[:5000]}"

    prompt = SQL_GENERATION_PROMPT.format(
        QUESTION=question,
        PROMPT=full_schema,
        DIALECT=dialect,
        DIALECT_OPTIMIZATION=opt_block,
    )

    try:
        response = _client().chat.completions.create(
            model=model or SQL_GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        msg = response.choices[0].message
        raw = msg.content or ""
        # For DeepSeek-R1: if `content` is empty, try `reasoning_content`
        if not raw:
            raw = getattr(msg, "reasoning_content", None) or ""
            if raw:
                print(f"  [INFO] Falling back to reasoning_content ({len(raw)} chars)")
        sql = extract_sql(raw)
        if not sql:
            print(f"  [WARN] Empty SQL extracted. Raw response length: {len(raw)}")
            print(f"  [WARN] Raw preview: {raw[:500]}")
        tokens = {
            "prompt": response.usage.prompt_tokens if response.usage else 0,
            "completion": response.usage.completion_tokens if response.usage else 0,
        }
        tokens["total"] = tokens["prompt"] + tokens["completion"]
        return {"sql": sql, "raw_response": raw, "tokens": tokens, "error": None}
    except Exception as e:
        return {
            "sql": "",
            "raw_response": "",
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "error": str(e),
        }


# ===========================================================================
# Revision
# ===========================================================================

def revise_sql(
    question: str,
    schema_prompt: str,
    failed_sql: str,
    error_message: str,
    external_knowledge: str | None = None,
    platform: str = "sqlite",
) -> dict:
    """Ask LLM to fix a failed SQL given execution error."""
    dialect, opt_block = _dialect_opts(platform)
    full_schema = schema_prompt
    if external_knowledge:
        full_schema += f"\n\nExternal knowledge:\n{external_knowledge[:5000]}"

    prompt = REVISE_ERROR_PROMPT.format(
        PROMPT=full_schema,
        QUESTION=question,
        SQL=failed_sql,
        ERROR_MESSAGE=error_message,
        DIALECT=dialect,
        DIALECT_OPTIMIZATION=opt_block,
    )

    try:
        response = _client().chat.completions.create(
            model=SQL_GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        msg = response.choices[0].message
        raw = msg.content or ""
        if not raw:
            raw = getattr(msg, "reasoning_content", None) or ""
        sql = extract_sql(raw)
        tokens = {
            "prompt": response.usage.prompt_tokens if response.usage else 0,
            "completion": response.usage.completion_tokens if response.usage else 0,
        }
        tokens["total"] = tokens["prompt"] + tokens["completion"]
        return {"sql": sql, "tokens": tokens, "error": None}
    except Exception as e:
        return {"sql": "", "tokens": {"prompt": 0, "completion": 0, "total": 0},
                "error": str(e)}


# ===========================================================================
# Cluster-based selection (from official AutoLink)
# ===========================================================================

def _rows_to_df(rows: list[tuple]) -> pd.DataFrame | None:
    if not rows:
        return None
    return pd.DataFrame(rows)


def _compare_pandas_table(pred: pd.DataFrame, gold: pd.DataFrame, ignore_order: bool = True) -> int:
    """Column-wise comparison from official AutoLink.

    For every column in `gold`, check if there is a matching column in `pred`
    (with optional row-order tolerance + float tolerance 1e-2).
    Returns 1 if all gold columns are matched, else 0.
    """
    tolerance = 1e-2

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=ignore_order):
        if ignore_order_:
            v1, v2 = (
                sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
            )
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            try:
                if pd.isna(a) and pd.isna(b):
                    continue
            except (TypeError, ValueError):
                pass
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True

    t_gold = gold.transpose().values.tolist()
    t_pred = pred.transpose().values.tolist()
    score = 1
    for gold_col in t_gold:
        if not any(vectors_match(gold_col, pred_col) for pred_col in t_pred):
            score = 0
    return score


def _cluster_by_execution(candidates: list[dict]) -> list[list[dict]]:
    """Group candidates by identical execution results (compare_pandas_table)."""
    clusters: list[list[dict]] = []
    for c in candidates:
        if c.get("error") or not c.get("rows"):
            continue
        df_c = _rows_to_df(c["rows"])
        if df_c is None or df_c.empty:
            continue
        matched = False
        for cluster in clusters:
            df_first = _rows_to_df(cluster[0]["rows"])
            if df_first is None:
                continue
            if _compare_pandas_table(df_c, df_first, ignore_order=True):
                cluster.append(c)
                matched = True
                break
        if not matched:
            clusters.append([c])
    return clusters


def _model_vote(
    candidates: list[dict],
    question: str,
    schema_prompt: str,
    platform: str,
    max_rows: int = 100,
    max_chars: int = 5000,
) -> dict:
    """LLM pairwise tie-break between equally-popular clusters."""
    import itertools
    dialect, _ = _dialect_opts(platform)
    scores = {c["candidate_idx"]: 0 for c in candidates}

    for a, b in itertools.combinations(candidates, 2):
        df_a = _rows_to_df(a["rows"])
        df_b = _rows_to_df(b["rows"])
        re1 = str(df_a.iloc[:min(len(df_a), max_rows)])[:max_chars] if df_a is not None else "(no rows)"
        re2 = str(df_b.iloc[:min(len(df_b), max_rows)])[:max_chars] if df_b is not None else "(no rows)"
        prompt = SQL_SELECTION_PROMPT.format(
            Database_Schema=schema_prompt[:20000],
            Question=question,
            DIALECT=dialect,
            sql1=a["sql"],
            sql2=b["sql"],
            re1=re1,
            re2=re2,
        )
        try:
            resp = _client().chat.completions.create(
                model=SQL_GEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            out = (resp.choices[0].message.content or "").lower()
            if "sql1" in out:
                scores[a["candidate_idx"]] += 1
            elif "sql2" in out:
                scores[b["candidate_idx"]] += 1
        except Exception:
            continue

    best_idx = max(scores.items(), key=lambda x: x[1])[0]
    return next(c for c in candidates if c["candidate_idx"] == best_idx)


# ===========================================================================
# Full pipeline
# ===========================================================================

def run_sql_pipeline(
    question: str,
    schema_prompt: str,
    sqlite_path=None,
    sf_executor=None,
    bq_executor=None,
    external_knowledge: str | None = None,
    num_candidates: int = 5,
    max_revisions: int = 2,
    platform: str = "sqlite",
) -> dict:
    """Generate N candidates → revise on error → cluster → select."""
    from src.ex_evaluator import _execute_sql, _execute_snowflake, _execute_bigquery

    def _run(sql):
        if sf_executor is not None:
            return _execute_snowflake(sql, sf_executor)
        if bq_executor is not None:
            return _execute_bigquery(sql, bq_executor)
        return _execute_sql(sqlite_path, sql)

    total_tokens = {"prompt": 0, "completion": 0, "total": 0}
    candidates: list[dict] = []

    # Use varied temperatures across candidates for diversity
    temps = [0.0, 0.3, 0.5, 0.7, 0.9]
    for i in range(num_candidates):
        temp = temps[i % len(temps)]
        gen = generate_sql(question, schema_prompt, external_knowledge,
                           temperature=temp, platform=platform)
        total_tokens["prompt"] += gen["tokens"]["prompt"]
        total_tokens["completion"] += gen["tokens"]["completion"]

        rows, err = _run(gen["sql"]) if gen["sql"] else ([], "no sql")
        candidate = {
            "candidate_idx": i,
            "sql": gen["sql"],
            "error": err,
            "rows": rows,
            "revised": False,
        }

        # Revise loop
        revs = 0
        while err and revs < max_revisions and gen["sql"]:
            rev = revise_sql(question, schema_prompt, gen["sql"], err,
                             external_knowledge, platform=platform)
            total_tokens["prompt"] += rev["tokens"]["prompt"]
            total_tokens["completion"] += rev["tokens"]["completion"]
            if rev["error"] or not rev["sql"]:
                break
            gen["sql"] = rev["sql"]
            rows, err = _run(rev["sql"])
            candidate["sql"] = rev["sql"]
            candidate["error"] = err
            candidate["rows"] = rows
            candidate["revised"] = True
            revs += 1

        candidates.append(candidate)

    total_tokens["total"] = total_tokens["prompt"] + total_tokens["completion"]

    # Select via clustering
    clusters = _cluster_by_execution(candidates)
    if not clusters:
        # All failed
        return {
            "final_sql": candidates[0]["sql"] if candidates else "",
            "candidates": [{k: v if k != "rows" else len(v) for k, v in c.items()}
                           for c in candidates],
            "tokens": total_tokens,
            "selection_method": "no_executable",
            "num_executable": 0,
            "num_groups": 0,
            "majority_size": 0,
        }

    # Largest cluster wins
    clusters.sort(key=len, reverse=True)
    max_len = len(clusters[0])
    top_clusters = [c for c in clusters if len(c) == max_len]

    sel_method = "majority_cluster"
    if len(top_clusters) == 1:
        final = top_clusters[0][0]  # representative
    else:
        # Tie — LLM pairwise vote between cluster representatives
        sel_method = "llm_pairwise_vote"
        reps = [c[0] for c in top_clusters]
        final = _model_vote(reps, question, schema_prompt, platform)

    return {
        "final_sql": final["sql"],
        "candidates": [{k: v if k != "rows" else len(v) for k, v in c.items()}
                       for c in candidates],
        "tokens": total_tokens,
        "selection_method": sel_method,
        "num_executable": sum(1 for c in candidates if not c["error"] and c["rows"]),
        "num_groups": len(clusters),
        "majority_size": max_len,
    }
