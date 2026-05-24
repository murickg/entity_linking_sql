"""Execution Accuracy (EX) evaluator.

Executes predicted SQL and gold SQL, compares result sets.
"""

import sqlite3
from pathlib import Path


def _execute_sql(db_path: Path, sql: str, timeout: float = 30.0) -> tuple[list[tuple], str | None]:
    """Execute SQL on SQLite DB. Returns (rows, error_message or None)."""
    try:
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            timeout=timeout,
        )
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return rows, None
    except Exception as e:
        return [], str(e)


def _normalize_value(v):
    """Normalize a single value for comparison."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        # Treat int and float as same: 1 == 1.0
        return round(float(v), 4)
    return str(v).strip()


def _normalize_rows(rows: list[tuple]) -> set[tuple]:
    """Normalize rows for order-independent (row order) comparison."""
    return {tuple(_normalize_value(v) for v in row) for row in rows}


def _normalize_rows_column_orderless(rows: list[tuple]) -> set[frozenset]:
    """Normalize rows ignoring BOTH row order AND column order.

    Each row becomes a frozenset of (col_idx_sorted_value) — well actually
    a sorted tuple representation of values. This is the typical Spider-style
    EX comparison.
    """
    result = set()
    for row in rows:
        normalized = tuple(sorted(
            (str(_normalize_value(v)) for v in row),
            key=lambda x: (x is None, x)
        ))
        result.add(normalized)
    return result


def _execute_snowflake(sql: str, sf_executor) -> tuple[list[tuple], str | None]:
    """Execute SQL via SnowflakeExecutor, return rows as list of tuples."""
    try:
        conn = sf_executor._get_connection()
        cur = conn.cursor()
        # Ensure DATABASE/SCHEMA context matches the executor's defaults
        sf_executor._use_db(cur, sf_executor.default_db, sf_executor.default_schema)
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return list(rows), None
    except Exception as e:
        return [], str(e)


def evaluate_ex(
    pred_sql: str,
    gold_sql: str,
    db_path: Path,
    timeout: float = 30.0,
) -> dict:
    """Execute both predicted and gold SQL, compare result sets.

    Returns: {
        "ex_match": bool,
        "pred_rows": int,
        "gold_rows": int,
        "pred_error": str | None,
        "gold_error": str | None,
        "match_type": "exact" | "set_match" | "row_count_diff" | "mismatch" | "pred_error",
    }
    """
    pred_rows, pred_err = _execute_sql(db_path, pred_sql, timeout)
    gold_rows, gold_err = _execute_sql(db_path, gold_sql, timeout)

    result = {
        "pred_rows": len(pred_rows),
        "gold_rows": len(gold_rows),
        "pred_error": pred_err,
        "gold_error": gold_err,
        "ex_match": False,
        "match_type": "mismatch",
    }

    if pred_err:
        result["match_type"] = "pred_error"
        return result
    if gold_err:
        result["match_type"] = "gold_error"
        return result

    # Order-sensitive (exact) check
    if pred_rows == gold_rows:
        result["ex_match"] = True
        result["match_type"] = "exact"
        return result

    # Order-insensitive (set) check — row order ignored
    if _normalize_rows(pred_rows) == _normalize_rows(gold_rows):
        result["ex_match"] = True
        result["match_type"] = "set_match"
        return result

    # Column-order-insensitive — Spider-style EX comparison
    if _normalize_rows_column_orderless(pred_rows) == _normalize_rows_column_orderless(gold_rows):
        result["ex_match"] = True
        result["match_type"] = "col_orderless_match"
        return result

    # Subset match: pred rows are subset of gold (same #rows, fewer columns)
    if len(pred_rows) == len(gold_rows) and pred_rows and gold_rows:
        pred_norm = _normalize_rows_column_orderless(pred_rows)
        gold_norm = _normalize_rows_column_orderless(gold_rows)
        # Check if every pred row's values are a subset of some gold row
        gold_value_sets = [frozenset(_normalize_value(v) for v in row) for row in gold_rows]
        pred_value_sets = [frozenset(_normalize_value(v) for v in row) for row in pred_rows]
        if all(any(p.issubset(g) for g in gold_value_sets) for p in pred_value_sets):
            result["ex_match"] = True
            result["match_type"] = "subset_match"
            return result

    # AutoLink-style column-wise match (most lenient — last chance)
    if _pandas_compare_match(pred_rows, gold_rows):
        result["ex_match"] = True
        result["match_type"] = "pandas_match"
        return result

    if len(pred_rows) != len(gold_rows):
        result["match_type"] = "row_count_diff"
    return result


def _pandas_compare_match(pred_rows, gold_rows) -> bool:
    """AutoLink-style column-wise comparison with float tolerance.

    Returns True iff for EVERY gold column, there is a matching pred column
    (allowing row-order shuffle + float tolerance 1e-2). Extra pred columns OK.
    """
    if not pred_rows or not gold_rows:
        return False
    import math
    import pandas as pd

    pred_df = pd.DataFrame(pred_rows)
    gold_df = pd.DataFrame(gold_rows)
    tol = 1e-2

    def vectors_match(v1, v2):
        v1 = sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
        v2 = sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
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

    t_gold = gold_df.transpose().values.tolist()
    t_pred = pred_df.transpose().values.tolist()
    for gold_col in t_gold:
        if not any(vectors_match(gold_col, pred_col) for pred_col in t_pred):
            return False
    return True


def _do_match(pred_rows, gold_rows, result):
    """Shared matching logic for evaluate_ex_snowflake."""
    if pred_rows == gold_rows:
        result["ex_match"] = True
        result["match_type"] = "exact"
        return result
    if _normalize_rows(pred_rows) == _normalize_rows(gold_rows):
        result["ex_match"] = True
        result["match_type"] = "set_match"
        return result
    if _normalize_rows_column_orderless(pred_rows) == _normalize_rows_column_orderless(gold_rows):
        result["ex_match"] = True
        result["match_type"] = "col_orderless_match"
        return result
    if len(pred_rows) == len(gold_rows) and pred_rows and gold_rows:
        gold_value_sets = [frozenset(_normalize_value(v) for v in row) for row in gold_rows]
        pred_value_sets = [frozenset(_normalize_value(v) for v in row) for row in pred_rows]
        if all(any(p.issubset(g) for g in gold_value_sets) for p in pred_value_sets):
            result["ex_match"] = True
            result["match_type"] = "subset_match"
            return result
    # AutoLink-style column-wise match (most lenient — last chance)
    if _pandas_compare_match(pred_rows, gold_rows):
        result["ex_match"] = True
        result["match_type"] = "pandas_match"
        return result
    if len(pred_rows) != len(gold_rows):
        result["match_type"] = "row_count_diff"
    return result


def _execute_bigquery(sql: str, bq_executor) -> tuple[list[tuple], str | None]:
    """Execute SQL via BigQueryExecutor's underlying client."""
    try:
        client = bq_executor._get_client()
        query_job = client.query(sql)
        rows = list(query_job.result(timeout=bq_executor.timeout))
        # Convert BigQuery Row to tuple
        out = [tuple(r.values()) for r in rows]
        return out, None
    except Exception as e:
        return [], str(e)


def evaluate_ex_bigquery(pred_sql: str, gold_sql: str, bq_executor) -> dict:
    """EX evaluation for BigQuery — executes both pred and gold via google-cloud-bigquery."""
    pred_rows, pred_err = _execute_bigquery(pred_sql, bq_executor)
    gold_rows, gold_err = _execute_bigquery(gold_sql, bq_executor)
    result = {
        "pred_rows": len(pred_rows),
        "gold_rows": len(gold_rows),
        "pred_error": pred_err,
        "gold_error": gold_err,
        "ex_match": False,
        "match_type": "mismatch",
    }
    if pred_err:
        result["match_type"] = "pred_error"
        return result
    if gold_err:
        result["match_type"] = "gold_error"
        return result
    return _do_match(pred_rows, gold_rows, result)


def evaluate_ex_snowflake(pred_sql: str, gold_sql: str, sf_executor) -> dict:
    """EX evaluation for Snowflake — uses SnowflakeExecutor connection."""
    pred_rows, pred_err = _execute_snowflake(pred_sql, sf_executor)
    gold_rows, gold_err = _execute_snowflake(gold_sql, sf_executor)
    result = {
        "pred_rows": len(pred_rows),
        "gold_rows": len(gold_rows),
        "pred_error": pred_err,
        "gold_error": gold_err,
        "ex_match": False,
        "match_type": "mismatch",
    }
    if pred_err:
        result["match_type"] = "pred_error"
        return result
    if gold_err:
        result["match_type"] = "gold_error"
        return result
    return _do_match(pred_rows, gold_rows, result)
