import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import PROJECT_ROOT, RESULTS_DIR, SQLITE_DB_DIR
from src.data_loader import (
    detect_platform,
    get_instances_with_gold_sql,
    load_gold_sql,
    load_external_knowledge,
    load_ddl,
    load_local_map,
    resolve_db_name,
)
from src.eval_sql_parser import extract_tables_columns, normalize_columns, validate_columns_against_schema
from src.schema_index import get_index
from src.agent import run_agent

RUNS_DIR = PROJECT_ROOT / "runs"


class RunLogger:
    """Writes evaluation output to both stdout and a log file in runs/."""

    def __init__(self, command: str, platform: str | None, dry_run: bool):
        RUNS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plat = platform or "all"
        suffix = "_dry" if dry_run else ""
        self.path = RUNS_DIR / f"{ts}_{plat}{suffix}.txt"
        self.file = open(self.path, "w", encoding="utf-8")
        self._write_header(command)

    def _write_header(self, command: str):
        self.file.write(f"Command: {command}\n")
        self.file.write(f"Started: {datetime.now().isoformat()}\n")
        self.file.write("=" * 70 + "\n\n")
        self.file.flush()

    def log(self, msg: str, end: str = "\n"):
        print(msg, end=end, flush=True)
        self.file.write(msg + end)
        self.file.flush()

    def close(self):
        self.file.write(f"\nFinished: {datetime.now().isoformat()}\n")
        self.file.close()


def compute_metrics(predicted: set[str], ground_truth: set[str]) -> dict:
    """Compute precision, recall, F1 between predicted and ground truth sets."""
    if not ground_truth:
        return {"precision": None, "recall": None, "f1": None}
    tp = len(predicted & ground_truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_strict_recall(predicted: set[str], ground_truth: set[str]) -> int:
    """Strict Recall Rate (AutoLink paper): 1 if predicted ⊇ ground_truth, else 0.

    Empty ground_truth counts as success (vacuously true).
    """
    if not ground_truth:
        return 1
    return 1 if ground_truth.issubset(predicted) else 0


def _get_schema_columns(index) -> dict[str, set[str]]:
    """Extract {table_lower: {col_lower, ...}} from a SchemaIndex for validation.

    For Snowflake / BigQuery: also expose short name (after last dot) so GT
    columns referenced without full path can be resolved.
    """
    schema_cols = {}
    for table_name, table_info in index.tables.items():
        t_lower = table_name.lower()
        cols = set()
        for col_name, col_type in table_info.columns:
            if "." not in col_name:  # skip nested
                cols.add(col_name.lower())
        # Full key
        if t_lower not in schema_cols:
            schema_cols[t_lower] = set()
        schema_cols[t_lower] |= cols
        # Short-name alias for DB.SCHEMA.TABLE -> TABLE
        if "." in t_lower:
            short = t_lower.rsplit(".", 1)[-1]
            schema_cols.setdefault(short, set())
            schema_cols[short] |= cols
    return schema_cols


def evaluate_instance_dry(instance: dict, local_map: dict) -> dict | None:
    """Dry run: parse gold SQL and extract ground truth without calling LLM."""
    instance_id = instance["instance_id"]
    gold_sql = load_gold_sql(instance_id)
    if not gold_sql:
        return None

    platform = detect_platform(instance_id)
    db_name = resolve_db_name(instance, local_map)
    if not db_name:
        return None

    gt_tables, gt_columns = extract_tables_columns(gold_sql, platform=platform)
    gt_columns = normalize_columns(gt_columns, gt_tables)

    index = get_index(db_name, platform=platform)
    known_tables = set(t.lower() for t in index.tables.keys())

    # Validate GT columns against real schema — drop aliases and computed names
    schema_cols = _get_schema_columns(index)
    gt_columns = validate_columns_against_schema(gt_columns, gt_tables, schema_cols)

    return {
        "instance_id": instance_id,
        "db_name": db_name,
        "platform": platform,
        "question": instance["question"][:100],
        "gt_tables": sorted(gt_tables),
        "gt_columns": sorted(gt_columns),
        "known_tables": sorted(known_tables),
        "gt_tables_in_schema": sorted(gt_tables & known_tables),
        "gt_tables_missing": sorted(gt_tables - known_tables),
    }


def _resolve_sqlite_path(db_name: str):
    """Find the .sqlite file for a database name."""
    path = SQLITE_DB_DIR / f"{db_name}.sqlite"
    if path.exists():
        return path
    for f in SQLITE_DB_DIR.iterdir():
        if f.suffix == ".sqlite" and f.stem.lower() == db_name.lower():
            return f
    return None


def evaluate_instance(instance: dict, local_map: dict, use_autolink: bool = False) -> dict | None:
    """Full evaluation: run agent and compare with ground truth."""
    instance_id = instance["instance_id"]
    gold_sql = load_gold_sql(instance_id)
    if not gold_sql:
        return None

    platform = detect_platform(instance_id)
    db_name = resolve_db_name(instance, local_map)
    if not db_name:
        return None

    # Ground truth
    gt_tables, gt_columns = extract_tables_columns(gold_sql, platform=platform)
    gt_columns = normalize_columns(gt_columns, gt_tables)

    # Validate GT columns against real schema — drop aliases and computed names
    index_for_schema = get_index(db_name, platform=platform)
    schema_cols = _get_schema_columns(index_for_schema)
    gt_columns = validate_columns_against_schema(gt_columns, gt_tables, schema_cols)

    # Load external knowledge
    ext_knowledge = None
    if instance.get("external_knowledge"):
        ext_knowledge = load_external_knowledge(instance["external_knowledge"])

    # Run agent
    start_time = time.time()

    if use_autolink and platform == "sqlite":
        from src.autolink_agent import run_autolink_agent
        from src.sqlite_executor import SQLiteExecutor
        from src.vector_store import get_vector_store

        sqlite_path = _resolve_sqlite_path(db_name)
        if not sqlite_path:
            return None

        ddl_data_full = load_ddl(db_name, platform="sqlite")
        executor = SQLiteExecutor(sqlite_path)
        vector_store = get_vector_store(db_name, ddl_data_full, sqlite_path)

        result = run_autolink_agent(
            question=instance["question"],
            db_name=db_name,
            ddl_data=ddl_data_full,
            executor=executor,
            vector_store=vector_store,
            external_knowledge=ext_knowledge,
        )
    elif use_autolink and platform == "snowflake":
        from src.autolink_agent import run_autolink_agent
        from src.snowflake_executor import SnowflakeExecutor
        from src.vector_store import get_vector_store
        from src.data_loader import load_sample_rows_from_json

        ddl_data_full = load_ddl(db_name, platform="snowflake")
        # Infer schema from full-path keys
        schema_counts: dict[str, int] = {}
        for tname in ddl_data_full:
            parts = tname.split(".")
            if len(parts) == 3:
                schema_counts[parts[1]] = schema_counts.get(parts[1], 0) + 1
        default_schema = max(schema_counts, key=schema_counts.get) if schema_counts else None

        executor = SnowflakeExecutor(default_db=db_name, default_schema=default_schema)
        external_samples = load_sample_rows_from_json(db_name, "snowflake")
        vector_store = get_vector_store(
            db_name, ddl_data_full,
            sqlite_path=None,
            external_samples=external_samples,
        )

        result = run_autolink_agent(
            question=instance["question"],
            db_name=db_name,
            ddl_data=ddl_data_full,
            executor=executor,
            vector_store=vector_store,
            external_knowledge=ext_knowledge,
        )
    elif use_autolink and platform == "bigquery":
        from src.autolink_agent import run_autolink_agent
        from src.bigquery_executor import BigQueryExecutor
        from src.vector_store import get_vector_store
        from src.data_loader import load_sample_rows_from_json

        ddl_data_full = load_ddl(db_name, platform="bigquery")
        external_samples = load_sample_rows_from_json(db_name, "bigquery")
        executor = BigQueryExecutor(sample_rows=external_samples)
        vector_store = get_vector_store(
            db_name, ddl_data_full,
            sqlite_path=None,
            external_samples=external_samples,
        )

        result = run_autolink_agent(
            question=instance["question"],
            db_name=db_name,
            ddl_data=ddl_data_full,
            executor=executor,
            vector_store=vector_store,
            external_knowledge=ext_knowledge,
        )
    else:
        index = get_index(db_name, platform=platform)
        result = run_agent(
            question=instance["question"],
            index=index,
            external_knowledge=ext_knowledge,
        )

    elapsed = time.time() - start_time

    def _short_table(t: str) -> str:
        return t.rsplit(".", 1)[-1].lower().strip('"')

    def _short_col(c: str) -> str:
        # "X.Y.Z.col" -> "z.col"
        c = c.lower().strip('"')
        if "." in c:
            parts = c.rsplit(".", 2) if c.count(".") >= 2 else c.rsplit(".", 1)
            if len(parts) == 3:
                return f"{parts[1]}.{parts[2]}"
            return c
        return c

    # Normalize predictions — for snowflake/bigquery collapse to short names
    if platform in ("snowflake", "bigquery"):
        pred_tables = {_short_table(t) for t in result.get("tables", [])}
        pred_columns = {_short_col(c) for c in result.get("columns", [])}
        gt_tables = {_short_table(t) for t in gt_tables}
        gt_columns = {_short_col(c) for c in gt_columns}
    else:
        pred_tables = set(t.lower() for t in result.get("tables", []))
        pred_columns = set(c.lower() for c in result.get("columns", []))

    # Compute metrics (P/R/F1)
    table_metrics = compute_metrics(pred_tables, gt_tables)
    column_metrics = compute_metrics(pred_columns, gt_columns)

    # Compute Strict Recall Rate (AutoLink paper metric)
    srr_table = compute_strict_recall(pred_tables, gt_tables)
    srr_column = compute_strict_recall(pred_columns, gt_columns)
    srr_all = 1 if (srr_table and srr_column) else 0

    return {
        "instance_id": instance_id,
        "db_name": db_name,
        "platform": platform,
        "question": instance["question"][:100],
        "gt_tables": sorted(gt_tables),
        "gt_columns": sorted(gt_columns),
        "pred_tables": sorted(pred_tables),
        "pred_columns": sorted(pred_columns),
        "table_metrics": table_metrics,
        "column_metrics": column_metrics,
        "srr_table": srr_table,
        "srr_column": srr_column,
        "srr_all": srr_all,
        "n_pred_tables": len(pred_tables),
        "n_pred_columns": len(pred_columns),
        "iterations": result.get("iterations", 0),
        "tool_calls_count": len(result.get("tool_calls", [])),
        "tool_calls": result.get("tool_calls", []),
        "elapsed_seconds": round(elapsed, 2),
        "error": result.get("error"),
    }


def run_evaluation(
    platform: str | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    use_autolink: bool = False,
    instance_ids: list[str] | None = None,
) -> dict:
    """Run evaluation on instances with gold SQL.

    Args:
        platform: Filter to 'sqlite', 'bigquery', 'snowflake', or None for all.
        dry_run: If True, only parse gold SQL without LLM calls.
        limit: Max number of instances to evaluate.
        use_autolink: If True, use AutoLink agent.
        instance_ids: If provided, runs only these specific instances.
    """
    command = " ".join(sys.argv)
    logger = RunLogger(command=command, platform=platform, dry_run=dry_run)

    instances = get_instances_with_gold_sql(platform=platform)
    if instance_ids:
        wanted = set(instance_ids)
        instances = [i for i in instances if i["instance_id"] in wanted]
        missing = wanted - {i["instance_id"] for i in instances}
        if missing:
            logger.log(f"⚠️  Requested instances not found: {sorted(missing)}")
    elif limit is not None:
        instances = instances[:limit]
    local_map = load_local_map()

    platform_label = platform or "all"
    logger.log(f"Found {len(instances)} instances with gold SQL (platform={platform_label})")
    logger.log("")

    # Set up incremental JSON output
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "_dry" if dry_run else ""
    output_path = RESULTS_DIR / f"evaluation_results_{platform_label}{suffix}.json"

    def flush_json(results, summary=None, partial=True):
        payload = {
            "platform": platform_label,
            "command": " ".join(sys.argv),
            "completed_instances": len(results),
            "total_instances": len(instances),
            "in_progress": partial,
            "instances": results,
            "summary": summary,
        }
        tmp = output_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp.replace(output_path)

    results = []
    for i, inst in enumerate(instances):
        instance_id = inst["instance_id"]
        inst_platform = detect_platform(instance_id)

        try:
            if dry_run:
                result = evaluate_instance_dry(inst, local_map)
            else:
                result = evaluate_instance(inst, local_map, use_autolink=use_autolink)
        except Exception as e:
            logger.log(f"[{i+1}/{len(instances)}] {instance_id} ({inst_platform})... ERROR: {e}")
            results.append({
                "instance_id": instance_id,
                "error": str(e),
            })
            # Flush even on per-instance crash
            try:
                flush_json(results, partial=(i < len(instances) - 1))
            except Exception:
                pass
            continue

        if result:
            results.append(result)
            if not dry_run:
                tm = result["table_metrics"]
                cm = result["column_metrics"]
                srr_t = "✓" if result.get("srr_table") else "✗"
                srr_c = "✓" if result.get("srr_column") else "✗"
                logger.log(
                    f"[{i+1}/{len(instances)}] {instance_id} ({inst_platform})  "
                    f"tables P={tm['precision']:.2f} R={tm['recall']:.2f} F1={tm['f1']:.2f} SRR={srr_t}  "
                    f"cols P={cm['precision']:.2f} R={cm['recall']:.2f} F1={cm['f1']:.2f} SRR={srr_c}  "
                    f"time={result['elapsed_seconds']}s"
                )
            else:
                missing = result.get("gt_tables_missing", [])
                line = f"[{i+1}/{len(instances)}] {instance_id} ({inst_platform})  tables={result['gt_tables']}"
                if missing:
                    line += f"  MISSING={missing}"
                logger.log(line)
        else:
            logger.log(f"[{i+1}/{len(instances)}] {instance_id} ({inst_platform})  SKIPPED")

        # Flush JSON after each instance — preserves state on crash
        try:
            flush_json(results, partial=(i < len(instances) - 1))
        except Exception as e:
            logger.log(f"  [WARN] Failed to flush JSON: {e}")

    # Aggregate metrics
    logger.log("")
    summary = aggregate_metrics(results, dry_run, logger)

    # Final flush with summary
    flush_json(results, summary=summary, partial=False)
    logger.log(f"\nJSON results saved to {output_path}")
    logger.log(f"Run log saved to {logger.path}")

    logger.close()
    return output


def aggregate_metrics(results: list[dict], dry_run: bool, logger: RunLogger) -> dict:
    """Compute aggregate metrics across all instances."""
    valid = [r for r in results if "gt_tables" in r]

    if dry_run:
        by_platform = {}
        for r in valid:
            p = r.get("platform", "unknown")
            if p not in by_platform:
                by_platform[p] = {"count": 0, "gt_tables": 0, "gt_columns": 0, "missing_tables": 0}
            by_platform[p]["count"] += 1
            by_platform[p]["gt_tables"] += len(r["gt_tables"])
            by_platform[p]["gt_columns"] += len(r["gt_columns"])
            by_platform[p]["missing_tables"] += len(r.get("gt_tables_missing", []))

        summary = {
            "total_instances": len(valid),
            "total_gt_tables": sum(len(r["gt_tables"]) for r in valid),
            "total_gt_columns": sum(len(r["gt_columns"]) for r in valid),
            "total_missing_tables": sum(len(r.get("gt_tables_missing", [])) for r in valid),
            "by_platform": by_platform,
        }

        logger.log("=" * 70)
        logger.log(f"DRY RUN SUMMARY ({len(valid)} instances)")
        logger.log("=" * 70)
        for p, stats in by_platform.items():
            logger.log(f"  {p}: {stats['count']} instances, "
                       f"{stats['gt_tables']} GT tables, "
                       f"{stats['gt_columns']} GT columns, "
                       f"{stats['missing_tables']} missing from schema")
        return summary

    table_metrics = [r["table_metrics"] for r in valid
                     if "table_metrics" in r and r["table_metrics"]["f1"] is not None]
    col_metrics = [r["column_metrics"] for r in valid
                   if "column_metrics" in r and r["column_metrics"]["f1"] is not None]

    def avg(values):
        return round(sum(values) / len(values), 4) if values else 0.0

    # SRR metrics (AutoLink paper)
    srr_table_vals = [r["srr_table"] for r in valid if "srr_table" in r]
    srr_col_vals = [r["srr_column"] for r in valid if "srr_column" in r]
    srr_all_vals = [r["srr_all"] for r in valid if "srr_all" in r]

    summary = {
        "total_instances": len(valid),
        "table_precision": avg([m["precision"] for m in table_metrics]),
        "table_recall": avg([m["recall"] for m in table_metrics]),
        "table_f1": avg([m["f1"] for m in table_metrics]),
        "column_precision": avg([m["precision"] for m in col_metrics]),
        "column_recall": avg([m["recall"] for m in col_metrics]),
        "column_f1": avg([m["f1"] for m in col_metrics]),
        # AutoLink-style metrics
        "srr_table": avg(srr_table_vals),
        "srr_column": avg(srr_col_vals),
        "srr_all": avg(srr_all_vals),
        "avg_pred_tables": avg([r.get("n_pred_tables", 0) for r in valid]),
        "avg_pred_columns": avg([r.get("n_pred_columns", 0) for r in valid]),
        "avg_gt_tables": avg([len(r["gt_tables"]) for r in valid]),
        "avg_gt_columns": avg([len(r["gt_columns"]) for r in valid]),
        "avg_iterations": avg([r["iterations"] for r in valid if "iterations" in r]),
        "avg_elapsed": avg([r["elapsed_seconds"] for r in valid if "elapsed_seconds" in r]),
    }

    logger.log("=" * 70)
    logger.log(f"EVALUATION SUMMARY ({len(valid)} instances)")
    logger.log("=" * 70)
    logger.log(f"Tables  - P: {summary['table_precision']:.4f}  R: {summary['table_recall']:.4f}  F1: {summary['table_f1']:.4f}")
    logger.log(f"Columns - P: {summary['column_precision']:.4f}  R: {summary['column_recall']:.4f}  F1: {summary['column_f1']:.4f}")
    logger.log("")
    logger.log("AutoLink-style Strict Recall Rate (SRR):")
    logger.log(f"  SRR-Table:  {summary['srr_table']:.4f}  ({sum(srr_table_vals)}/{len(srr_table_vals)} instances)")
    logger.log(f"  SRR-Column: {summary['srr_column']:.4f}  ({sum(srr_col_vals)}/{len(srr_col_vals)} instances)")
    logger.log(f"  SRR-All:    {summary['srr_all']:.4f}  ({sum(srr_all_vals)}/{len(srr_all_vals)} instances)")
    logger.log("")
    logger.log(f"Efficiency: avg #pred_tables={summary['avg_pred_tables']:.1f} (gt={summary['avg_gt_tables']:.1f}), "
               f"avg #pred_columns={summary['avg_pred_columns']:.1f} (gt={summary['avg_gt_columns']:.1f})")
    logger.log(f"Avg iterations: {summary['avg_iterations']:.1f}, Avg time: {summary['avg_elapsed']:.1f}s")

    return summary
