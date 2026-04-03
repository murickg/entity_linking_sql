import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.data_loader import (
    detect_platform,
    get_instances_with_gold_sql,
    load_gold_sql,
    load_external_knowledge,
    load_local_map,
    resolve_db_name,
)
from src.eval_sql_parser import extract_tables_columns, normalize_columns
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


def evaluate_instance(instance: dict, local_map: dict) -> dict | None:
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

    # Build index
    index = get_index(db_name, platform=platform)

    # Load external knowledge
    ext_knowledge = None
    if instance.get("external_knowledge"):
        ext_knowledge = load_external_knowledge(instance["external_knowledge"])

    # Run agent
    start_time = time.time()
    result = run_agent(
        question=instance["question"],
        index=index,
        external_knowledge=ext_knowledge,
    )
    elapsed = time.time() - start_time

    # Normalize predictions
    pred_tables = set(t.lower() for t in result.get("tables", []))
    pred_columns = set(c.lower() for c in result.get("columns", []))

    # Compute metrics
    table_metrics = compute_metrics(pred_tables, gt_tables)
    column_metrics = compute_metrics(pred_columns, gt_columns)

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
        "iterations": result.get("iterations", 0),
        "tool_calls_count": len(result.get("tool_calls", [])),
        "elapsed_seconds": round(elapsed, 2),
        "error": result.get("error"),
    }


def run_evaluation(platform: str | None = None, dry_run: bool = False, limit: int | None = None) -> dict:
    """Run evaluation on instances with gold SQL.

    Args:
        platform: Filter to 'sqlite', 'bigquery', 'snowflake', or None for all.
        dry_run: If True, only parse gold SQL without LLM calls.
        limit: Max number of instances to evaluate.
    """
    command = " ".join(sys.argv)
    logger = RunLogger(command=command, platform=platform, dry_run=dry_run)

    instances = get_instances_with_gold_sql(platform=platform)
    if limit is not None:
        instances = instances[:limit]
    local_map = load_local_map()

    platform_label = platform or "all"
    logger.log(f"Found {len(instances)} instances with gold SQL (platform={platform_label})")
    logger.log("")

    results = []
    for i, inst in enumerate(instances):
        instance_id = inst["instance_id"]
        inst_platform = detect_platform(instance_id)

        try:
            if dry_run:
                result = evaluate_instance_dry(inst, local_map)
            else:
                result = evaluate_instance(inst, local_map)
        except Exception as e:
            logger.log(f"[{i+1}/{len(instances)}] {instance_id} ({inst_platform})... ERROR: {e}")
            results.append({
                "instance_id": instance_id,
                "error": str(e),
            })
            continue

        if result:
            results.append(result)
            if not dry_run:
                tm = result["table_metrics"]
                cm = result["column_metrics"]
                logger.log(
                    f"[{i+1}/{len(instances)}] {instance_id} ({inst_platform})  "
                    f"tables P={tm['precision']:.2f} R={tm['recall']:.2f} F1={tm['f1']:.2f}  "
                    f"cols P={cm['precision']:.2f} R={cm['recall']:.2f} F1={cm['f1']:.2f}  "
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

    # Aggregate metrics
    logger.log("")
    summary = aggregate_metrics(results, dry_run, logger)

    # Save JSON results
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {"platform": platform_label, "instances": results, "summary": summary}
    suffix = "_dry" if dry_run else ""
    output_path = RESULTS_DIR / f"evaluation_results_{platform_label}{suffix}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
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

    summary = {
        "total_instances": len(valid),
        "table_precision": avg([m["precision"] for m in table_metrics]),
        "table_recall": avg([m["recall"] for m in table_metrics]),
        "table_f1": avg([m["f1"] for m in table_metrics]),
        "column_precision": avg([m["precision"] for m in col_metrics]),
        "column_recall": avg([m["recall"] for m in col_metrics]),
        "column_f1": avg([m["f1"] for m in col_metrics]),
        "avg_iterations": avg([r["iterations"] for r in valid if "iterations" in r]),
        "avg_elapsed": avg([r["elapsed_seconds"] for r in valid if "elapsed_seconds" in r]),
    }

    logger.log("=" * 70)
    logger.log(f"EVALUATION SUMMARY ({len(valid)} instances)")
    logger.log("=" * 70)
    logger.log(f"Tables  - P: {summary['table_precision']:.4f}  R: {summary['table_recall']:.4f}  F1: {summary['table_f1']:.4f}")
    logger.log(f"Columns - P: {summary['column_precision']:.4f}  R: {summary['column_recall']:.4f}  F1: {summary['column_f1']:.4f}")
    logger.log(f"Avg iterations: {summary['avg_iterations']:.1f}, Avg time: {summary['avg_elapsed']:.1f}s")

    return summary
