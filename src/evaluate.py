import json
import time

from src.config import RESULTS_DIR
from src.data_loader import (
    get_instances_with_gold_sql,
    load_gold_sql,
    load_external_knowledge,
    load_local_map,
)
from src.eval_sql_parser import extract_tables_columns, normalize_columns
from src.schema_index import get_index, resolve_db_name
from src.agent import run_agent


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

    db_name = resolve_db_name(instance_id, local_map)
    if not db_name:
        return None

    gt_tables, gt_columns = extract_tables_columns(gold_sql)
    gt_columns = normalize_columns(gt_columns, gt_tables)

    index = get_index(db_name)
    known_tables = set(t.lower() for t in index.tables.keys())

    return {
        "instance_id": instance_id,
        "db_name": db_name,
        "question": instance["question"][:100],
        "gt_tables": sorted(gt_tables),
        "gt_columns": sorted(gt_columns),
        "known_tables": sorted(known_tables),
        "gt_tables_in_schema": sorted(gt_tables & known_tables),
    }


def evaluate_instance(instance: dict, local_map: dict) -> dict | None:
    """Full evaluation: run agent and compare with ground truth."""
    instance_id = instance["instance_id"]
    gold_sql = load_gold_sql(instance_id)
    if not gold_sql:
        return None

    db_name = resolve_db_name(instance_id, local_map)
    if not db_name:
        return None

    # Ground truth
    gt_tables, gt_columns = extract_tables_columns(gold_sql)
    gt_columns = normalize_columns(gt_columns, gt_tables)

    # Build index
    index = get_index(db_name)

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


def run_evaluation(dry_run: bool = False) -> dict:
    """Run evaluation on all local instances with gold SQL."""
    instances = get_instances_with_gold_sql(only_local=True)
    local_map = load_local_map()

    print(f"Found {len(instances)} instances with gold SQL")

    results = []
    for i, inst in enumerate(instances):
        instance_id = inst["instance_id"]
        print(f"[{i+1}/{len(instances)}] {instance_id}...", end=" ", flush=True)

        if dry_run:
            result = evaluate_instance_dry(inst, local_map)
        else:
            result = evaluate_instance(inst, local_map)

        if result:
            results.append(result)
            if not dry_run:
                tm = result["table_metrics"]
                print(f"tables F1={tm['f1']}, cols F1={result['column_metrics']['f1']}, "
                      f"time={result['elapsed_seconds']}s")
            else:
                print(f"tables={result['gt_tables']}")
        else:
            print("SKIPPED (no gold SQL or db mapping)")

    # Aggregate metrics
    summary = aggregate_metrics(results, dry_run)

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {"instances": results, "summary": summary}
    suffix = "_dry" if dry_run else ""
    output_path = RESULTS_DIR / f"evaluation_results{suffix}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    return output


def aggregate_metrics(results: list[dict], dry_run: bool = False) -> dict:
    """Compute aggregate metrics across all instances."""
    if dry_run:
        return {
            "total_instances": len(results),
            "total_gt_tables": sum(len(r["gt_tables"]) for r in results),
            "total_gt_columns": sum(len(r["gt_columns"]) for r in results),
        }

    table_metrics = [r["table_metrics"] for r in results if r["table_metrics"]["f1"] is not None]
    col_metrics = [r["column_metrics"] for r in results if r["column_metrics"]["f1"] is not None]

    def avg(values):
        return round(sum(values) / len(values), 4) if values else 0.0

    summary = {
        "total_instances": len(results),
        "table_precision": avg([m["precision"] for m in table_metrics]),
        "table_recall": avg([m["recall"] for m in table_metrics]),
        "table_f1": avg([m["f1"] for m in table_metrics]),
        "column_precision": avg([m["precision"] for m in col_metrics]),
        "column_recall": avg([m["recall"] for m in col_metrics]),
        "column_f1": avg([m["f1"] for m in col_metrics]),
        "avg_iterations": avg([r["iterations"] for r in results]),
        "avg_elapsed": avg([r["elapsed_seconds"] for r in results]),
    }

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY ({len(results)} instances)")
    print(f"{'='*60}")
    print(f"Tables  - P: {summary['table_precision']:.4f}  R: {summary['table_recall']:.4f}  F1: {summary['table_f1']:.4f}")
    print(f"Columns - P: {summary['column_precision']:.4f}  R: {summary['column_recall']:.4f}  F1: {summary['column_f1']:.4f}")
    print(f"Avg iterations: {summary['avg_iterations']:.1f}, Avg time: {summary['avg_elapsed']:.1f}s")

    return summary
