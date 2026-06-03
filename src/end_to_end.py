"""End-to-end EX evaluation comparing 3 schema linking approaches.

Approaches:
1. full       — feed entire DB schema, no linking
2. autolink   — use official AutoLink's pre-computed linking results
3. ours       — run our agent, use its predicted schema

For each: build schema prompt → generate SQL → execute → compare with gold.
Tracks token usage at each stage.
"""

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
    load_sample_rows_from_json,
    get_bq_table_fullnames,
    resolve_db_name,
)
from src.sqlite_executor import SQLiteExecutor
from src.vector_store import get_vector_store
from src.autolink_agent import run_autolink_agent
from src.sql_generator import generate_sql, run_sql_pipeline
from src.schema_formatters import (
    format_full_schema,
    format_autolink_schema,
    format_our_schema,
)
from src.ex_evaluator import evaluate_ex, evaluate_ex_snowflake, evaluate_ex_bigquery

RUNS_DIR = PROJECT_ROOT / "runs"


def _resolve_sqlite_path(db_name: str) -> Path | None:
    path = SQLITE_DB_DIR / f"{db_name}.sqlite"
    if path.exists():
        return path
    for f in SQLITE_DB_DIR.iterdir():
        if f.suffix == ".sqlite" and f.stem.lower() == db_name.lower():
            return f
    return None


def evaluate_one_instance(
    instance: dict,
    local_map: dict,
    approaches: list[str],
    use_pipeline: bool = False,
    num_candidates: int = 5,
    platform: str = "sqlite",
    max_missing_descriptions: int | None = None,
) -> dict:
    """Evaluate one instance across all chosen approaches."""
    instance_id = instance["instance_id"]
    gold_sql = load_gold_sql(instance_id)
    if not gold_sql:
        return {"instance_id": instance_id, "error": "no gold SQL"}

    db_name = resolve_db_name(instance, local_map)
    if not db_name:
        return {"instance_id": instance_id, "error": "could not resolve db_name"}

    # Platform-specific setup
    sqlite_path = None
    executor = None
    sf_executor = None
    bq_executor = None
    external_samples = None
    bq_fullnames = None

    if platform == "sqlite":
        sqlite_path = _resolve_sqlite_path(db_name)
        if not sqlite_path:
            return {"instance_id": instance_id, "error": f"no sqlite for {db_name}"}
        executor = SQLiteExecutor(sqlite_path)
    elif platform == "snowflake":
        from src.snowflake_executor import SnowflakeExecutor
        ddl_data_tmp = load_ddl(db_name, platform="snowflake")
        schema_counts: dict[str, int] = {}
        for tname in ddl_data_tmp:
            parts = tname.split(".")
            if len(parts) == 3:
                schema_counts[parts[1]] = schema_counts.get(parts[1], 0) + 1
        default_schema = max(schema_counts, key=schema_counts.get) if schema_counts else None
        sf_executor = SnowflakeExecutor(default_db=db_name, default_schema=default_schema)
        executor = sf_executor
        external_samples = load_sample_rows_from_json(db_name, "snowflake")
    elif platform == "bigquery":
        from src.bigquery_executor import BigQueryExecutor
        external_samples = load_sample_rows_from_json(db_name, "bigquery")
        bq_executor = BigQueryExecutor(sample_rows=external_samples)
        executor = bq_executor
        bq_fullnames = get_bq_table_fullnames(db_name)
    else:
        return {"instance_id": instance_id, "error": f"unsupported platform {platform}"}

    ddl_data = load_ddl(db_name, platform=platform)
    ext_knowledge = None
    if instance.get("external_knowledge"):
        ext_knowledge = load_external_knowledge(instance["external_knowledge"])

    record = {
        "instance_id": instance_id,
        "db_name": db_name,
        "question": instance["question"][:150],
        "approaches": {},
    }

    for approach in approaches:
        result = {
            "schema_tokens": 0,        # tokens spent during schema linking
            "sql_tokens": 0,           # tokens spent during SQL generation
            "schema_len_chars": 0,
            "linking_seconds": 0.0,
            "sql_gen_seconds": 0.0,
            "ex_seconds": 0.0,
            "sql": "",
            "ex_match": False,
            "match_type": "",
            "pred_error": None,
        }

        # --- Build schema prompt ---
        link_start = time.time()
        schema_prompt = None

        if approach == "full":
            if platform == "sqlite":
                schema_prompt = format_full_schema(db_name, ddl_data, executor=executor)
            else:
                schema_prompt = format_full_schema(
                    db_name, ddl_data,
                    external_samples=external_samples,
                    bq_fullnames=bq_fullnames,
                )

        elif approach == "autolink":
            schema_prompt = format_autolink_schema(instance_id)
            if schema_prompt is None:
                result["error"] = "no official autolink result"
                record["approaches"][approach] = result
                continue

        elif approach == "ours":
            from src.column_describer import DescriptionLimitExceeded
            try:
                vs = get_vector_store(
                    db_name, ddl_data,
                    sqlite_path=sqlite_path,
                    external_samples=external_samples,
                    max_missing_descriptions=max_missing_descriptions,
                )
            except DescriptionLimitExceeded as e:
                result["error"] = f"skipped: {e}"
                record["approaches"][approach] = result
                continue
            agent_result = run_autolink_agent(
                question=instance["question"],
                db_name=db_name,
                ddl_data=ddl_data,
                executor=executor,
                vector_store=vs,
                external_knowledge=ext_knowledge,
            )
            result["schema_tokens"] = agent_result.get("tokens", {}).get("total", 0)
            if platform == "sqlite":
                schema_prompt = format_our_schema(
                    agent_result.get("tables", []),
                    agent_result.get("columns", []),
                    ddl_data, executor=executor,
                )
            else:
                schema_prompt = format_our_schema(
                    agent_result.get("tables", []),
                    agent_result.get("columns", []),
                    ddl_data,
                    external_samples=external_samples,
                    bq_fullnames=bq_fullnames,
                )
            result["agent_iterations"] = agent_result.get("iterations", 0)
            result["agent_pred_tables"] = agent_result.get("tables", [])
            result["agent_pred_columns"] = agent_result.get("columns", [])

        result["linking_seconds"] = round(time.time() - link_start, 2)
        result["schema_len_chars"] = len(schema_prompt or "")

        # --- Generate SQL ---
        if not schema_prompt:
            result["error"] = "empty schema prompt"
            record["approaches"][approach] = result
            continue

        sql_start = time.time()

        if use_pipeline:
            # Multi-candidate + revise + select pipeline
            pipeline_result = run_sql_pipeline(
                question=instance["question"],
                schema_prompt=schema_prompt,
                sqlite_path=sqlite_path,
                sf_executor=sf_executor,
                bq_executor=bq_executor,
                external_knowledge=ext_knowledge,
                num_candidates=num_candidates,
                platform=platform,
            )
            result["sql_gen_seconds"] = round(time.time() - sql_start, 2)
            result["sql"] = pipeline_result["final_sql"]
            result["sql_tokens"] = pipeline_result["tokens"]["total"]
            result["pipeline_candidates"] = pipeline_result["candidates"]
            result["selection_method"] = pipeline_result["selection_method"]
            result["num_executable"] = pipeline_result.get("num_executable", 0)
            result["num_groups"] = pipeline_result.get("num_groups", 0)
            result["majority_size"] = pipeline_result.get("majority_size", 0)
            if not pipeline_result["final_sql"]:
                result["error"] = "pipeline produced no sql"
                record["approaches"][approach] = result
                continue
        else:
            gen = generate_sql(
                question=instance["question"],
                schema_prompt=schema_prompt,
                external_knowledge=ext_knowledge,
                platform=platform,
            )
            result["sql_gen_seconds"] = round(time.time() - sql_start, 2)
            result["sql"] = gen["sql"]
            result["sql_tokens"] = gen["tokens"]["total"]

            if gen["error"]:
                result["error"] = gen["error"]
                record["approaches"][approach] = result
                continue

        # --- Execute & evaluate EX ---
        ex_start = time.time()
        if platform == "sqlite":
            ex = evaluate_ex(result["sql"], gold_sql, sqlite_path)
        elif platform == "snowflake":
            ex = evaluate_ex_snowflake(result["sql"], gold_sql, sf_executor)
        elif platform == "bigquery":
            ex = evaluate_ex_bigquery(result["sql"], gold_sql, bq_executor)
        else:
            ex = {"ex_match": False, "match_type": "unsupported", "pred_rows": 0, "gold_rows": 0}
        result["ex_seconds"] = round(time.time() - ex_start, 2)
        result["ex_match"] = ex["ex_match"]
        result["match_type"] = ex["match_type"]
        result["pred_error"] = ex.get("pred_error")
        result["pred_rows"] = ex.get("pred_rows")
        result["gold_rows"] = ex.get("gold_rows")

        record["approaches"][approach] = result

    return record


def run_end_to_end(
    approaches: list[str] | None = None,
    limit: int | None = None,
    use_pipeline: bool = False,
    num_candidates: int = 5,
    platform: str = "sqlite",
    instance_ids: list[str] | None = None,
    max_missing_descriptions: int | None = None,
):
    """Main entry: orchestrate all approaches × all instances.

    Args:
        instance_ids: If provided, runs only these specific instances
                      (by their instance_id like 'local019' or 'bq042').
                      Takes precedence over `limit`.
    """
    if approaches is None:
        approaches = ["full", "autolink", "ours"]

    instances = get_instances_with_gold_sql(platform=platform)
    if instance_ids:
        wanted = set(instance_ids)
        instances = [i for i in instances if i["instance_id"] in wanted]
        missing = wanted - {i["instance_id"] for i in instances}
        if missing:
            print(f"⚠️  Requested instances not found: {sorted(missing)}")
    elif limit is not None:
        instances = instances[:limit]
    local_map = load_local_map()

    print(f"Running {len(approaches)} approach(es) × {len(instances)} instances")
    print(f"Approaches: {approaches}")
    print(f"Pipeline: {'multi-candidate(' + str(num_candidates) + ') + revise + select' if use_pipeline else 'single-shot'}")
    print()

    # Set up log
    RUNS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RUNS_DIR / f"{ts}_{platform}_ex_comparison.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Command: {' '.join(sys.argv)}")
    log(f"Started: {datetime.now().isoformat()}")
    log(f"Approaches: {approaches}")
    log(f"Instances: {len(instances)}")
    log("=" * 70)
    log("")

    all_records = []

    # Prepare incremental JSON output (flushed after each instance)
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"ex_comparison_{platform}_{ts}.json"

    def flush_json(partial: bool = True):
        """Write current state to JSON. Safe to call after every instance."""
        payload = {
            "approaches": approaches,
            "platform": platform,
            "started": ts,
            "command": " ".join(sys.argv),
            "completed_instances": len(all_records),
            "total_instances": len(instances),
            "in_progress": partial,
            "records": all_records,
        }
        # Atomic write: tmp + rename, in case of crash mid-write
        tmp_path = output_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp_path.replace(output_path)

    for i, inst in enumerate(instances):
        instance_id = inst["instance_id"]
        log(f"[{i+1}/{len(instances)}] {instance_id}")

        try:
            record = evaluate_one_instance(
                inst, local_map, approaches,
                use_pipeline=use_pipeline,
                num_candidates=num_candidates,
                platform=platform,
                max_missing_descriptions=max_missing_descriptions,
            )
        except Exception as e:
            record = {
                "instance_id": instance_id,
                "error": f"crash: {type(e).__name__}: {str(e)[:300]}",
            }
            log(f"  CRASH: {record['error']}")
        all_records.append(record)

        # Flush JSON after each instance — guarantees persistence even on crash
        try:
            flush_json(partial=(i < len(instances) - 1))
        except Exception as e:
            log(f"  [WARN] Failed to flush JSON: {e}")

        if "error" in record:
            log(f"  SKIPPED: {record['error']}")
            continue

        for approach in approaches:
            r = record["approaches"].get(approach, {})
            if "error" in r:
                log(f"  {approach:<10}  ERROR: {r['error']}")
                continue
            tok_s = r.get("schema_tokens", 0)
            tok_q = r.get("sql_tokens", 0)
            match = "✓" if r.get("ex_match") else "✗"
            extra = ""
            if use_pipeline:
                extra = f"  cands={r.get('num_executable', 0)}/{num_candidates} maj={r.get('majority_size', 0)} sel={r.get('selection_method', '?')[:10]}"
            log(
                f"  {approach:<10}  EX={match} ({r.get('match_type', '?'):<15})  "
                f"sch_tok={tok_s:>6}  sql_tok={tok_q:>6}  "
                f"sch_chars={r.get('schema_len_chars', 0):>6}  "
                f"link_t={r.get('linking_seconds', 0):.1f}s  "
                f"gen_t={r.get('sql_gen_seconds', 0):.1f}s"
                f"{extra}"
            )

    # --- Aggregate ---
    log("")
    log("=" * 70)
    log("AGGREGATE RESULTS")
    log("=" * 70)

    for approach in approaches:
        valid = [r for r in all_records if approach in r.get("approaches", {})
                 and "error" not in r["approaches"][approach]]
        if not valid:
            log(f"\n{approach}: no valid runs")
            continue
        n = len(valid)
        ex_count = sum(1 for r in valid if r["approaches"][approach]["ex_match"])
        avg_sch_tok = sum(r["approaches"][approach]["schema_tokens"] for r in valid) / n
        avg_sql_tok = sum(r["approaches"][approach]["sql_tokens"] for r in valid) / n
        avg_total_tok = avg_sch_tok + avg_sql_tok
        avg_sch_chars = sum(r["approaches"][approach]["schema_len_chars"] for r in valid) / n
        total_sch_tok = sum(r["approaches"][approach]["schema_tokens"] for r in valid)
        total_sql_tok = sum(r["approaches"][approach]["sql_tokens"] for r in valid)

        log(f"\n--- {approach} ---")
        log(f"  EX accuracy:        {ex_count}/{n} = {ex_count/n:.3f}")
        log(f"  Avg schema tokens:  {avg_sch_tok:.0f}  (total: {total_sch_tok})")
        log(f"  Avg SQL tokens:     {avg_sql_tok:.0f}  (total: {total_sql_tok})")
        log(f"  Avg total tokens:   {avg_total_tok:.0f}")
        log(f"  Avg schema chars:   {avg_sch_chars:.0f}")

    # Final flush — mark run as completed
    flush_json(partial=False)
    log(f"\nJSON saved to {output_path}")
    log(f"Log saved to {log_path}")
    log(f"Finished: {datetime.now().isoformat()}")

    log_file.close()
    return all_records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--approaches", nargs="+",
                        default=["full", "autolink", "ours"],
                        help="Subset of: full, autolink, ours")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pipeline", action="store_true",
                        help="Use multi-candidate + revise + select pipeline")
    parser.add_argument("--candidates", type=int, default=5,
                        help="Number of SQL candidates (default: 5, used with --pipeline)")
    parser.add_argument("--platform", type=str, default="sqlite",
                        choices=["sqlite", "snowflake", "bigquery"])
    parser.add_argument("--instances", nargs="+", default=None,
                        help="Run only specific instances by ID, e.g. --instances bq042 bq119 local019")
    parser.add_argument("--max-missing-descriptions", type=int, default=None,
                        help="Skip instance if its DB needs more than N fresh LLM column descriptions "
                             "(saves time/tokens on huge schemas). Only applies to 'ours' approach.")
    args = parser.parse_args()
    run_end_to_end(
        approaches=args.approaches,
        limit=args.limit,
        use_pipeline=args.pipeline,
        num_candidates=args.candidates,
        platform=args.platform,
        instance_ids=args.instances,
        max_missing_descriptions=args.max_missing_descriptions,
    )
