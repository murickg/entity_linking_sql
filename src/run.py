import argparse
import json
import sys

from src.config import SQLITE_DB_DIR
from src.data_loader import (
    detect_platform,
    load_instances,
    load_local_map,
    load_external_knowledge,
    load_ddl,
    resolve_db_name,
)
from src.schema_index import get_index
from src.agent import run_agent
from src.evaluate import run_evaluation


def _resolve_sqlite_path(db_name: str):
    """Find the .sqlite file for a database name."""
    path = SQLITE_DB_DIR / f"{db_name}.sqlite"
    if path.exists():
        return path
    # Try case-insensitive
    for f in SQLITE_DB_DIR.iterdir():
        if f.suffix == ".sqlite" and f.stem.lower() == db_name.lower():
            return f
    return None


def run_single(instance_id: str, use_autolink: bool = False):
    """Run agent on a single instance and print results."""
    instances = load_instances(platform=None)
    local_map = load_local_map()

    instance = next((i for i in instances if i["instance_id"] == instance_id), None)
    if not instance:
        print(f"Instance '{instance_id}' not found.")
        sys.exit(1)

    platform = detect_platform(instance_id)
    db_name = resolve_db_name(instance, local_map)
    if not db_name:
        print(f"No database mapping found for '{instance_id}'.")
        sys.exit(1)

    print(f"Instance: {instance_id}")
    print(f"Platform: {platform}")
    print(f"Database: {db_name}")
    print(f"Agent: {'AutoLink' if use_autolink else 'BM25 baseline'}")
    print(f"Question: {instance['question']}")
    print()

    ext_knowledge = None
    if instance.get("external_knowledge"):
        ext_knowledge = load_external_knowledge(instance["external_knowledge"])
        if ext_knowledge:
            print(f"External knowledge loaded: {instance['external_knowledge']} "
                  f"({len(ext_knowledge)} chars)")

    if use_autolink and platform == "sqlite":
        from src.autolink_agent import run_autolink_agent
        from src.sqlite_executor import SQLiteExecutor
        from src.vector_store import get_vector_store

        sqlite_path = _resolve_sqlite_path(db_name)
        if not sqlite_path:
            print(f"SQLite file not found for '{db_name}'.")
            sys.exit(1)

        ddl_data = load_ddl(db_name, platform="sqlite")
        executor = SQLiteExecutor(sqlite_path)
        vector_store = get_vector_store(db_name, ddl_data, sqlite_path)

        print(f"Schema tables ({len(ddl_data)}): {sorted(ddl_data.keys())}")
        print(f"Vector store: {len(vector_store.documents)} column documents")
        print()
        print("Running AutoLink agent...")

        result = run_autolink_agent(
            question=instance["question"],
            db_name=db_name,
            ddl_data=ddl_data,
            executor=executor,
            vector_store=vector_store,
            external_knowledge=ext_knowledge,
        )
    else:
        if use_autolink and platform != "sqlite":
            print("Warning: --autolink only supported for SQLite, falling back to baseline.")

        index = get_index(db_name, platform=platform)
        print(f"Schema tables ({len(index.tables)}): {sorted(index.tables.keys())}")
        print()
        print("Running BM25 agent...")

        result = run_agent(
            question=instance["question"],
            index=index,
            external_knowledge=ext_knowledge,
        )

    print(f"\nAgent completed in {result.get('iterations', 0)} iterations, "
          f"{len(result.get('tool_calls', []))} tool calls")
    print(f"\nPredicted tables: {result['tables']}")
    print(f"Predicted columns: {result['columns']}")

    if result.get("error"):
        print(f"\nError: {result['error']}")

    if result.get("tool_calls"):
        print(f"\nTool call history:")
        for tc in result["tool_calls"]:
            print(f"  - {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:100]})")


def main():
    parser = argparse.ArgumentParser(description="Entity Linking for SQL databases")
    parser.add_argument("--instance", type=str, help="Run agent on a single instance")
    parser.add_argument("--evaluate", action="store_true", help="Run full evaluation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run: only parse gold SQL, no LLM calls")
    parser.add_argument("--platform", type=str, choices=["sqlite", "bigquery", "snowflake"],
                        default=None, help="Filter to a specific platform (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of instances to evaluate")
    parser.add_argument("--autolink", action="store_true",
                        help="Use AutoLink agent (SQLite only)")

    args = parser.parse_args()

    if args.instance:
        run_single(args.instance, use_autolink=args.autolink)
    elif args.evaluate:
        run_evaluation(
            platform=args.platform,
            dry_run=args.dry_run,
            limit=args.limit,
            use_autolink=args.autolink,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
