import argparse
import json
import sys

from src.data_loader import load_instances, load_local_map, load_external_knowledge
from src.schema_index import get_index, resolve_db_name
from src.agent import run_agent
from src.evaluate import run_evaluation


def run_single(instance_id: str):
    """Run agent on a single instance and print results."""
    instances = load_instances(only_local=True)
    local_map = load_local_map()

    instance = next((i for i in instances if i["instance_id"] == instance_id), None)
    if not instance:
        print(f"Instance '{instance_id}' not found among local instances.")
        sys.exit(1)

    db_name = resolve_db_name(instance_id, local_map)
    if not db_name:
        print(f"No database mapping found for '{instance_id}'.")
        sys.exit(1)

    print(f"Instance: {instance_id}")
    print(f"Database: {db_name}")
    print(f"Question: {instance['question']}")
    print()

    index = get_index(db_name)
    print(f"Schema tables: {sorted(index.tables.keys())}")
    print()

    ext_knowledge = None
    if instance.get("external_knowledge"):
        ext_knowledge = load_external_knowledge(instance["external_knowledge"])
        if ext_knowledge:
            print(f"External knowledge loaded: {instance['external_knowledge']}")

    print("Running agent...")
    result = run_agent(
        question=instance["question"],
        index=index,
        external_knowledge=ext_knowledge,
    )

    print(f"\nAgent completed in {result['iterations']} iterations, "
          f"{len(result.get('tool_calls', []))} tool calls")
    print(f"\nPredicted tables: {result['tables']}")
    print(f"Predicted columns: {result['columns']}")

    if result.get("error"):
        print(f"\nError: {result['error']}")

    # Show tool call history
    if result.get("tool_calls"):
        print(f"\nTool call history:")
        for tc in result["tool_calls"]:
            print(f"  - {tc['name']}({json.dumps(tc['args'])})")


def main():
    parser = argparse.ArgumentParser(description="Entity Linking for SQL databases")
    parser.add_argument("--instance", type=str, help="Run agent on a single instance")
    parser.add_argument("--evaluate", action="store_true", help="Run full evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Dry run: only parse gold SQL, no LLM calls")

    args = parser.parse_args()

    if args.instance:
        run_single(args.instance)
    elif args.evaluate:
        run_evaluation(dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
