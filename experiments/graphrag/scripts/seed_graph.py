"""Seed the synthetic graph. Database deletion requires --clean-first."""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from graphrag.neo4j_client import get_neo4j_client
from graphrag.ingestion import DrugInteractionDataset, seed_dataset


def main():
    parser = argparse.ArgumentParser(description="Seed Neo4j with drug interaction data")
    parser.add_argument("--clean-first", action="store_true", help="Delete all nodes and relationships before seeding")
    args = parser.parse_args()
    with (PROJECT_ROOT / "data/raw/drug_interactions.json").open(encoding="utf-8") as f:
        dataset = DrugInteractionDataset.model_validate(json.load(f))
    with get_neo4j_client() as client:
        if not client.verify_connectivity():
            raise SystemExit("Failed to connect to Neo4j. Check your .env file.")
        if args.clean_first:
            client.write("MATCH (n) DETACH DELETE n")
        counts = seed_dataset(client, dataset)
        print("Seeded:", counts)
        info = client.get_schema_info()
        print(f"Graph: {info['total_nodes']} nodes, {info['total_relationships']} relationships")


if __name__ == "__main__":
    main()
