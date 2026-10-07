"""
Initialize Neo4j graph schema — create constraints and indexes.

Usage:
    python scripts/init_graph.py           # Create schema only
    python scripts/init_graph.py --clean   # Clear graph first, then create schema
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
load_dotenv(PROJECT_ROOT / ".env")

from graphrag.neo4j_client import get_neo4j_client

from graphrag.schema import CONSTRAINT_QUERIES, INDEX_QUERIES


def main():
    parser = argparse.ArgumentParser(description="Initialize Neo4j graph schema")
    parser.add_argument("--clean", action="store_true", help="Delete all nodes and relationships first")
    args = parser.parse_args()

    with get_neo4j_client() as client:
        if not client.verify_connectivity():
            print("Failed to connect to Neo4j. Check your .env file.")
            sys.exit(1)

        if args.clean:
            print("Cleaning graph — deleting all nodes and relationships...")
            client.write("MATCH (n) DETACH DELETE n")
            print("Graph cleaned.")

        print("Creating constraints...")
        for q in CONSTRAINT_QUERIES:
            client.write(q)

        print("Creating indexes...")
        for q in INDEX_QUERIES:
            client.write(q)

        print("Schema setup complete!")

        info = client.get_schema_info()
        print(f"\nGraph state: {info['total_nodes']} nodes, {info['total_relationships']} relationships")


if __name__ == "__main__":
    main()
