.PHONY: help install init-graph seed-graph init-and-seed reset-graph status notebooks clean

help:
	@echo "Neo4j GraphRAG experiment"
	@echo "  make install        Install dependencies"
	@echo "  make init-graph     Create Neo4j constraints/indexes"
	@echo "  make seed-graph     Load the drug interaction graph"
	@echo "  make init-and-seed  Initialize and populate the graph"
	@echo "  make reset-graph    Clear and rebuild the graph"
	@echo "  make status         Print graph counts"
	@echo "  make notebooks      Launch Jupyter Lab"

install:
	pip install -r ../../requirements/graphrag.txt

init-graph:
	PYTHONPATH=src python scripts/init_graph.py

seed-graph:
	PYTHONPATH=src python scripts/seed_graph.py

init-and-seed: init-graph seed-graph

reset-graph:
	PYTHONPATH=src python scripts/init_graph.py --clean
	PYTHONPATH=src python scripts/seed_graph.py

status:
	PYTHONPATH=src python -c "from dotenv import load_dotenv; load_dotenv(); from graphrag.neo4j_client import get_neo4j_client; c=get_neo4j_client(); c.verify_connectivity(); i=c.get_schema_info(); print('Nodes: {}, Relationships: {}'.format(i['total_nodes'], i['total_relationships'])); c.close()"

notebooks:
	PYTHONPATH=src jupyter lab notebooks/

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} + 2>/dev/null || true
