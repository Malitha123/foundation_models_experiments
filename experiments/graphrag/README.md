# Neo4j GraphRAG

Knowledge-graph retrieval experiment for multi-hop reasoning over drug, condition, side-effect, contraindication, and drug-class relationships.

## Pipeline

1. Build Neo4j constraints and indexes.
2. Ingest structured drug and interaction records with idempotent `MERGE` queries.
3. Optionally extract entities and relationships from unstructured text with structured LLM output.
4. Resolve entities with exact, fuzzy, and semantic matching.
5. Retrieve single-hop and multi-hop graph context with Cypher.
6. Grade context relevance and rewrite weak queries before generation.

## Setup

```bash
pip install -r ../../requirements/graphrag.txt
cp ../../.env.example .env
make init-and-seed
make notebooks
```

Start with `notebooks/knowledge_graph_etl.ipynb`, then run `notebooks/corrective_graphrag.ipynb`.

The bundled JSON is a small experiment dataset for graph construction and retrieval testing. It is not a clinical drug-interaction database.

## Data note

The bundled graph data is a prototype dataset assembled for retrieval engineering. Some interaction records refer to drugs that do not have full metadata entries; the seeding pipeline marks these explicitly as `metadata_status=reference_only` rather than silently treating them as complete records. It contains medical-style relationship text and recommendation fields, but it is not an authoritative pharmacology source and is not suitable for clinical decision-making. Any source labels inside the synthetic passages should be treated as dataset metadata rather than verified citations.

## Current implementation

The notebooks share schema, ingestion, retrieval, and agent modules with the CLI. Schema creation uses uniqueness constraints and their backing indexes. Notebook execution preserves existing data; database resets remain explicit CLI options.

The bundled 244 interaction records describe 227 unordered drug pairs. Seeding stores one structured edge per pair and retains all source records in `evidence_records`. Reversed structured duplicates are reconciled on rerun. Fully described drugs receive `metadata_status=complete`, including nodes previously marked reference-only.

The agent replaces context after every retrieval attempt, grades against the original question, and returns an explicit insufficient-evidence response after the configured retries. Retrieval returns no documents for missing evidence. Exact aliases and scored fuzzy matching precede semantic matching; equal names across labels remain distinct in the embedding cache. All queried drug pairs are checked, and paths retain each relationship's stored direction.

Provider factories apply YAML defaults and require the selected provider's own credentials. Notebook graph visualization emits Mermaid source without a remote rendering request. Run the shared checks from the repository root using `make test`; see the main README for disposable-database checks.
