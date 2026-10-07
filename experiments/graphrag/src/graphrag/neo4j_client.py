"""
Neo4j Aura client — context-manager interface for the knowledge graph.

All queries use parameterized Cypher (never string interpolation for values).
Supports batch writes via UNWIND for efficient bulk ingestion.
"""

from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase
from loguru import logger


class Neo4jClient:
    """Thread-safe Neo4j driver wrapper with context-manager support."""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self._uri = uri
        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    # -- Context Manager --

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -- Connection --

    def verify_connectivity(self) -> bool:
        """Test that the driver can reach the server."""
        try:
            self._driver.verify_connectivity()
            logger.success(f"Connected to Neo4j at {self._uri}")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False

    def close(self):
        """Release the driver and all pooled connections."""
        self._driver.close()

    # -- Read --

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a read query and return list of record dicts."""
        params = params or {}
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, parameters=params)
            return [dict(record) for record in result]

    # -- Write --

    def write(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Execute a single write statement."""
        params = params or {}
        with self._driver.session(database=self._database) as session:
            session.execute_write(lambda tx: tx.run(cypher, parameters=params).consume())

    def batch_write(self, cypher: str, param_list: List[Dict[str, Any]]) -> int:
        """Execute a write for each param dict in the list using UNWIND.

        The cypher should reference `$batch` as an UNWIND variable.
        Example:
            UNWIND $batch AS row
            MERGE (d:Drug {name: row.name})
            SET d.description = row.description

        Returns the number of items processed.
        """
        if not param_list:
            return 0
        with self._driver.session(database=self._database) as session:
            session.execute_write(lambda tx: tx.run(cypher, parameters={"batch": param_list}).consume())
        return len(param_list)

    # -- Introspection --

    def get_node_count(self, label: Optional[str] = None) -> int:
        """Count nodes, optionally filtered by label."""
        if label:
            result = self.query(f"MATCH (n:{_quote_identifier(label)}) RETURN count(n) AS cnt")
        else:
            result = self.query("MATCH (n) RETURN count(n) AS cnt")
        return result[0]["cnt"] if result else 0

    def get_relationship_count(self, rel_type: Optional[str] = None) -> int:
        """Count relationships, optionally filtered by type."""
        if rel_type:
            result = self.query(f"MATCH ()-[r:{_quote_identifier(rel_type)}]->() RETURN count(r) AS cnt")
        else:
            result = self.query("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return result[0]["cnt"] if result else 0

    def get_schema_info(self) -> Dict[str, Any]:
        """Introspect the graph for labels, relationship types, and counts."""
        labels = self.query("CALL db.labels() YIELD label RETURN label")
        rel_types = self.query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")

        info = {
            "labels": {},
            "relationship_types": {},
            "total_nodes": self.get_node_count(),
            "total_relationships": self.get_relationship_count(),
        }

        for row in labels:
            lbl = row["label"]
            info["labels"][lbl] = self.get_node_count(lbl)

        for row in rel_types:
            rt = row["relationshipType"]
            info["relationship_types"][rt] = self.get_relationship_count(rt)

        return info


def get_neo4j_client() -> Neo4jClient:
    """Factory: create a Neo4jClient from environment config."""
    from graphrag.config import get_graph_credentials
    credentials = get_graph_credentials()

    if not credentials["uri"]:
        raise ValueError("NEO4J_URI is not set. Check your .env file.")
    if not credentials["password"]:
        raise ValueError("NEO4J_PASSWORD is not set. Check your .env file.")

    return Neo4jClient(**credentials)


def _quote_identifier(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Schema identifier must be a nonempty string")
    return "`" + value.replace("`", "``") + "`"
