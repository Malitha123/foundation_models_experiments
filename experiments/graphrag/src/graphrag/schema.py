"""One schema definition shared by CLI scripts and notebooks."""

NODE_LABELS = frozenset({"Drug", "Condition", "SideEffect", "DrugClass"})
RELATIONSHIP_TYPES = frozenset({
    "INTERACTS_WITH", "TREATS", "CAUSES_SIDE_EFFECT", "CONTRAINDICATED_FOR", "BELONGS_TO_CLASS",
})

CONSTRAINT_QUERIES = [
    "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT condition_name IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT side_effect_name IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT drug_class_name IF NOT EXISTS FOR (dc:DrugClass) REQUIRE dc.name IS UNIQUE",
]
# Uniqueness constraints already create indexes on these exact properties.
INDEX_QUERIES = []


def ensure_schema(client):
    for query in CONSTRAINT_QUERIES:
        client.write(query)
