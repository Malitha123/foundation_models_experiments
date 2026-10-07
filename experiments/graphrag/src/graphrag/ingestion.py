"""Validated and repeatable ingestion of the bundled demonstration data."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

from graphrag.schema import ensure_schema

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DrugRecord(BaseModel):
    name: Name
    aliases: list[Name] = Field(default_factory=list)
    drug_class: Name
    description: Name
    treats: list[Name] = Field(default_factory=list)
    side_effects: list[Name] = Field(default_factory=list)
    contraindications: list[Name] = Field(default_factory=list)


class InteractionRecord(BaseModel):
    drug_a: Name
    drug_b: Name
    severity: Literal["Major", "Moderate", "Minor"]
    mechanism: Name
    effect: Name
    recommendation: Name

    @model_validator(mode="after")
    def distinct_drugs(self):
        if self.drug_a.casefold() == self.drug_b.casefold():
            raise ValueError("An interaction must have two distinct drugs")
        return self


class TextPassage(BaseModel):
    source: Name
    text: Name


class DrugInteractionDataset(BaseModel):
    drugs: list[DrugRecord]
    interactions: list[InteractionRecord]
    text_passages: list[TextPassage] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_drug_names(self):
        names = [drug.name.casefold() for drug in self.drugs]
        if len(names) != len(set(names)):
            raise ValueError("Drug records must have unique names, ignoring case")
        return self


def interaction_rows(records):
    """Collapse reversed pairs, keeping every source record as evidence."""
    pairs = defaultdict(list)
    for record in records:
        pairs[tuple(sorted((record.drug_a, record.drug_b)))].append(record)
    severity_rank = {"Minor": 0, "Moderate": 1, "Major": 2}
    rows = []
    for (a, b), evidence in sorted(pairs.items()):
        row = {"drug_a": a, "drug_b": b,
               "severity": max((r.severity for r in evidence), key=severity_rank.get)}
        for key in ("mechanism", "effect", "recommendation"):
            row[key] = "\n".join(dict.fromkeys(getattr(r, key) for r in evidence))
        row["evidence_records"] = list(dict.fromkeys(
            json.dumps(r.model_dump(), sort_keys=True, ensure_ascii=False) for r in evidence
        ))
        rows.append(row)
    return rows


def seed_dataset(client, dataset):
    """Upsert all records. Clearing an entire database remains an explicit CLI action."""
    ensure_schema(client)
    client.batch_write("""
        UNWIND $batch AS row
        MERGE (d:Drug {name: row.name})
        SET d.description = row.description, d.aliases = row.aliases,
            d.metadata_status = 'complete'
    """, [dict(name=d.name, description=d.description, aliases=d.aliases) for d in dataset.drugs])
    described = {d.name for d in dataset.drugs}
    reference_only = sorted({name for r in dataset.interactions for name in (r.drug_a, r.drug_b)} - described)
    client.batch_write("""
        UNWIND $batch AS row
        MERGE (d:Drug {name: row.name})
        ON CREATE SET d.aliases = [],
            d.description = 'Referenced by an interaction record; full metadata is not included in this prototype dataset.',
            d.metadata_status = 'reference_only'
    """, [{"name": name} for name in reference_only])
    # Labels and relationship identifiers are fixed source-code constants.
    mappings = [
        ("treats", "Condition", "TREATS"),
        ("side_effects", "SideEffect", "CAUSES_SIDE_EFFECT"),
        ("contraindications", "Condition", "CONTRAINDICATED_FOR"),
        ("drug_class", "DrugClass", "BELONGS_TO_CLASS"),
    ]
    for attribute, label, rel_type in mappings:
        batch = []
        for drug in dataset.drugs:
            values = [drug.drug_class] if attribute == "drug_class" else getattr(drug, attribute)
            batch.extend({"drug": drug.name, "target": value} for value in dict.fromkeys(values))
        client.batch_write(f"""
            UNWIND $batch AS row
            MATCH (d:Drug {{name: row.drug}})
            MERGE (m:{label} {{name: row.target}})
            MERGE (d)-[:{rel_type}]->(m)
        """, batch)
    pairs = interaction_rows(dataset.interactions)
    client.batch_write("""
        UNWIND $batch AS row
        MATCH (a:Drug {name: row.drug_a}), (b:Drug {name: row.drug_b})
        MERGE (a)-[r:INTERACTS_WITH]->(b)
        SET r.severity = row.severity, r.mechanism = row.mechanism,
            r.effect = row.effect, r.recommendation = row.recommendation,
            r.evidence_records = row.evidence_records, r.source = 'structured'
        WITH a, b
        OPTIONAL MATCH (b)-[reverse:INTERACTS_WITH]->(a)
        WHERE reverse.source = 'structured'
        DELETE reverse
    """, pairs)
    return {"described_drugs": len(described), "reference_only_drugs": len(reference_only),
            "interaction_records": len(dataset.interactions), "interaction_pairs": len(pairs)}
