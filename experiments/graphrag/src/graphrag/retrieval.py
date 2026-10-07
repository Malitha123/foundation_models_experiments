"""Entity resolution and bounded graph/vector retrieval used by notebook 02."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from typing import Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graphrag.config import get_config
from graphrag.llm_services import get_strong_llm
from graphrag.prompts import ENTITY_EXTRACTION_PROMPT
from graphrag.schema import NODE_LABELS, RELATIONSHIP_TYPES


def _positive_int(value, name, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or (maximum and value > maximum):
        raise ValueError(f'{name} must be a positive integer' + (f' at most {maximum}' if maximum else ''))
    return value


def _label(label):
    if label is not None and label not in NODE_LABELS:
        raise ValueError(f'Unsupported entity label: {label}')


@dataclass
class ResolvedEntity:
    canonical_name: str
    label: str
    score: float
    match_type: str


class ExtractedQueryEntities(BaseModel):
    entities: list[str] = Field(description='Drug, condition, side-effect, or drug-class names from the question')


class EntityResolver:
    def __init__(self, client, embeddings=None):
        self._client, self._embeddings = client, embeddings
        self._name_cache = None
        self._embedding_cache = None
        settings = get_config().get('retrieval', {})
        self._fuzzy_threshold = settings.get('fuzzy_match_threshold', 0.8)
        self._semantic_threshold = settings.get('similarity_threshold', 0.7)
        for value in (self._fuzzy_threshold, self._semantic_threshold):
            if not 0 <= value <= 1:
                raise ValueError('Matching thresholds must be between 0 and 1')

    def clear_cache(self):
        """Call after changing graph nodes in a long-lived notebook session."""
        self._name_cache = self._embedding_cache = None

    def extract_entities_from_question(self, question, llm=None):
        if not isinstance(question, str) or not question.strip():
            return []
        llm = get_strong_llm() if llm is None else llm
        prompt = ChatPromptTemplate.from_messages([('system', ENTITY_EXTRACTION_PROMPT), ('human', '{question}')])
        result = (prompt | llm.with_structured_output(ExtractedQueryEntities)).invoke({'question': question})
        return list(dict.fromkeys(term.strip() for term in result.entities if term.strip()))

    def resolve(self, query_entity, label=None):
        _label(label)
        if not isinstance(query_entity, str) or not query_entity.strip():
            return []
        query_entity = query_entity.strip()
        return (self._exact_match(query_entity, label) or self._fuzzy_match(query_entity, label)
                or (self._semantic_match(query_entity, label) if self._embeddings is not None else []))

    def resolve_all(self, entities):
        resolved, seen = [], set()
        for entity in entities:
            matches = self.resolve(entity)
            # Preserve equal exact matches across labels rather than silently losing
            # a condition/drug-class with the same name as another node.
            for match in matches:
                if match.score < matches[0].score:
                    break
                key = (match.canonical_name, match.label)
                if key not in seen:
                    seen.add(key)
                    resolved.append(match)
        return resolved

    def _exact_match(self, query, label=None):
        rows = self._client.query('''
            MATCH (n)
            WHERE ($label IS NULL OR $label IN labels(n))
              AND any(lbl IN labels(n) WHERE lbl IN $labels)
              AND (toLower(n.name) = toLower($name)
                OR any(alias IN coalesce(n.aliases, []) WHERE toLower(alias) = toLower($name)))
            RETURN n.name AS name, labels(n) AS labels
            ORDER BY name
        ''', {'name': query, 'label': label, 'labels': sorted(NODE_LABELS)})
        return [ResolvedEntity(row['name'], lbl, 1.0, 'exact')
                for row in rows for lbl in sorted(set(row['labels']) & NODE_LABELS)
                if label is None or label == lbl]

    def _names(self):
        if self._name_cache is None:
            self._name_cache = self._client.query('''
                MATCH (n) WHERE n.name IS NOT NULL
                  AND any(lbl IN labels(n) WHERE lbl IN $labels)
                RETURN n.name AS name, labels(n) AS labels, coalesce(n.aliases, []) AS aliases
                ORDER BY name
            ''', {'labels': sorted(NODE_LABELS)})
        return self._name_cache

    def _fuzzy_match(self, query, label=None):
        if len(query) < 3:
            return []
        scored = []
        for row in self._names():
            score = max(SequenceMatcher(None, query.casefold(), term.casefold()).ratio()
                        for term in [row['name'], *row.get('aliases', [])])
            if score >= self._fuzzy_threshold:
                for lbl in sorted(set(row['labels']) & NODE_LABELS):
                    if label is None or label == lbl:
                        scored.append(ResolvedEntity(row['name'], lbl, score, 'fuzzy'))
        return sorted(scored, key=lambda match: (-match.score, match.canonical_name, match.label))[:5]

    def _semantic_match(self, query, label=None, top_k=3):
        _positive_int(top_k, 'top_k')
        if self._embedding_cache is None:
            rows = self._names()
            vectors = self._embeddings.embed_documents([row['name'] for row in rows]) if rows else []
            if len(vectors) != len(rows):
                raise ValueError('Embedding provider returned the wrong number of vectors')
            self._embedding_cache = [(row, np.asarray(vector, dtype=float)) for row, vector in zip(rows, vectors)]
        if not self._embedding_cache:
            return []
        q = np.asarray(self._embeddings.embed_query(query), dtype=float)
        if q.ndim != 1 or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
            return []
        matches = []
        for row, vector in self._embedding_cache:
            if vector.shape != q.shape or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                continue
            score = float(np.dot(q, vector)/(np.linalg.norm(q)*np.linalg.norm(vector)))
            if score >= self._semantic_threshold:
                for lbl in sorted(set(row['labels']) & NODE_LABELS):
                    if label is None or label == lbl:
                        matches.append(ResolvedEntity(row['name'], lbl, score, 'semantic'))
        return sorted(matches, key=lambda match: (-match.score, match.canonical_name, match.label))[:top_k]


class GraphRetriever:
    def __init__(self, client, entity_resolver, top_k=None):
        self._client, self._resolver = client, entity_resolver
        defaults = get_config().get('retrieval', {})
        self.top_k = _positive_int(defaults.get('top_k', 5) if top_k is None else top_k, 'top_k')
        self.max_hops = _positive_int(defaults.get('max_hops', 2), 'max_hops', 4)

    def retrieve_single_hop(self, entity, rel_types=None, label=None):
        _label(label)
        if rel_types is not None and not set(rel_types) <= RELATIONSHIP_TYPES:
            raise ValueError('Unsupported relationship type')
        return self._client.query('''
            MATCH (n)-[r]-(m)
            WHERE n.name = $name AND ($label IS NULL OR $label IN labels(n))
              AND ($rel_types IS NULL OR type(r) IN $rel_types)
            RETURN DISTINCT startNode(r).name AS source, labels(startNode(r)) AS source_labels,
                   type(r) AS rel_type, properties(r) AS rel_props,
                   endNode(r).name AS target, labels(endNode(r)) AS target_labels
            ORDER BY rel_type, source, target
            LIMIT $limit
        ''', {'name': entity, 'label': label, 'rel_types': rel_types, 'limit': self.top_k*3})

    def retrieve_multi_hop(self, entity, hops=None, label=None):
        _label(label)
        hops = self.max_hops if hops is None else _positive_int(hops, 'hops', 4)
        # Cypher's path upper bound cannot be parameterized. Only validated integers
        # are inserted; all user values stay in the parameters dictionary.
        return self._client.query(f'''
            MATCH path = (n)-[*2..{hops}]-(m)
            WHERE n.name = $name AND n <> m AND ($label IS NULL OR $label IN labels(n))
            WITH path, m
            RETURN [node IN nodes(path) | node.name] AS node_names,
                   [rel IN relationships(path) | {{source: startNode(rel).name,
                     target: endNode(rel).name, rel_type: type(rel), rel_props: properties(rel)}}] AS edges,
                   m.name AS endpoint, labels(m) AS endpoint_labels, length(path) AS path_length
            ORDER BY path_length, endpoint, node_names
            LIMIT $limit
        ''', {'name': entity, 'label': label, 'limit': self.top_k*2}) if hops > 1 else []

    def retrieve_interaction(self, drug_a, drug_b):
        if drug_a == drug_b:
            return []
        results = []
        queries = {
            'direct': '''MATCH (a:Drug)-[r:INTERACTS_WITH]-(b:Drug)
                WHERE a.name = $a AND b.name = $b
                RETURN a.name AS drug_a, b.name AS drug_b, r.severity AS severity,
                       r.mechanism AS mechanism, r.effect AS effect, r.recommendation AS recommendation''',
            'shared_side_effects': '''MATCH (a:Drug)-[:CAUSES_SIDE_EFFECT]->(se:SideEffect)<-[:CAUSES_SIDE_EFFECT]-(b:Drug)
                WHERE a.name = $a AND b.name = $b
                RETURN a.name AS drug_a, b.name AS drug_b, collect(DISTINCT se.name) AS shared_side_effects''',
            'same_class': '''MATCH (a:Drug)-[:BELONGS_TO_CLASS]->(dc:DrugClass)<-[:BELONGS_TO_CLASS]-(b:Drug)
                WHERE a.name = $a AND b.name = $b
                RETURN a.name AS drug_a, b.name AS drug_b, collect(DISTINCT dc.name) AS shared_classes''',
        }
        for kind, query in queries.items():
            results.extend(dict(row, interaction_type=kind) for row in self._client.query(query, {'a': drug_a, 'b': drug_b}))
        return results

    def retrieve_for_question(self, question, llm=None, max_hops=None):
        hops = self.max_hops if max_hops is None else _positive_int(max_hops, 'max_hops', 4)
        raw = self._resolver.extract_entities_from_question(question, llm=llm)
        resolved = self._resolver.resolve_all(raw)
        if not resolved:
            return []
        results = []
        drugs = list(dict.fromkeys(r.canonical_name for r in resolved if r.label == 'Drug'))
        for a, b in combinations(drugs, 2):
            results.extend(self.retrieve_interaction(a, b))
        for entity in resolved:
            results.extend(self.retrieve_single_hop(entity.canonical_name, label=entity.label))
            if hops > 1:
                results.extend(self.retrieve_multi_hop(entity.canonical_name, hops, label=entity.label))
        documents, seen = [], set()
        for row in results:
            text = self._format_result(row)
            if text and text not in seen:
                seen.add(text)
                documents.append(Document(page_content=text, metadata={'source': 'knowledge_graph'}))
        return documents[:self.top_k*3]

    @staticmethod
    def _format_result(result):
        if 'rel_type' in result:
            line = f"{result['source']} --[{result['rel_type']}]--> {result['target']}"
            details = [f'  {key}: {value}' for key, value in result.get('rel_props', {}).items()
                       if value and key not in {'source', 'evidence_records'}]
            return '\n'.join([line, *details])
        if 'drug_a' in result:
            kind = result.get('interaction_type', '')
            heading = 'Recorded interaction' if kind == 'direct' else 'Shared graph attributes, not evidence of an interaction'
            parts = [f"{heading}: {result['drug_a']} and {result['drug_b']} ({kind})"]
            parts.extend(f'  {key}: {result[key]}' for key in ('severity','mechanism','effect','recommendation','shared_side_effects','shared_classes') if result.get(key))
            return '\n'.join(parts)
        if 'node_names' in result:
            parts = ['Graph path (each edge retains its stored direction): ' + ' / '.join(result['node_names'])]
            parts.extend(GraphRetriever._format_result(edge) for edge in result.get('edges', []))
            return '\n'.join(parts)
        return ''


class VectorRetriever:
    def __init__(self, embeddings):
        self._embeddings = embeddings
        self._documents = []
        self._vectors = None

    def index_passages(self, passages):
        if not passages:
            self._documents, self._vectors = [], None
            return
        vectors = np.asarray(self._embeddings.embed_documents(passages), dtype=float)
        if vectors.ndim != 2 or len(vectors) != len(passages) or not np.isfinite(vectors).all():
            raise ValueError('Embedding vectors must be a finite matrix matching the passages')
        self._documents, self._vectors = list(passages), vectors

    def retrieve(self, query, top_k=5):
        _positive_int(top_k, 'top_k')
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be a nonempty string')
        if not self._documents:
            return []
        vector = np.asarray(self._embeddings.embed_query(query), dtype=float)
        if vector.shape != (self._vectors.shape[1],) or not np.isfinite(vector).all():
            raise ValueError('Query embedding must match the index dimension and be finite')
        if np.linalg.norm(vector) == 0:
            return []
        norms = np.linalg.norm(self._vectors, axis=1)*np.linalg.norm(vector)
        scores = np.divide(self._vectors @ vector, norms, out=np.full(len(norms), -np.inf), where=norms>0)
        indices = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))[:top_k]
        return [Document(page_content=self._documents[idx], metadata={'source': 'vector_search', 'score': round(float(scores[idx]), 4)})
                for idx in indices if np.isfinite(scores[idx])]
