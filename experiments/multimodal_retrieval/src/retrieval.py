"""Dataset validation, repeatable indexing, and score fusion without model downloads."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def validate_search(query, top_k):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")


def load_dermamnist(path, subset_n, seed=42):
    if isinstance(subset_n, bool) or not isinstance(subset_n, int) or subset_n < 1:
        raise ValueError("subset_n must be a positive integer")
    with np.load(path, allow_pickle=False) as data:
        images, labels = data['train_images'], data['train_labels'].reshape(-1)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError("train_images must be an N x H x W x 3 uint8 RGB array")
    if not len(images) or len(images) != len(labels):
        raise ValueError("Dataset must be nonempty, with one label per image")
    if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, range(7)).all():
        raise ValueError("DermaMNIST labels must be integers in 0..6")
    n = min(subset_n, len(images))
    rng = np.random.default_rng(seed)
    classes = rng.permutation(np.unique(labels))
    chosen = []
    per_class = max(1, n // len(classes))
    for label in classes:
        pool = np.flatnonzero(labels == label)
        take = min(per_class, len(pool), n - len(chosen))
        if take:
            chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    remaining = np.setdiff1d(np.arange(len(images)), chosen)
    if len(chosen) < n:
        chosen.extend(rng.choice(remaining, size=n-len(chosen), replace=False).tolist())
    rng.shuffle(chosen)
    indices = np.array(chosen, dtype=int)
    return images[indices], labels[indices], indices


def encode_images(df, clip_model, data_root):
    """Keep DataFrame index labels aligned with their vectors, even after filtering."""
    embeddings, labels = [], []
    for idx, row in df.iterrows():
        path = Path(data_root) / row['img_path']
        try:
            with Image.open(path) as image:
                rgb = image.convert('RGB')
        except (OSError, ValueError) as exc:
            print(f"Skipping unreadable image {path.name}: {exc}")
            continue
        vector = np.asarray(clip_model.encode(rgb, convert_to_numpy=True, normalize_embeddings=True), dtype=float)
        if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError(f"Invalid CLIP embedding for {path.name}")
        embeddings.append(vector)
        labels.append(idx)
    if not embeddings:
        raise ValueError("No readable images were available to index")
    return np.stack(embeddings), labels


def collection_names(config):
    """Keep incompatible model spaces in different persistent collections."""
    def digest(values):
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]
    text = digest([config['text_emb_provider'], config['text_emb_model'], config.get('normalize_embeddings', True)])
    image = digest([config['clip_model'], 'normalized-cosine-v1'])
    return f'health_text_{text}', f'health_image_{image}'


def sync_text_index(store, documents, ids):
    if not ids or len(ids) != len(documents) or len(ids) != len(set(ids)):
        raise ValueError("Text documents require nonempty, unique, matching IDs")
    store.add_documents(documents=documents, ids=ids)
    stale = sorted(set(store.get(include=[])['ids']) - set(ids))
    if stale:
        store.delete(ids=stale)


def sync_image_index(collection, ids, vectors, metadatas):
    vectors = np.asarray(vectors)
    if not ids or len(ids) != len(set(ids)) or len(ids) != len(metadatas):
        raise ValueError("Image metadata requires nonempty, unique, matching IDs")
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or not np.isfinite(vectors).all():
        raise ValueError("Image vectors must be a finite matrix matching the IDs")
    collection.upsert(ids=ids, embeddings=vectors.tolist(), metadatas=metadatas)
    stale = sorted(set(collection.get(include=[])['ids']) - set(ids))
    if stale:
        collection.delete(ids=stale)


def image_search(collection, clip_model, query_text, top_k=3, category_filter=None):
    validate_search(query_text, top_k)
    if collection.count() == 0:
        return {'ids': [[]], 'metadatas': [[]], 'distances': [[]]}
    vector = np.asarray(clip_model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True), dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError("CLIP query must produce a finite, nonzero vector")
    options = {'where': {'category': category_filter}} if category_filter else {}
    return collection.query(query_embeddings=[vector.tolist()], n_results=min(top_k, collection.count()), **options)


def normalize_scores(scores):
    if not scores:
        return {}
    if not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("Retrieval scores must be finite")
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {key: 1.0 for key in scores}
    return {key: (value-lo)/(hi-lo) for key, value in scores.items()}


def fuse_scores(text_scores, image_scores, top_k=3, text_weight=0.5):
    validate_search('fusion', top_k)
    if isinstance(text_weight, bool) or not isinstance(text_weight, (int, float)) or not math.isfinite(text_weight) or not 0 <= text_weight <= 1:
        raise ValueError("text_weight must be finite and between 0 and 1")
    text = normalize_scores(text_scores) if text_weight > 0 else {}
    image = normalize_scores(image_scores) if text_weight < 1 else {}
    scores = {key: text_weight*text.get(key, 0)+(1-text_weight)*image.get(key, 0) for key in text.keys() | image.keys()}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return ranked, text, image


def fused_search(store, collection, clip_model, query, top_k=3, text_weight=0.5):
    validate_search(query, top_k)
    # Validate weights before making provider/model calls.
    fuse_scores({}, {}, top_k, text_weight)
    # Both stores use cosine distance, where lower is better. Avoid Chroma's
    # L2-specific relevance conversion on unnormalized or cosine vectors.
    text = {}
    if text_weight > 0:
        text = {doc.metadata['item_id']: _distance_score(distance)
                for doc, distance in store.similarity_search_with_score(query, k=top_k*2)}
    images = {}
    if text_weight < 1:
        results = image_search(collection, clip_model, query, top_k*2)
        images = {key: _distance_score(distance)
                  for key, distance in zip(results['ids'][0], results['distances'][0])}
    return fuse_scores(text, images, top_k, text_weight)


def _distance_score(distance):
    distance = float(distance)
    if not math.isfinite(distance):
        raise ValueError("Retrieval distances must be finite")
    return 1 / (1 + max(0, distance))
