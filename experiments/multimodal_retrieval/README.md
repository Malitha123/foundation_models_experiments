# CLIP Multimodal Retrieval

Text-to-image retrieval experiment using CLIP image embeddings, semantic text embeddings, ChromaDB, and weighted rank/score fusion.

## Data

The notebook expects `data/dermamnist_128.npz`. DermaMNIST contains 10,015 dermatoscopic images across seven classes, with 7,007 images in the official training split. Use the official MedMNIST/Zenodo distribution rather than committing the dataset to Git. The source dataset is licensed CC BY-NC 4.0.

## Run

```bash
pip install -r ../../requirements/multimodal.txt
cp ../../.env.example .env
jupyter lab clip_multimodal_retrieval.ipynb
```

The LLM portion is optional. CLIP indexing and retrieval run without an LLM API key; only the final retrieval-grounded generation demo is skipped.

## Reruns and edge cases

Paths are anchored to the configuration file, so starting Jupyter at the repository root works. Images retain their original training indices as IDs. Corrupt images are excluded consistently from both indexes. Reruns prune IDs outside the current subset, and model-specific collection names prevent embedding-dimension collisions. The manifest records both actual collection names.

Search parameters are checked before retrieval. Empty fusion and visualization results complete cleanly; zero and one fusion weights use only their selected modality. LLM generation is disabled if its optional API connectivity check fails. OpenRouter always requires its own key, even if an OpenAI key exists.
