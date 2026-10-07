.PHONY: help install-finetune install-multimodal install-graphrag notebooks test install-test clean

help:
	@echo "Foundation Model + GenAI Experiments"
	@echo "  make install-finetune    Install QLoRA dependencies"
	@echo "  make install-multimodal  Install CLIP retrieval dependencies"
	@echo "  make install-graphrag    Install Neo4j GraphRAG dependencies"
	@echo "  make notebooks           Launch Jupyter Lab"
	@echo "  make test                Run CPU regression checks"
	@echo "  make clean               Remove local caches"

install-finetune:
	pip install -r requirements/finetuning.txt

install-multimodal:
	pip install -r requirements/multimodal.txt

install-graphrag:
	pip install -r requirements/graphrag.txt

notebooks:
	jupyter lab experiments/

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true

install-test:
	python -m pip install -r requirements/test.txt

test:
	python -m pytest -q
