import ast
import json
import os
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import nbformat
import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS=sorted((ROOT/'experiments').rglob('*.ipynb'))


@pytest.mark.parametrize('path',NOTEBOOKS,ids=lambda p:p.stem)
def test_notebook_schema_and_all_code_cells_compile(path):
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':
            ast.parse(cell.source)
            assert not cell.outputs and cell.execution_count is None


@pytest.mark.parametrize('experiment,cell_index',[
    ('multimodal_retrieval',2),('graphrag',1),('qlora_finetuning',4),
])
def test_startup_from_repository_root(tmp_path,monkeypatch,experiment,cell_index):
    import experiments.multimodal_retrieval.src.services as mm
    monkeypatch.chdir(ROOT)
    if experiment=='graphrag':
        path=ROOT/'experiments/graphrag/notebooks/knowledge_graph_etl.ipynb'
        monkeypatch.setenv('NEO4J_URI','bolt://fixture')
        monkeypatch.setenv('NEO4J_PASSWORD','fixture')
    elif experiment=='multimodal_retrieval':
        path=ROOT/'experiments/multimodal_retrieval/clip_multimodal_retrieval.ipynb'
    else:
        path=ROOT/'experiments/qlora_finetuning/qlora_medical_finetuning.ipynb'
    source=''.join(json.loads(path.read_text())['cells'][cell_index]['source'])
    scope={};exec(compile(source,'startup cell','exec'),scope)
    assert scope['PROJECT_ROOT']==ROOT/'experiments'/experiment


class TextEmbeddings(Embeddings):
    def embed_documents(self,texts): return [[1.,0.] if 'Basal' in text else [0.,1.] for text in texts]
    def embed_query(self,text): return [1.,0.]


class Clip:
    def encode(self,subject,**kwargs): return np.array([1.,0.])


def test_multimodal_run_all_without_paid_apis_or_downloads(tmp_path,monkeypatch):
    """Execute every actual notebook cell, replacing only external model factories."""
    import matplotlib
    matplotlib.use('Agg')
    import experiments.multimodal_retrieval.src.services as services
    config=services.load_config()
    config.update(data_root=str(tmp_path/'data'),artifacts_root=str(tmp_path/'artifacts'),subset_n=7)
    (tmp_path/'data').mkdir()
    np.savez(tmp_path/'data/dermamnist_128.npz',train_images=np.zeros((14,8,8,3),dtype=np.uint8),
             train_labels=np.repeat(np.arange(7),2)[:,None])
    monkeypatch.setattr(services,'load_config',lambda *args:config)
    monkeypatch.setattr(services,'get_clip_model',lambda cfg:Clip())
    monkeypatch.setattr(services,'get_text_embeddings',lambda cfg:TextEmbeddings())
    monkeypatch.delenv('OPENROUTER_API_KEY',raising=False)
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    monkeypatch.chdir(ROOT)
    path=ROOT/'experiments/multimodal_retrieval/clip_multimodal_retrieval.ipynb'
    nb=json.loads(path.read_text());scope={}
    with redirect_stdout(StringIO()):
        for i,cell in enumerate(nb['cells']):
            if cell['cell_type']=='code': exec(compile(''.join(cell['source']),f'multimodal cell {i}','exec'),scope)
    assert scope['llm'] is None
    assert len(scope['df'])==7
    assert scope['image_collection'].count()==7
    saved=json.loads((tmp_path/'artifacts/manifests/health.json').read_text())
    assert saved['image_collection_name']==scope['image_collection_name']
    # Empty visualization and no-context generation also finish cleanly.
    assert scope['visualize_results']([]) is None
    empty=lambda *a,**kwargs:([],{}, {})
    scope['fused_search']=empty
    assert scope['multimodal_rag']('missing')==('No relevant items were retrieved.',[])
