import types
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from experiments.multimodal_retrieval.src.retrieval import (
    load_dermamnist,encode_images,collection_names,sync_text_index,sync_image_index,
    image_search,fuse_scores,fused_search,validate_search,
)
from experiments.multimodal_retrieval.src.services import load_config,get_llm


def test_subset_ids_use_source_indices_and_cover_classes(tmp_path):
    labels=np.repeat(np.arange(7,dtype=np.int64),4)
    images=np.zeros((28,4,4,3),dtype=np.uint8)
    path=tmp_path/'data.npz';np.savez(path,train_images=images,train_labels=labels[:,None])
    a,l,indices=load_dermamnist(path,14)
    assert set(l)==set(range(7)) and len(set(indices))==14
    assert np.array_equal(l,labels[indices])
    assert np.array_equal(indices,load_dermamnist(path,14)[2])
    assert load_dermamnist(path,100)[0].shape[0]==28


@pytest.mark.parametrize('value',[0,-1,True,1.5])
def test_invalid_subset_size(tmp_path,value):
    with pytest.raises(ValueError): load_dermamnist(tmp_path/'missing',value)


def test_empty_or_bad_dataset(tmp_path):
    path=tmp_path/'bad.npz'
    np.savez(path,train_images=np.zeros((0,4,4,3),dtype=np.uint8),train_labels=np.zeros((0,1)))
    with pytest.raises(ValueError): load_dermamnist(path,7)
    np.savez(path,train_images=np.zeros((2,4,4,3),dtype=np.uint8),train_labels=np.array([0,8]))
    with pytest.raises(ValueError): load_dermamnist(path,7)


class Clip:
    def encode(self,*a,**kwargs): return np.array([1.,0.])


def test_image_alignment_with_nonpositional_dataframe_index(tmp_path):
    Image.new('RGB',(3,3)).save(tmp_path/'ok.png')
    df=pd.DataFrame([{'img_path':'missing.png'},{'img_path':'ok.png'}],index=[10,42])
    vectors,indices=encode_images(df,Clip(),tmp_path)
    assert indices==[42]
    assert df.loc[indices]['img_path'].tolist()==['ok.png']
    assert vectors.shape==(1,2)
    with pytest.raises(ValueError): encode_images(df.loc[[10]],Clip(),tmp_path)


def test_empty_fusion_single_modality_and_deterministic_ties():
    assert fuse_scores({},{} )[0]==[]
    assert fuse_scores({'b':1.,'a':1.},{},text_weight=1)[0]==[('a',1.),('b',1.)]
    assert [k for k,_ in fuse_scores({'text':.3},{'image':.5},text_weight=0)[0]]==['image']
    assert [k for k,_ in fuse_scores({'text':.3},{'image':.5},text_weight=1)[0]]==['text']


@pytest.mark.parametrize('weight',[-.1,1.1,float('nan'),float('inf'),True])
def test_invalid_fusion_weight(weight):
    with pytest.raises(ValueError): fuse_scores({}, {}, text_weight=weight)


@pytest.mark.parametrize('query,k',[('',3),('  ',3),('ok',0),('ok',-1),('ok',True),('ok',1.2)])
def test_invalid_search(query,k):
    with pytest.raises(ValueError): validate_search(query,k)


def test_model_names_are_separate_spaces():
    c={'text_emb_provider':'sbert','text_emb_model':'one','clip_model':'clip'}
    names=collection_names(c)
    assert collection_names(dict(c,text_emb_model='two'))[0]!=names[0]
    assert collection_names(dict(c,clip_model='other'))[1]!=names[1]


def test_invalid_distance_is_not_silently_converted_to_a_perfect_score():
    store=types.SimpleNamespace(similarity_search_with_score=lambda *a,**k:[
        (Document(page_content='one',metadata={'item_id':'one'}),float('nan'))])
    with pytest.raises(ValueError): fused_search(store,None,None,'query',text_weight=1)


def test_config_paths_are_relative_to_config_file(tmp_path,monkeypatch):
    path=tmp_path/'config.yaml';path.write_text('data_root: ./data\nartifacts_root: ./artifacts\n')
    monkeypatch.chdir('/')
    c=load_config(str(path))
    assert c['data_root']==str(tmp_path/'data')
    path.write_text('- wrong\n')
    with pytest.raises(ValueError): load_config(str(path))


def test_openrouter_missing_key_cannot_fall_back_to_openai(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY',raising=False)
    monkeypatch.setenv('OPENAI_API_KEY','fixture-openai-key')
    with pytest.raises(ValueError,match='OPENROUTER_API_KEY'): get_llm({'llm_provider':'openrouter'})


class TextEmbeddings(Embeddings):
    def embed_documents(self,texts): return [[1.,0.] if 'one' in text else [0.,1.] for text in texts]
    def embed_query(self,text): return [1.,0.]


def test_real_chroma_upserts_prune_stale_ids_and_cap_queries(tmp_path):
    import chromadb
    from langchain_chroma import Chroma
    client=chromadb.PersistentClient(path=str(tmp_path/'images'))
    collection=client.get_or_create_collection('test_images',metadata={'hnsw:space':'cosine'},embedding_function=None)
    sync_image_index(collection,['old','keep'],[[0.,1.],[1.,0.]],[{'item_id':'old'},{'item_id':'keep'}])
    sync_image_index(collection,['keep'],[[1.,0.]],[{'item_id':'keep'}])
    assert collection.get()['ids']==['keep']
    assert image_search(collection,Clip(),'query',20)['ids']==[['keep']]
    store=Chroma(collection_name='test_text',embedding_function=TextEmbeddings(),persist_directory=str(tmp_path/'text'),collection_metadata={'hnsw:space':'cosine'})
    docs=[Document(page_content='one',metadata={'item_id':'keep'}),Document(page_content='two',metadata={'item_id':'old'})]
    sync_text_index(store,docs,['keep','old'])
    sync_text_index(store,docs[:1],['keep'])
    assert store.get()['ids']==['keep']
    ranked,_,_=fused_search(store,collection,Clip(),'query',20)
    assert ranked==[('keep',1.)]
    collection.delete(ids=['keep'])
    assert image_search(collection,Clip(),'query')['ids']==[[]]
