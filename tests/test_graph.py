import json
from pathlib import Path
import types
import numpy as np
import pytest
from pydantic import ValidationError
from graphrag import config
from graphrag.ingestion import DrugInteractionDataset, InteractionRecord, interaction_rows
from graphrag.neo4j_client import Neo4jClient, _quote_identifier, get_neo4j_client
from graphrag.retrieval import EntityResolver, GraphRetriever, ResolvedEntity, VectorRetriever
from graphrag.llm_services import get_general_llm,get_embeddings
ROOT=Path(__file__).resolve().parents[1]


def test_reversed_interactions_preserve_every_source_record():
    raw=json.loads((ROOT/'experiments/graphrag/data/raw/drug_interactions.json').read_text())
    dataset=DrugInteractionDataset.model_validate(raw)
    rows=interaction_rows(dataset.interactions)
    assert len(rows)==227
    assert sum(len(row['evidence_records']) for row in rows)==244
    assert all(row['drug_a']<row['drug_b'] for row in rows)
    assert interaction_rows(list(reversed(dataset.interactions)))[0]['drug_a']==rows[0]['drug_a']


@pytest.mark.parametrize('change',[{'severity':'Critical'},{'drug_a':''},{'drug_b':'a'}])
def test_invalid_interaction_data(change):
    row={'drug_a':'a','drug_b':'b','severity':'Major','mechanism':'m','effect':'e','recommendation':'r'}
    with pytest.raises(ValidationError): InteractionRecord.model_validate(dict(row,**change))


class Session:
    def __init__(self): self.runs=[];self.consumed=False;self.write_calls=0
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def run(self,cypher,parameters):
        self.runs.append((cypher,parameters));return self
    def consume(self): self.consumed=True
    def execute_write(self,fn): self.write_calls+=1;return fn(self)


def test_neo4j_writes_consume_results_and_use_managed_transactions():
    session=Session();client=object.__new__(Neo4jClient);client._database='neo4j'
    client._driver=types.SimpleNamespace(session=lambda **kwargs:session)
    client.write('RETURN $parameters',{'parameters':7})
    assert session.consumed and session.write_calls==1
    assert session.runs[-1][1]=={'parameters':7}
    assert client.batch_write('UNWIND $batch AS row RETURN row',[{'value':1}])==1
    assert client.batch_write('ignored',[])==0


def test_schema_identifiers_escape_backticks():
    assert _quote_identifier('Drug`name')=='`Drug``name`'


def test_credentials_read_after_import(monkeypatch):
    monkeypatch.setenv('NEO4J_URI','bolt://test-after-import')
    monkeypatch.setenv('NEO4J_PASSWORD','new-password')
    assert config.get_graph_credentials()['uri']=='bolt://test-after-import'
    assert config.get_graph_credentials()['password']=='new-password'


def test_llm_config_defaults_and_provider_key(monkeypatch):
    import graphrag.llm_services as services
    seen=[]
    monkeypatch.setattr(services,'ChatOpenAI',lambda **kwargs:seen.append(kwargs) or kwargs)
    monkeypatch.setattr(services,'OpenAIEmbeddings',lambda **kwargs:kwargs)
    monkeypatch.setenv('OPENROUTER_API_KEY','fixture')
    llm=get_general_llm()
    assert llm['max_tokens']==2000 and llm['timeout']==60
    assert get_general_llm(temperature=.6)['temperature']==.6
    assert get_embeddings()['chunk_size']==100
    monkeypatch.delenv('OPENROUTER_API_KEY')
    monkeypatch.setenv('OPENAI_API_KEY','different-provider')
    with pytest.raises(ValueError,match='OPENROUTER_API_KEY'): get_general_llm()


@pytest.mark.parametrize('provider,tier',[('google','general'),('openai','missing')])
def test_invalid_models_fail_instead_of_provider_mismatch(provider,tier):
    with pytest.raises(ValueError): config.get_chat_model(provider,tier)


class NamesClient:
    def __init__(self): self.calls=[]
    def query(self,cypher,params=None):
        self.calls.append((cypher,params))
        if 'toLower(n.name) =' in cypher:
            return [{'name':'Warfarin','labels':['Drug']}] if params['name'].casefold() in {'warfarin','coumadin'} else []
        return [{'name':'Warfarin','labels':['Drug'],'aliases':['Coumadin']},
                {'name':'Same','labels':['Drug'],'aliases':[]}, {'name':'Same','labels':['Condition'],'aliases':[]}]


def test_aliases_typos_and_blank_entities():
    c=NamesClient();r=EntityResolver(c)
    assert r.resolve(' Coumadin ')[0].match_type=='exact'
    assert r.resolve('Warfari')[0].canonical_name=='Warfarin'
    assert r.resolve('   ')==[]
    assert len(r.resolve_all(['Warfarin','warfarin','Coumadin']))==1
    with pytest.raises(ValueError): r.resolve('drug','Drug` malicious')


def test_semantic_cache_preserves_equal_names_across_labels():
    embeddings=types.SimpleNamespace(embed_documents=lambda names:[[1.,0.] for _ in names],embed_query=lambda q:[1.,0.])
    r=EntityResolver(NamesClient(),embeddings)
    assert r._semantic_match('synonym',label='Condition')[0].label=='Condition'
    assert len(r._embedding_cache)==3
    r.clear_cache();assert r._embedding_cache is None


def test_all_drug_pairs_and_no_sentinel_evidence():
    resolver=types.SimpleNamespace(extract_entities_from_question=lambda *a,**k:['a','b','c'],
                                  resolve_all=lambda terms:[ResolvedEntity(x,'Drug',1.,'exact') for x in terms])
    c=types.SimpleNamespace(query=lambda *args:[])
    retriever=GraphRetriever(c,resolver);pairs=[]
    retriever.retrieve_interaction=lambda a,b:pairs.append((a,b)) or []
    assert retriever.retrieve_for_question('three drugs')==[]
    assert pairs==[('a','b'),('a','c'),('b','c')]


@pytest.mark.parametrize('hops',[0,-1,True,'2] malicious',5])
def test_hop_limits_are_checked(hops):
    r=GraphRetriever(None,None)
    with pytest.raises(ValueError): r.retrieve_multi_hop('Drug',hops)


def test_rel_type_filters_are_parameters():
    calls=[];r=GraphRetriever(types.SimpleNamespace(query=lambda q,p:calls.append((q,p)) or []),None)
    r.retrieve_single_hop('Warfarin',['TREATS'])
    assert calls[0][1]['rel_types']==['TREATS']
    assert "'TREATS'" not in calls[0][0]
    with pytest.raises(ValueError): r.retrieve_single_hop('Warfarin',["x' OR true"])


def test_multihop_output_keeps_stored_direction():
    text=GraphRetriever._format_result({'node_names':['Condition','Drug'],
            'edges':[{'source':'Drug','target':'Condition','rel_type':'TREATS','rel_props':{}}]})
    assert 'Drug --[TREATS]--> Condition' in text
    assert 'Condition --[TREATS]--> Drug' not in text


def test_vector_index_empty_rebuild_clears_previous_passages():
    embedding=types.SimpleNamespace(embed_documents=lambda x:[[1.,0.] for _ in x],embed_query=lambda x:[1.,0.])
    r=VectorRetriever(embedding);r.index_passages(['a'])
    assert r.retrieve('q')[0].page_content=='a'
    r.index_passages([]);assert r.retrieve('q')==[]
    with pytest.raises(ValueError): r.retrieve('q',0)
