"""Opt-in checks for a disposable database, never the user's normal Neo4j URI."""
import json
import os
from pathlib import Path
import pytest
from graphrag.ingestion import DrugInteractionDataset,seed_dataset
from graphrag.neo4j_client import Neo4jClient
from graphrag.retrieval import EntityResolver,GraphRetriever
from graphrag.schema import ensure_schema

pytestmark=pytest.mark.live_neo4j
URI=os.getenv('GRAPHRAG_TEST_URI')
ALLOW_CLEAR=os.getenv('GRAPHRAG_TEST_ALLOW_CLEAR')=='1'
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def graph():
    if not URI or not ALLOW_CLEAR:
        pytest.skip('Set GRAPHRAG_TEST_URI and GRAPHRAG_TEST_ALLOW_CLEAR=1 for a disposable database')
    with Neo4jClient(URI,'neo4j','test-only') as client:
        assert client.verify_connectivity()
        client.write('MATCH (n) DETACH DELETE n')
        client.write("CREATE (:QAOnly {name: 'preserved'})")
        dataset=DrugInteractionDataset.model_validate(json.loads((ROOT/'experiments/graphrag/data/raw/drug_interactions.json').read_text()))
        counts=seed_dataset(client,dataset)
        yield client,dataset,counts
        client.write('MATCH (n) DETACH DELETE n')


def test_live_schema_seed_and_rerun_preserve_data(graph):
    client,dataset,counts=graph
    before=client.get_schema_info()
    ensure_schema(client);seed_dataset(client,dataset)
    assert client.get_schema_info()==before
    assert client.query("MATCH (n:QAOnly {name:'preserved'}) RETURN count(n) AS count")[0]['count']==1
    assert client.get_relationship_count('INTERACTS_WITH')==227
    assert client.query('MATCH ()-[r:INTERACTS_WITH]->() RETURN sum(size(r.evidence_records)) AS count')[0]['count']==244
    assert counts['interaction_pairs']==227


def test_live_all_query_shapes_and_alias_typo_resolution(graph):
    client,_,_=graph
    resolver=EntityResolver(client)
    assert resolver.resolve('coumadin')[0].canonical_name=='Warfarin'
    assert resolver.resolve('Warfari')[0].canonical_name=='Warfarin'
    retriever=GraphRetriever(client,resolver)
    assert retriever.retrieve_single_hop('Warfarin',['TREATS'],label='Drug')
    paths=retriever.retrieve_multi_hop('Warfarin',2,'Drug')
    assert paths and paths[0]['edges']
    assert any(row['interaction_type']=='direct' for row in retriever.retrieve_interaction('Warfarin','Aspirin'))
    assert any(row['interaction_type']=='shared_side_effects' for row in retriever.retrieve_interaction('Warfarin','Aspirin'))
    assert any(row['interaction_type']=='same_class' for row in retriever.retrieve_interaction('Aspirin','Ibuprofen'))
    resolver.extract_entities_from_question=lambda *a,**k:['Warfarin','Aspirin','Ibuprofen']
    docs=retriever.retrieve_for_question('question')
    assert docs and len(docs)<=15
    assert any('Warfarin' in doc.page_content and 'Aspirin' in doc.page_content for doc in docs)
    resolver.extract_entities_from_question=lambda *a,**k:['zzzz-not-found-zzzz']
    assert retriever.retrieve_for_question('missing')==[]


def test_live_upgraded_metadata_stops_being_reference_only(graph):
    client,dataset,_=graph
    client.write("MATCH (d:Drug {name:'Warfarin'}) SET d.metadata_status='reference_only'")
    seed_dataset(client,dataset)
    assert client.query("MATCH (d:Drug {name:'Warfarin'}) RETURN d.metadata_status AS status")[0]['status']=='complete'


def test_live_etl_notebook_runs_from_repository_root(graph,monkeypatch):
    from contextlib import redirect_stdout
    from io import StringIO
    client,_,_=graph
    monkeypatch.setenv('NEO4J_URI',URI)
    monkeypatch.setenv('NEO4J_PASSWORD','test-only')
    monkeypatch.delenv('OPENROUTER_API_KEY',raising=False)
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    monkeypatch.chdir(ROOT)
    notebook=json.loads((ROOT/'experiments/graphrag/notebooks/01_knowledge_graph_etl.ipynb').read_text())
    scope={}
    with redirect_stdout(StringIO()):
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':
                exec(compile(''.join(cell['source']),f'ETL cell {i}','exec'),scope)
    assert scope['counts']['interaction_pairs']==227
    assert client.query("MATCH (n:QAOnly {name:'preserved'}) RETURN count(n) AS count")[0]['count']==1


def test_live_crag_notebook_all_cells_with_stubbed_provider_calls(graph,monkeypatch):
    """Use real Neo4j, real LangGraph, and deterministic provider substitutes."""
    from contextlib import redirect_stdout
    from io import StringIO
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    from graphrag.agent import RouteDecision,GradeDecision
    from graphrag.retrieval import ExtractedQueryEntities
    import graphrag.llm_services as providers
    client,_,_=graph
    requested_models=[]
    class LLM(RunnableLambda):
        def __init__(self,**options):
            requested_models.append(options['model'])
            super().__init__(lambda prompt:AIMessage(content='Fixture answer grounded in graph context'))
        def with_structured_output(self,schema):
            def reply(prompt):
                question=prompt.to_messages()[-1].content.lower()
                if schema is RouteDecision:
                    return schema(destination='chitchat' if 'hello' in question else 'drug_interaction')
                if schema is GradeDecision: return schema(binary_score='yes')
                if schema is ExtractedQueryEntities:
                    entities=[]
                    for prefix,name in [('warfar','Warfarin'),('aspirin','Aspirin'),('ibuprofen','Ibuprofen'),
                                        ('nsaid','NSAID'),('metformin','Metformin'),('bleeding','Bleeding')]:
                        if prefix in question:entities.append(name)
                    return schema(entities=entities)
                raise AssertionError(f'Unexpected schema {schema}')
            return RunnableLambda(reply)
    class Embeddings:
        def embed_documents(self,texts):return [[1.,0.] for text in texts]
        def embed_query(self,text):return [1.,0.]
    monkeypatch.setenv('NEO4J_URI',URI)
    monkeypatch.setenv('NEO4J_PASSWORD','test-only')
    monkeypatch.setenv('OPENROUTER_API_KEY','stub-only')
    monkeypatch.setattr(providers,'ChatOpenAI',LLM)
    monkeypatch.setattr(providers,'OpenAIEmbeddings',lambda **kwargs:Embeddings())
    monkeypatch.chdir(ROOT)
    notebook=json.loads((ROOT/'experiments/graphrag/notebooks/02_crag_graphrag.ipynb').read_text())
    scope={}
    with redirect_stdout(StringIO()):
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':
                exec(compile(''.join(cell['source']),f'CRAG cell {i}','exec'),scope)
    assert 'openai/gpt-4o' in requested_models
    assert 'openai/gpt-4o-mini' in requested_models
    assert scope['app'] is not None
    assert client.get_relationship_count('INTERACTS_WITH')==227
