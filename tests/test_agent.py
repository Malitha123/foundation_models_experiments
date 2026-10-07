import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from graphrag.agent import build_workflow, GraphRAGState, RouteDecision, GradeDecision, NO_CONTEXT_RESPONSE, run_query


class FakeLLM(RunnableLambda):
    def __init__(self, grades, events):
        self.grades,self.events=grades,events
        super().__init__(self.answer)
    def answer(self,prompt):
        self.events.append(prompt.to_messages())
        system=prompt.to_messages()[0].content
        return AIMessage(content='rewritten retrieval query' if 'query rewriter' in system else 'grounded answer')
    def with_structured_output(self,schema):
        if schema is RouteDecision:
            return RunnableLambda(lambda prompt: RouteDecision(destination='drug_interaction'))
        def grade(prompt):
            self.events.append(prompt.to_messages())
            return GradeDecision(binary_score=next(self.grades))
        return RunnableLambda(grade)


class Retriever:
    def __init__(self,results): self.results=iter(results); self.calls=[]
    def retrieve_for_question(self,question,llm):
        self.calls.append(question)
        return [Document(page_content=x) for x in next(self.results)]


def test_rewrite_discards_failed_context_and_grades_original_question():
    events=[]; llm=FakeLLM(iter(['no','yes']),events)
    retriever=Retriever([['old rejected evidence'],['fresh relevant evidence']])
    app=build_workflow(retriever,lambda:llm,lambda:llm,max_retries=2)
    result=app.invoke({'messages':[],'question':'original user question','original_question':'original user question',
                       'documents':[],'retry_count':0,'entities':[]})
    assert result['documents']==['fresh relevant evidence']
    assert result['retry_count']==1
    assert result['final_answer']=='grounded answer'
    # Final generation never sees the rejected context or the rewritten intent.
    assert 'old rejected evidence' not in events[-1][-1].content
    assert 'original user question' in events[-1][-1].content
    assert retriever.calls==['original user question','rewritten retrieval query']


def test_exhausted_empty_retrieval_does_not_generate_an_answer():
    events=[]; llm=FakeLLM(iter([]),events); retriever=Retriever([[],[],[]])
    app=build_workflow(retriever,lambda:llm,lambda:llm,max_retries=2)
    assert run_query(app,'missing entity')==NO_CONTEXT_RESPONSE
    assert len(retriever.calls)==3
    assert all('Drug Interaction Expert' not in event[0].content for event in events)


@pytest.mark.parametrize('value',[-1,1.5,True,7])
def test_invalid_retry_limits(value):
    with pytest.raises(ValueError): build_workflow(None,max_retries=value)


def test_invalid_question():
    with pytest.raises(ValueError): run_query(None,'   ')
