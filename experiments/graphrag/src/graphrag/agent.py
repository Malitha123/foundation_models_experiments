"""Corrective RAG with fresh retrieval state and an explicit no-evidence exit."""
from typing import Annotated, Literal, Optional, TypedDict
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from graphrag.config import get_config
from graphrag.llm_services import get_general_llm, get_strong_llm
from graphrag import prompts

NO_CONTEXT_RESPONSE = 'The knowledge graph did not provide relevant evidence to answer this question. This does not establish whether the drugs are safe together or whether an interaction exists.'


class GraphRAGState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    original_question: str
    entities: list[str]
    # A new retrieval replaces the previous attempt's context.
    documents: list[str]
    route: str
    grade: str
    retry_count: int
    final_answer: Optional[str]


class RouteDecision(BaseModel):
    destination: Literal['drug_interaction', 'general_medical', 'chitchat']


class GradeDecision(BaseModel):
    binary_score: Literal['yes', 'no']
    reasoning: str = Field(default='')


def build_workflow(retriever, general_llm_factory=get_general_llm,
                   extraction_llm_factory=get_strong_llm, max_retries=None):
    max_retries = get_config().get('agent', {}).get('max_retries', 2) if max_retries is None else max_retries
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or not 0 <= max_retries <= 6:
        raise ValueError('max_retries must be an integer between 0 and 6')

    def invoke(system, human, values, schema=None):
        llm = general_llm_factory()
        if schema is not None:
            llm = llm.with_structured_output(schema)
        prompt = ChatPromptTemplate.from_messages([('system', system), ('human', human)])
        return (prompt | llm).invoke(values)

    def router(state):
        result = invoke(prompts.ROUTER_SYSTEM_PROMPT, '{question}', {'question': state['original_question']}, RouteDecision)
        return {'route': result.destination, 'retry_count': 0, 'documents': []}

    def retrieve(state):
        documents = retriever.retrieve_for_question(state['question'], llm=extraction_llm_factory())
        return {'documents': [doc.page_content for doc in documents]}

    def grade(state):
        if not state['documents']:
            return {'grade': 'not_relevant'}
        result = invoke(prompts.GRADER_SYSTEM_PROMPT, prompts.GRADER_HUMAN_PROMPT,
                        {'context': '\n\n'.join(state['documents']), 'question': state['original_question']}, GradeDecision)
        return {'grade': 'relevant' if result.binary_score == 'yes' else 'not_relevant'}

    def rewrite(state):
        result = invoke(prompts.REWRITER_SYSTEM_PROMPT, prompts.REWRITER_HUMAN_PROMPT,
                        {'question': state['original_question']})
        text = result.content.strip()
        return {'question': text or state['original_question'], 'retry_count': state['retry_count']+1, 'documents': []}

    def generate(state):
        result = invoke(prompts.GENERATOR_SYSTEM_PROMPT, prompts.GENERATOR_HUMAN_PROMPT,
                        {'context': '\n\n'.join(state['documents']), 'question': state['original_question']})
        return {'messages': [AIMessage(content=result.content)], 'final_answer': result.content}

    def insufficient_context(state):
        return {'messages': [AIMessage(content=NO_CONTEXT_RESPONSE)], 'final_answer': NO_CONTEXT_RESPONSE}

    def general_chat(state):
        if state['route'] == 'chitchat':
            answer = prompts.CHITCHAT_RESPONSE
        else:
            answer = invoke(prompts.GENERAL_MEDICAL_PROMPT, '{question}', {'question': state['original_question']}).content
        return {'messages': [AIMessage(content=answer)], 'final_answer': answer}

    workflow = StateGraph(GraphRAGState)
    for name, node in {'router':router, 'graph_retriever':retrieve, 'grader':grade, 'rewriter':rewrite,
                       'generator':generate, 'insufficient_context':insufficient_context, 'general_chat':general_chat}.items():
        workflow.add_node(name, node)
    workflow.set_entry_point('router')
    workflow.add_conditional_edges('router', lambda state: 'graph_retriever' if state['route']=='drug_interaction' else 'general_chat')
    workflow.add_edge('graph_retriever', 'grader')
    workflow.add_conditional_edges('grader', lambda state: 'generator' if state['grade']=='relevant' else
                                   ('insufficient_context' if state['retry_count']>=max_retries else 'rewriter'))
    workflow.add_edge('rewriter', 'graph_retriever')
    for name in ('generator', 'insufficient_context', 'general_chat'):
        workflow.add_edge(name, END)
    return workflow.compile()


def run_query(app, question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('question must be a nonempty string')
    question = question.strip()
    result = app.invoke({'messages':[], 'question':question, 'original_question':question, 'entities':[],
                         'documents':[], 'route':'', 'grade':'', 'retry_count':0, 'final_answer':None},
                        config={'recursion_limit':32})
    return result.get('final_answer') or NO_CONTEXT_RESPONSE
