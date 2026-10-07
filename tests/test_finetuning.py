from contextlib import nullcontext
import json
from pathlib import Path
import types
import numpy as np
import pytest
from experiments.qlora_finetuning.src.helpers import (
    clean_records, split_records, map_dataset_fields, render_chat, has_training_completion,
    decode_new_tokens, parse_judge_response, hub_publish_target,
)
ROOT = Path(__file__).resolve().parents[1]


def test_blank_rows_and_duplicates_do_not_leak():
    rows = [{'instruction':' ', 'question':' actual ', 'output':' answer ', 'input':' '},
            {'instruction':'actual', 'output':'answer'}, {'instruction':'missing', 'output':' '},
            {'question':'next', 'answer':'another'}, {'question':'actual', 'answer':'second answer'}]
    cleaned, counts = clean_records(rows)
    assert counts == {'invalid':1, 'duplicates':1}
    train, val = split_records(cleaned, .5)
    assert {r['instruction'] for r in train}.isdisjoint(r['instruction'] for r in val)
    assert len(train)+len(val)==3
    assert split_records(cleaned, .5)==(train,val)


@pytest.mark.parametrize('rows,ratio', [([], .9), ([{'instruction':'one','input':'','output':'ok'}],.9),([],0),([],1)])
def test_invalid_splits(rows, ratio):
    with pytest.raises(ValueError): split_records(rows,ratio)


class CharTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text=''.join(m['role']+':'+m['content']+';' for m in messages)
        return text+'assistant:' if add_generation_prompt else text
    def __call__(self, text, **kwargs): return {'input_ids':list(text.encode())}
    def decode(self, ids, **kwargs): return ''.join(map(chr,ids))


def test_prompt_only_truncation_is_rejected():
    t=CharTokenizer(); row={'instruction':'a'*100,'input':'','output':'answer'}
    assert not has_training_completion(row,t,10)
    assert has_training_completion(row,t,200)


def test_completion_is_sliced_by_tokens():
    tokenizer=CharTokenizer()
    assert decode_new_tokens(np.array([[1,2,65,66]]),np.array([[1,2]]),tokenizer)=='AB'


@pytest.mark.parametrize('text,label,expected',[
    ('SCORE: 4\nREASONING: first\nsecond','SCORE',4),(' safety: 5\nReasoning: fine','SAFETY',5),
    ('SCORE: 0','SCORE',None),('SCORE: 99','SCORE',None),('SCORE: 4.5','SCORE',None),
    ('SCORE: 4/5','SCORE',None),(None,'SCORE',None),('blocked','SCORE',None),
])
def test_judge_scores(text,label,expected):
    score,reason=parse_judge_response(text,label)
    assert score==expected
    assert reason


def test_multiline_judge_reasoning():
    assert parse_judge_response('SCORE: 3\nREASONING: one\ntwo')[1]=='one\ntwo'


def test_publishing_is_opt_in_even_with_repo_env():
    assert hub_publish_target({'push_to_hub':False},'user/model') is None
    assert hub_publish_target({'push_to_hub':True},'user/model')=='user/model'
    with pytest.raises(ValueError): hub_publish_target({'push_to_hub':True},'')


def test_rouge_only_evaluation_keeps_low_quality_cases(tmp_path):
    """Execute the actual evaluation cell, with no GPU or paid judge."""
    nb=json.loads((ROOT/'experiments/qlora_finetuning/qlora_medical_finetuning.ipynb').read_text())
    source=''.join(nb['cells'][30]['source'])
    # Only the provider/scorer setup is replaced; run the actual loop/report/save.
    source=source[source.index('def evaluate_with_llm_judge'):]
    class Dataset(list):
        def select(self,indices): return Dataset(self[i] for i in indices)
    class Inputs(dict):
        def to(self,device): return self
    class Tokenizer(CharTokenizer):
        pad_token_id=0; eos_token_id=0
        def __call__(self,*a,**kw): return Inputs(input_ids=np.array([[1,2]]))
    class Model:
        device='cpu'
        def generate(self,**kw): return np.array([[1,2,65]])
        def disable_adapter(self): return nullcontext()
    scorer=types.SimpleNamespace(score=lambda *a:{'rougeL':types.SimpleNamespace(fmeasure=.1)})
    scope={'val_dataset':Dataset([{'instruction':'question','input':'context','output':'answer'}]),
           'CONFIG':{'max_new_tokens':1,'max_length':512,'temperature':0.,'do_sample':False,
                     'dataset_name':'fixture','base_model':'fixture','output_dir':'fixture'},
           'PROJECT_ROOT':tmp_path,'SEED':42,'use_llm_judge':False,'rouge_scorer_obj':scorer,
           'tokenizer':Tokenizer(),'finetuned_model':Model(),'chatml_format':lambda x:x,
           'torch':types.SimpleNamespace(no_grad=nullcontext),'decode_new_tokens':decode_new_tokens,
           'parse_judge_response':parse_judge_response}
    exec(compile(source,'evaluation cell','exec'),scope)
    assert len(scope['low_quality_cases'])==1
    saved=json.loads((tmp_path/'outputs/evaluation.json').read_text())
    assert saved['judge_model'] is None
    assert saved['examples'][0]['instruction']=='question\n\nInput: context'
    assert saved['examples'][0]['baseline_prediction']=='A'
