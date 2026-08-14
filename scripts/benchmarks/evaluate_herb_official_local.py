#!/usr/bin/env python3
"""Salesforce HERB evaluate.py semantics with only get_gpt4_response replaced by localhost."""
from __future__ import annotations
import json, re, string
from collections import defaultdict
from pathlib import Path
from typing import Any
from herb_kgqa.config import Settings
from knowmoredirt.model import LocalModelClient


class KMDJudgeLLM:
    """HERB evaluator adapter backed by KMD's global full-context model client."""

    def __init__(self, settings: Settings):
        endpoint = str(settings.llm_base_url).rstrip("/")
        if not endpoint.endswith("/v1"):
            endpoint += "/v1"
        self.client = LocalModelClient(endpoint=endpoint)
        self.request_count = 0

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        run_dir: Path | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        retries: int = 2,
    ) -> dict[str, Any]:
        del temperature, max_tokens, retries
        self.request_count += 1
        prompt = "\n\n".join(
            f"[{str(message.get('role') or 'user').upper()}]\n{str(message.get('content') or '')}"
            for message in messages
        )
        result = self.client.complete_json(
            prompt,
            json_schema={"type": "object", "additionalProperties": True},
        )
        clean = {key: value for key, value in result.items() if not str(key).startswith("_model_")}
        if run_dir is not None:
            log_dir = run_dir / "llm_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{self.request_count:05d}-kmd-judge.json").write_text(
                json.dumps({"messages": messages, "response": clean}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return clean

def _read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as h: return [json.loads(x) for x in h if x.strip()]

def _local(llm, run_dir, messages):
    return llm.chat_json(messages, run_dir=run_dir, temperature=0.0, retries=2)

def unanswerability_eval(question, answer, llm, run_dir):
    prompt = '''Given a question and its corresponding answer, determine whether the answer provides sufficient information to fully or even partially respond to the question.

- Reply "Yes" if the answer directly addresses the question, either fully or partially, using information relevant to the question.
- Reply "No" if the answer does not address the question due to missing, vague, or insufficient information — even if it explicitly states that the information is not available.

Question: Which products were affected by the bug in March?

Answer: The context does not provide the specific information required to determine the products.

Output: No

Question: Find all authors and reviewers of the PRD?

Answer: John authored the PRD

Output: Yes

Question: Find all authors and reviewers of the PRD?
Answer: John authored the PRD. Yuan and Jessica reviewed the PRD.

Output: Yes

'''
    prompt += f'\n\nQuestion: {question}\n\nAnswer: {answer}\n\nRespond using the below json format:\n{{"done": "yes/no", "reason": "justification for the decision"}}'
    verification = _local(llm, run_dir, [{'role':'user','content':prompt}])
    return str(verification['done']).lower() != 'no'

def _extract(question, content, llm, run_dir, kind):
    if kind == 'pr':
        prompt=("The following is a question and its answer. The answer may include reasoning and a final answer. Extract only the pull request links that directly answer the question—that is, the links that refer to the specific PRs being asked about."
        "\n#Instructions:\n" "- Ignore PR links mentioned in reasoning or intermediate steps." "- If the question asks which PRs were reverted, extract the original PRs that were reverted, not the PRs that reverted them." "- If no pull request links directly answer the question, {{'links': []}}." "\n\nQuestion: {question}" "\n\nAnswer: {content}" "\n\nRespond in json format: {{'links': ['list of links']}}").format(question=question,content=content); key='links'
    elif kind == 'url':
        prompt=("The following is a question and its answer. The answer may include reasoning and a final answer. Extract only the URLs that directly answer the question—ignore any links mentioned in the reasoning or intermediate steps. If no URLs are part of the final answer, return {{'links': []}}." "\n\nQuestion: {question}" "\n\nAnswer: {content}" "\n\nRespond in json format: {{'links': ['list of links']}}").format(question=question,content=content); key='links'
    elif kind == 'company':
        prompt=("The following is a question and its answer. The answer may include reasoning and a final answer. Extract only the company names that directly answer the question—ignore any names mentioned in the reasoning or intermediate steps. If no company names are part of the final answer, return {{'names': []}}." "\n\nQuestion: {question}" "\n\nAnswer: {content}" "\n\nRespond in JSON format: {{'names': ['list of company names']}}").format(question=question,content=content); key='names'
    elif kind == 'person':
        prompt=("The following is a question and its answer. The answer may include reasoning and a final answer. Extract only the employee IDs that directly answer the question—ignore any IDs mentioned in the reasoning or intermediate steps. If no employee IDs are part of the final answer, return {{'ids': []}}." "\n\nQuestion: {question}" "\n\nAnswer: {content}" "\n\nRespond in json format: {{'ids': ['list of employee IDs']}}").format(question=question,content=content); key='ids'
    else: raise ValueError(kind)
    result=_local(llm,run_dir,[{'role':'user','content':prompt}]); return list(result.get(key,[])) if result else []

def answer_likert_score(question, reference, candidate, llm, run_dir):
    prompt=("You are an expert evaluator. Given a question, a reference answer, " "and a candidate answer, your task is to evaluate how well the candidate " "answer aligns with the reference.\n\nFor your evaluation:" "\nFocus on accuracy, completeness, and relevance." "\nIf the candidate includes extra information, check if it's correct and appropriate." "\nIf it omits key points from the reference, mention that." "\nQuestion: {question}\nReference Answer: {reference}\nCandidate Answer: {candidate}" "\nRespond in json format: {{'score': 'Your overall rating between 0-100', " "'reason': 'a brief justification'}}'.").format(question=question,reference=reference,candidate=candidate)
    return int(_local(llm,run_dir,[{'role':'user','content':prompt}])['score'])

def normalize(answer):
    text=str(answer).lower(); text=''.join(ch for ch in text if ch not in set(string.punctuation)); text=re.sub(r'\b(a|an|the)\b',' ',text); return ' '.join(text.split())

def f1_score_sets(y_true,y_pred):
    a={normalize(x) for x in y_true}; b={normalize(x) for x in y_pred}; tp=len(a&b); fp=len(b-a); fn=len(a-b); p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.; return 2*p*r/(p+r) if p+r else 0.

def evaluate_run_official_local(run_dir: Path, *, settings: Settings) -> dict[str, Any]:
    questions={str(x['question_id']):x for x in _read_jsonl(settings.normalized_root/'questions.jsonl')}; gold={str(x['question_id']):x for x in _read_jsonl(settings.normalized_root/'gold.jsonl')}; predictions={str(x['question_id']):x for x in _read_jsonl(run_dir/'predictions.jsonl')}
    if set(questions)!=set(gold) or set(questions)!=set(predictions): raise ValueError('official HERB scoring requires exactly one frozen prediction and gold row for every question')
    llm=KMDJudgeLLM(settings); ans=[]; bytype=defaultdict(list); unans=[]; details=[]; ac=uc=0
    for qid,q in questions.items():
        g=gold[qid]; raw=predictions[qid].get('answer',''); answer=', '.join(map(str,raw)) if isinstance(raw,list) else str(raw or ''); answer=answer.split('</think>')[-1].strip() if '</think>' in answer else answer; qtype=str(g.get('question_type') or q.get('question_type') or 'content'); ref=g.get('gold_answer',[])
        if bool(g.get('answerable',False)):
            ac+=1; candidate=answer or "I don't know."
            if qtype=='content': score=float(answer_likert_score(q['question'],ref,candidate,llm,run_dir)); bucket='content'
            elif qtype in {'person','company','url','pr'}: score=f1_score_sets([str(x) for x in ref], [str(x) for x in _extract(q['question'],candidate,llm,run_dir,qtype)]); bucket='url' if qtype=='pr' else qtype
            else: raise ValueError(f'unsupported official HERB type {qtype!r}')
            ans.append(score); bytype[bucket].append(score); details.append({'question_id':qid,'answerable':True,'type':qtype,'score':score})
        else:
            uc+=1
            if not answer.strip(): s=1; rendered='unanswered'
            else: s=int(not unanswerability_eval(q['question'],answer,llm,run_dir)); rendered=answer
            unans.append({'q':q['question'],'a':rendered,'s':s}); details.append({'question_id':qid,'answerable':False,'type':qtype,'score':s})
    if (ac,uc)!=(815,699): raise ValueError(f'official HERB split must be 815/699, got {ac}/{uc}')
    scores={'evaluation_protocol':'SalesforceAIResearch/HERB code/evaluate.py semantics','judge_substitution':'get_gpt4_response -> localhost KMD LocalModelClient only','judge_base_url':settings.llm_base_url,'judge_model':settings.llm_model,'answerable_count':ac,'unanswerable_count':uc,'question_count':ac+uc,'gpt4-correctness-avg':sum(ans)/len(ans),'answerable_type_averages':{k:sum(v)/len(v) if v else 0 for k,v in bytype.items()},'answerable_type_counts':{k:len(v) for k,v in bytype.items()},'unanswerable_accuracy':sum(x['s'] for x in unans)/len(unans),'unanswerable_correct':sum(x['s'] for x in unans),'local_judge_request_count':llm.request_count}
    (run_dir/'scores.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2,sort_keys=True)+'\n'); (run_dir/'official_unanswerable_results.json').write_text(json.dumps(unans,ensure_ascii=False,indent=2)+'\n');
    with (run_dir/'official_question_details.jsonl').open('w') as h:
        for row in details: h.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    return scores
