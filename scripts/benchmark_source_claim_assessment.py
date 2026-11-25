#!/usr/bin/env python3
"""Private, bounded loopback-only source-claim benchmark; no live index writes.

The caller prepares independently attested private source spans and frozen
manual labels, and supplies an already authorized local model endpoint.
No source file is downloaded or sent off host. No model is started here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from file_system_catalog.source_claim_assessment import (
    PROMPTS, GROUPS, validate_dataset, model_input, validate_model_result, score_results,
    pairwise_model_input, validate_pairwise_result, aggregate_pairwise,
)

MAX_INPUT = 2 * 1024 * 1024
MAX_RESPONSE = 128 * 1024


def _utc():
    return datetime.now(timezone.utc).isoformat()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url,code,"redirect refused",headers,None)


class LocalModel:
    def __init__(self, port:int, model:str, *, timeout:float=120):
        if type(port) is not int or not 1024<=port<=65535:
            raise ValueError("explicit loopback model port required")
        if not isinstance(model,str) or not 1<=len(model)<=150 or any(ord(x)<32 for x in model):
            raise ValueError("explicit local model ID required")
        if not 5<=timeout<=180:
            raise ValueError("invalid bounded request timeout")
        self.base=f"http://127.0.0.1:{port}"
        self.model=model
        self.timeout=timeout
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())

    def _json(self, route, payload=None):
        if route not in ("/v1/models","/v1/chat/completions"):
            raise ValueError("unsupported local model route")
        body=None if payload is None else json.dumps(payload,ensure_ascii=False).encode()
        req=urllib.request.Request(self.base+route,data=body,
                   headers={"Content-Type":"application/json","Accept":"application/json"})
        with self.opener.open(req,timeout=self.timeout) as response:
            raw=response.read(MAX_RESPONSE+1)
        if len(raw)>MAX_RESPONSE:
            raise ValueError("local model response exceeds byte budget")
        return json.loads(raw)

    def verify_model(self):
        models=self._json("/v1/models")
        ids=[x.get("id") for x in models.get("data",[]) if isinstance(x,dict)]
        if ids != [self.model]:
            raise ValueError("loopback endpoint does not expose exactly the requested model ID")

    def judge(self,case,sources,variant,seed):
        if variant=="pairwise":
            return self.judge_pairwise(case,sources,seed)
        system,user,schema=model_input(case,sources,variant)
        payload={"model":self.model,"messages":[{"role":"system","content":system},
                 {"role":"user","content":user}],"temperature":0.0,"seed":seed,
                 "max_tokens":80,"stream":False,"enable_thinking":False,
                 "chat_template_kwargs":{"enable_thinking":False},
                 "response_format":{"type":"json_schema","json_schema":{
                     "name":"kmd_source_claim_candidate","strict":True,"schema":schema}}}
        prompt_sha=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        started=time.monotonic()
        response=self._json("/v1/chat/completions",payload)
        elapsed=round(time.monotonic()-started,3)
        receipt={"prompt_sha256":prompt_sha,"elapsed_seconds":elapsed,"result":None,
                 "usage":response.get("usage"),"finish_reason":None}
        try:
            choice=response["choices"][0]
            receipt["finish_reason"]=choice.get("finish_reason")
            message=choice["message"]
            if choice.get("finish_reason")!="stop" or message.get("reasoning_content"):
                raise ValueError("incomplete or unexpected reasoning output")
            content=message.get("content")
            receipt["result"]=validate_model_result(case,content)
        except (ValueError,TypeError,KeyError,IndexError) as exc:
            # Retain diagnostic only; no publisher evidence or untrusted raw
            # model response in the receipt. Invalid is not a true unknown.
            receipt["validation_error"]=type(exc).__name__+": "+str(exc)[:240]
        return receipt


    def judge_pairwise(self,case,sources,seed):
        started=time.monotonic()
        per_source=[];relations={}
        for source_id in case["evidence_ids"]:
            system,user,schema=pairwise_model_input(case,sources,source_id)
            payload={"model":self.model,"messages":[{"role":"system","content":system},
                     {"role":"user","content":user}],"temperature":0.0,"seed":seed,
                     "max_tokens":40,"stream":False,"enable_thinking":False,
                     "chat_template_kwargs":{"enable_thinking":False},
                     "response_format":{"type":"json_schema","json_schema":{
                         "name":"kmd_single_source_relation","strict":True,"schema":schema}}}
            sha=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            t=time.monotonic()
            response=self._json("/v1/chat/completions",payload)
            item={"evidence_id":source_id,"prompt_sha256":sha,
                  "elapsed_seconds":round(time.monotonic()-t,3),"relation":None,
                  "usage":response.get("usage") if isinstance(response,dict) else None,
                  "finish_reason":None}
            try:
                choice=response["choices"][0]
                item["finish_reason"]=choice.get("finish_reason")
                message=choice["message"]
                if choice.get("finish_reason")!="stop" or message.get("reasoning_content"):
                    raise ValueError("incomplete or unexpected reasoning output")
                item["relation"]=validate_pairwise_result(message.get("content"))
                relations[source_id]=item["relation"]
            except (ValueError,TypeError,KeyError,IndexError,AttributeError) as exc:
                item["validation_error"]=type(exc).__name__+": "+str(exc)[:180]
            per_source.append(item)
        result=None
        try:
            result=aggregate_pairwise(case,relations)
        except ValueError:
            pass
        return {"result":result,"elapsed_seconds":round(time.monotonic()-started,3),
                "source_local_model_calls":per_source,
                "aggregation":"complete_valid_source_local_relations_only"}


def _read_private(path:Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_INPUT:
        raise ValueError("private frozen dataset missing, unsafe or oversized")
    raw=path.read_bytes()
    if len(raw)>MAX_INPUT:
        raise ValueError("private frozen dataset exceeds byte budget")
    return raw


def _new_private(path:Path):
    if path.exists() or path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("benchmark output must be a new private path")
    return os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o600),
                     "w",encoding="utf8")


def run_benchmark(dataset,client,*,variants,seed,checkpoint,groups=None,deadline_seconds=None):
    validate_dataset(dataset)
    if (not isinstance(variants,list) or not variants or len(set(variants))!=len(variants)
            or not all(v in PROMPTS for v in variants)
            or type(seed) is not int or not 0<=seed<=2**31-1):
        raise ValueError("invalid benchmark prompt variants or seed")
    if groups is not None and (not groups or not set(groups)<=set(GROUPS)):
        raise ValueError("invalid benchmark case group")
    cases=[c for c in dataset["cases"] if groups is None or c["group"] in groups]
    if not cases:
        raise ValueError("no selected cases")
    rows={v:[] for v in variants}
    started=time.monotonic()
    interrupted=None
    for i,case in enumerate(cases):
        # Fixed alternating A/B order to reduce systematic ordering bias.
        ordered=variants if i%2==0 else list(reversed(variants))
        for variant in ordered:
            if deadline_seconds is not None and time.monotonic()-started>=deadline_seconds:
                interrupted="explicit resource-lease deadline reached before next case"
                break
            try:
                record=client.judge(case,dataset["sources"],variant,seed)
            except (urllib.error.URLError,OSError,TimeoutError,ValueError) as exc:
                # Transport refusal or endpoint error is not silently retried
                # and never turned into a semantic label.
                interrupted=f"local endpoint interruption: {type(exc).__name__}: {str(exc)[:180]}"
                break
            row={"id":case["id"],"variant":variant,**record}
            rows[variant].append(row)
            checkpoint.write(json.dumps(row,ensure_ascii=False)+"\n")
            checkpoint.flush();os.fsync(checkpoint.fileno())
            print("CASE",case["id"],variant,"valid" if row["result"] else "invalid",
                  row.get("elapsed_seconds"),flush=True)
        if interrupted:
            break
    # Partial comparisons use exactly the cases completed by every variant.
    matched=set.intersection(*(set(r["id"] for r in rows[v]) for v in variants))
    paired=[c for c in cases if c["id"] in matched]
    scores={v:score_results(paired,[r for r in rows[v] if r["id"] in matched]) for v in variants} if paired else {}
    return {"selected_cases":len(cases),"paired_cases_completed":len(paired),
            "complete":interrupted is None and len(paired)==len(cases),
            "interruption":interrupted,"elapsed_seconds":round(time.monotonic()-started,3),
            "completed_rows_per_variant":{k:len(v) for k,v in rows.items()},
            "scores_on_same_paired_cases":scores,"model_candidate_not_factual_verification":True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--port",required=True,type=int)
    ap.add_argument("--model",required=True)
    ap.add_argument("--variant",action="append",choices=sorted(PROMPTS),required=True)
    ap.add_argument("--group",action="append",choices=GROUPS)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--deadline-seconds",type=int,default=2400)
    args=ap.parse_args()
    if not 60<=args.deadline_seconds<=3600:
        raise ValueError("invalid bounded lease deadline")
    raw=_read_private(args.dataset)
    data=validate_dataset(json.loads(raw))
    output=args.output
    if output.exists() or output.is_symlink():
        raise ValueError("private final output already exists")
    checkpoint=output.with_name(output.name+".jsonl")
    client=LocalModel(args.port,args.model)
    client.verify_model()
    with _new_private(checkpoint) as log:
        log.write(json.dumps({"schema":"kmd.private_source_claim_benchmark_checkpoint.v1",
                   "dataset_sha256":hashlib.sha256(raw).hexdigest(),"model":args.model,
                   "started_utc":_utc(),"variants":args.variant})+"\n")
        log.flush();os.fsync(log.fileno())
        report=run_benchmark(data,client,variants=args.variant,seed=args.seed,
                    checkpoint=log,groups=args.group,deadline_seconds=args.deadline_seconds)
    report.update(schema="kmd.private_source_claim_benchmark.v1",completed_utc=_utc(),
                  dataset_sha256=hashlib.sha256(raw).hexdigest(),model=args.model,
                  prompt_sha256={v:hashlib.sha256(PROMPTS[v].encode()).hexdigest() for v in args.variant},
                  source_text_not_written_to_benchmark_receipt=True,
                  seed=args.seed,checkpoint=checkpoint.name)
    with _new_private(output) as f:
        json.dump(report,f,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    print("BENCHMARK",json.dumps(report,ensure_ascii=False),flush=True)
    raise SystemExit(0 if report["complete"] else 2)


if __name__=="__main__":
    main()
