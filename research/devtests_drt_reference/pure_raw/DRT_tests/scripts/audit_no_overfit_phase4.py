#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOLVER_FILES=[ROOT/'dspg.py',ROOT/'contracts.py',ROOT/'extract.py',ROOT/'merge.py',ROOT/'dspg_store.py',ROOT/'run_staged_tests.py',ROOT/'config'/'dspg_system.yaml',ROOT/'scripts'/'dspg_ingest_folder.py',ROOT/'scripts'/'dspg_query.py'] + [p for p in (ROOT/'dspg_system').rglob('*') if p.is_file()]
BANNED_IMPORTS=['evaluation','generate_blackbox_corpora','run_blackbox_evaluation','blackbox_oracles','oracle.json','phase4_blackbox']
GENERIC_ALLOWED_PATTERNS=[r'PR-\\d+',r'BUG',r'ISSUE',r'SUP',r'TICKET',r'https?',r'\\.cpp',r'who_author',r'which_customer']
SCHEMA_ENUM_LITERALS={'reported','asserted','quoted','unknown','same','different','uncertain','open','closed','customer','person','artifact','content','people','context'}

def read(path:Path)->str:
    return path.read_text(encoding='utf-8',errors='ignore') if path.exists() else ''

def collect_generated_literals(generated_root:Path|None, oracle_root:Path|None)->dict[str,set[str]]:
    buckets={k:set() for k in ['names','companies','ids','urls','files','questions','answers','source_phrases','case_ids','family_ids']}
    roots=[p for p in [generated_root, oracle_root] if p and p.exists()]
    for root in roots:
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in {'.txt','.json','.jsonl','.md','.csv','.tsv'}:
                continue
            text=read(path)
            for m in re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',text): buckets['names'].add(m)
            for m in re.findall(r'\b(?:PR|BUG|SUP|TICKET|ISSUE)-\d+\b',text): buckets['ids'].add(m)
            for m in re.findall(r'https://[^\s"\']+',text): buckets['urls'].add(m.rstrip('.,'))
            for m in re.findall(r'\b[A-Za-z0-9_./-]+\.(?:cpp|tmp|py|js|md|txt|json|yaml|yml)\b',text): buckets['files'].add(m)
            if path.name=='questions.jsonl':
                for line in text.splitlines():
                    try:
                        obj=json.loads(line); q=obj.get('question')
                        if q: buckets['questions'].add(q)
                    except Exception: pass
            if path.name=='oracle.json':
                try:
                    obj=json.loads(text); buckets['case_ids'].add(obj.get('case_id','')); buckets['family_ids'].add(obj.get('component','')); buckets['family_ids'].add(obj.get('category',''))
                    for vals in (obj.get('answers') or {}).values():
                        for v in vals: buckets['answers'].add(str(v))
                except Exception: pass
            for sent in re.split(r'[\n.!?]+',text):
                phrase=' '.join(sent.split())
                if 24 <= len(phrase) <= 140:
                    buckets['source_phrases'].add(phrase)
    return buckets

def load_templates(path:Path|None)->list[str]:
    if not path or not path.exists(): return []
    try:
        obj=json.loads(read(path)); return [str(x) for x in obj.get('semantic_cue_templates',[])]
    except Exception: return []

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--generated-root')
    ap.add_argument('--oracle-root')
    ap.add_argument('--templates')
    ap.add_argument('--output-json',default=str(ROOT/'logs'/'NO_OVERFIT_PHASE4_AUDIT.json'))
    ap.add_argument('--output-md',default=str(ROOT/'logs'/'NO_OVERFIT_PHASE4_AUDIT.md'))
    args=ap.parse_args()
    generated=Path(args.generated_root) if args.generated_root else None
    oracle=Path(args.oracle_root) if args.oracle_root else None
    fixtures=collect_generated_literals(generated,oracle)
    templates=load_templates(Path(args.templates) if args.templates else None)
    findings=[]; justified=[]
    solver_texts={str(p.relative_to(ROOT)):read(p) for p in SOLVER_FILES if p.exists() and p.is_file()}
    for rel,text in solver_texts.items():
        for token in BANNED_IMPORTS:
            if re.search(rf'(^|\n)\s*(from|import)\s+{re.escape(token)}\b', text):
                findings.append({'path':rel,'type':'banned_solver_import','token':token})
        for bucket,vals in fixtures.items():
            for val in vals:
                if not val or len(str(val)) < 6: continue
                if str(val).lower() in SCHEMA_ENUM_LITERALS:
                    continue
                if bucket in {'family_ids'} and val in {'people','customer','artifact','content','frame','scope','identity','query','mention_type','unknown'}:
                    continue
                if str(val) in text:
                    findings.append({'path':rel,'type':f'exact_generated_{bucket}','literal':str(val)[:160]})
    # Stronger cue-template check: full instantiated-like source phrases are disallowed;
    # short generic domain cues are reported but justified only when they are schema/prompt examples.
    for rel,text in solver_texts.items():
        for templ in templates:
            cleaned=re.sub(r'\{[^}]+\}','',templ).strip()
            cleaned=' '.join(cleaned.split())
            if len(cleaned)>=24 and cleaned in text:
                findings.append({'path':rel,'type':'generator_template_overlap','literal':cleaned})
            else:
                for cue in re.findall(r'[A-Za-z]+(?:\s+[A-Za-z]+){1,3}', cleaned):
                    if len(cue)>=14 and cue.lower() in text.lower():
                        justified.append({'path':rel,'type':'generic_cue_overlap_reviewed','literal':cue,'justification':'reported for review; not an exact generated template or answer literal'})
    passed=not findings
    payload={'passed':passed,'disallowed_findings':findings,'justified_findings':justified[:200],'fixture_counts':{k:len(v) for k,v in fixtures.items()},'solver_files':sorted(solver_texts)}
    Path(args.output_json).write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Phase 4 No-Overfit Audit','',f'- passed: `{passed}`',f'- disallowed_findings: `{len(findings)}`',f'- justified_findings: `{len(justified)}`','','## Disallowed']
    lines += [f"- `{f['path']}` {f['type']} `{f.get('literal') or f.get('token')}`" for f in findings] or ['- none']
    lines += ['','## Justified Cue Overlaps']
    lines += [f"- `{f['path']}` {f['type']} `{f['literal']}`" for f in justified[:50]] or ['- none']
    Path(args.output_md).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'passed':passed,'disallowed_findings':len(findings),'output':args.output_json},indent=2))
    return 0 if passed else 1
if __name__=='__main__': raise SystemExit(main())
