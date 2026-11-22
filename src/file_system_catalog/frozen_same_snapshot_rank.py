"""Read-only source-label discovery metrics from fixed native vector snapshots.

This is a rank-evaluation helper, not a factual-answer judge or deployed
search ranking. Lexical tokenization and title/summary weights remain frozen
from an earlier pre-holdout private experiment. No source downloads, model
training, live index writes, database creation or search behavior changes.
"""
from __future__ import annotations
import hashlib
import math
import re

import numpy as np

STOP=set(('a an the is are was were be been am and or if when how what why where who which do does did i we you your me my our their them it its he she his her for as of at by to into from with without on in inside outside between after before under over through using use using make need needs can would should could may perhaps might than then that this these those get gets become not no yes too many much more most few fewer some such same one two three four five data way work look find research about an another including include correct effective via while instead other people time point help answer asked question questions')).union(('there','here','using','based','example','examples','information','system','systems','method','methods'))

def frozen_lexical_terms(query:str)->list[str]:
    """Reproduce prior November 22 lexing EXACTLY, including its unsplit stopwords literal.

    The large literal was inadvertently not split into words in the original
    private experiment. A better stoplist requires a NEW independent benchmark,
    never a silent alteration to previously observed gold test results.
    """
    seen=set();out=[]
    for token in re.findall(r'[a-z0-9]+',query.lower()):
        if len(token)>2 and token not in STOP and token not in seen:
            seen.add(token);out.append(token)
    return out[:42]


def verified_native_vector(blob:bytes,digest:str,model:str,revision:str,
                           dimension:int,dtype:str,*,expected_model:str,
                           expected_revision:str)->np.ndarray|None:
    """Reject stale, corrupt or foreign native E5 source embeddings.

    Original source/revision and exact excerpt proof are separate caller
    responsibilities; an embedding digest alone never authorizes citations.
    """
    if (not isinstance(blob,bytes) or not isinstance(digest,str)
            or model!=expected_model or revision!=expected_revision
            or dimension!=384 or dtype not in ('float16','float32')
            or len(blob)!=(2 if dtype=='float16' else 4)*384
            or hashlib.sha256(blob).hexdigest()!=digest):
        return None
    matrix=np.frombuffer(blob,dtype='<f2' if dtype=='float16' else '<f4').astype('<f4')
    norm=float(np.linalg.norm(matrix))
    if not np.isfinite(matrix).all() or not math.isfinite(norm) or norm<=0:
        return None
    return matrix/norm


def rank_max_similarity(vectors:np.ndarray,docids:np.ndarray,
                        query:np.ndarray,doc_paths:list[str],*,top:int=30)->list[str]:
    """Rank the strongest current unit of each original, not fabricated claims.

    Query/model/passage vectors are 384-dimensional. A trusted caller must
    already have checked vector hash, model revision, source revisions and
    original document ID before passing rows here.
    """
    if (type(top) is not int or not 1<=top<=100
            or vectors.ndim!=2 or vectors.shape[1]!=384
            or not 0<vectors.shape[0]<=100_000 or vectors.shape[0]!=len(docids)):
        raise ValueError('invalid bounded original candidate matrix')
    if (query.shape!=(384,) or not np.isfinite(query).all() or
            not np.isfinite(vectors).all() or not 1<=len(doc_paths)<=20_000
            or len(doc_paths)!=len(set(doc_paths))
            or docids.ndim!=1 or not np.issubdtype(docids.dtype,np.integer)
            or np.any(docids<0) or np.any(docids>=len(doc_paths))):
        raise ValueError('untrusted current original document vector metadata')
    norm=np.linalg.norm(query)
    if not math.isfinite(float(norm)) or norm<=0:
        raise ValueError('non-normalized/zero native query embedding')
    score=vectors @ (query / norm)
    best=np.full(len(doc_paths),-np.inf,dtype='<f4')
    np.maximum.at(best,docids,score)
    ordered=sorted((i for i,v in enumerate(best) if math.isfinite(float(v))),
                   key=lambda i:(-float(best[i]),doc_paths[i]))
    return [doc_paths[i] for i in ordered[:top]]


def reciprocal_rank_fusion(channels:dict[str,list[str]],*,top:int=30,k:int=60)->list[str]:
    """Fixed equal channel RRF; never scores a missing source as evidence."""
    if (type(top) is not int or not 1<=top<=100 or type(k) is not int or k!=60
            or not isinstance(channels,dict) or not 1<=len(channels)<=3
            or not set(channels)<= {'original_passage','summary','lexical'}):
        raise ValueError('invalid frozen channel fusion contract')
    scores={}
    for channel,paths in channels.items():
        if not isinstance(paths,list) or len(paths)>100 or len(set(paths))!=len(paths):
            raise ValueError('duplicate or excessive original paths in channel '+channel)
        for rank,path in enumerate(paths,1):
            if not isinstance(path,str) or not path or path.startswith('/'):
                raise ValueError('invalid original path')
            scores[path]=scores.get(path,0.0)+1/(k+rank)
    return sorted(scores,key=lambda p:(-scores[p],p))[:top]


def label_coverage(known:list[str],channels:dict[str,list[str]],*,top:int=10)->dict:
    if (not known or len(set(known))!=len(known) or not 1<=top<=100
            or any(not isinstance(p,str) for p in known)):
        raise ValueError('ambiguous source-label gold')
    flags={key:sum(p in paths[:top] for p in known) for key,paths in channels.items()}
    union=set()
    for paths in channels.values():union.update(paths[:top])
    return {'known_source_count':len(known),**{k+'_hits':v for k,v in flags.items()},
            'either_channel_hits':sum(p in union for p in known),
            'all_known_sources_found_by_some_channel':all(p in union for p in known)}
