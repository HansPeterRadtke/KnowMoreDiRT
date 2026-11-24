"""Exact native semantic candidates held for ONE read-only, short-lived query batch.

This is not an ANN index, persistent vector cache, source revision cutover, or
live installed Infra replacement. Reads the same SQL rows and preserves the
legacy per-file strict-winner order and exact cosine floating-point scores.
Rejects ANY observed native database change across loading and all queries.
Individual and final combined original/derivative SHA + native ID checks are
performed by the existing, unchanged Infra source-verification reader.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sqlite3
import stat
import threading
import time
from typing import Any

import numpy as np

from .content_pipeline import vector_from_blob
from .content_schema import CHUNK_TABLE_NAME, REPRESENTATION_TABLE_NAME
from .schema import TABLE_NAME

MAX_TOTAL_ROWS=100_000
MAX_SOURCE_ORIGINALS=30_000


@dataclass(frozen=True)
class _Candidate:
    metadata: dict[str,Any]
    vector: np.ndarray
    vector_norm: np.floating | float


class NativeOneBatchSnapshot:
    """Bounded, single-threaded, ephemeral current SQLite source candidate cache."""
    @staticmethod
    def _physical_ram_available_bytes()->int:
        """Linux Raspi MemAvailable; an absent/unknown value fails closed."""
        try:
            with open('/proc/meminfo','r',encoding='ascii') as fh:
                for line in fh:
                    if line.startswith('MemAvailable:'):
                        fields=line.split()
                        if len(fields)==3 and fields[2]=='kB':
                            return int(fields[1])*1024
        except (OSError,ValueError):
            pass
        raise ValueError('Linux available physical RAM cannot be verified for bounded vector snapshot')

    def __init__(self,db_path:Path):
        beginning=time.monotonic()
        # Cached native candidate vectors + SQLite page cache + the shared E5
        # model peaked near 762 MiB in the 2026-11-24 Raspi real acceptance.
        # Leave at least 2 GiB OS headroom for resident summarization workers.
        if self._physical_ram_available_bytes()<2*1024*1024*1024:
            raise ValueError('insufficient physical RAM headroom for opt-in native batch vector snapshot')
        db_path=Path(db_path)
        if (db_path.is_symlink() or not db_path.is_file()
                or not db_path.is_absolute()):
            raise ValueError('native candidate database must be exact existing regular file')
        self.db_path=db_path
        self.thread=threading.get_ident()
        self.connection=sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True,timeout=35)
        self.closed=False
        self.candidates=[]
        try:
            self.connection.execute('PRAGMA query_only=ON')
            self.connection.execute('PRAGMA busy_timeout=35000')
            self.initial_metadata=self._file_identity()
            self.version=self._data_version()
            self._load_candidate_rows()
            self.require_current()
            self.build_seconds=round(time.monotonic()-beginning,3)
        except Exception:
            self.close()
            raise

    def _file_identity(self):
        s=self.db_path.stat(follow_symlinks=False)
        if not stat.S_ISREG(s.st_mode):raise ValueError('native indexed database changed type')
        return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

    def _data_version(self):
        return self.connection.execute('PRAGMA data_version').fetchone()[0]

    def require_current(self):
        if self.closed or threading.get_ident()!=self.thread:
            raise ValueError('native source batch snapshot used after close or on another thread')
        # Fail closed if other agents consume the remaining physical RAM while
        # this one-shot batch is active. The original live services continue.
        if self._physical_ram_available_bytes()<512*1024*1024:
            raise ValueError('native original snapshot memory headroom exhausted during multi-query search')
        if (self._data_version()!=self.version
                or self._file_identity()!=self.initial_metadata):
            raise ValueError('native KMD FileDB changed since one-batch source vector snapshot; refuse stale retrieval')

    def _append(self,metadata,blob,dimension,dtype):
        if len(self.candidates)>=MAX_TOTAL_ROWS:
            raise ValueError('native source candidates exceed safe one-shot memory bound')
        vector=vector_from_blob(blob,int(dimension),dtype)
        if vector.shape!=(384,):
            raise ValueError('unexpected native source embedding dimension')
        self.candidates.append(_Candidate(metadata,vector,np.linalg.norm(vector)))

    def _load_candidate_rows(self):
        # Exactly the same filtered CTE as Infra _ScopedNativeConnection(mode=any).
        # Use SELECT cursors without an explicit long-lasting SQLite BEGIN;
        # data_version and file identity must stay constant across both scans.
        prefix=("WITH content_chunks AS (SELECT * FROM main.content_chunks "
                "WHERE collection_id NOT LIKE ?) ")
        source_filter=('%:summary',)
        count_files=set()
        for row in self.connection.execute(prefix+f"""
            SELECT c.chunk_id,c.file_id,c.filesystem_entry_id,f.relative_path_display,f.relative_path_b64,
            c.chunk_kind,c.chunk_index,c.start_char,c.end_char,c.embedding_dimension,c.embedding_dtype,c.embedding_blob
            FROM {CHUNK_TABLE_NAME} c JOIN {TABLE_NAME} f ON f.id=c.filesystem_entry_id
            WHERE c.embedding_blob IS NOT NULL""",source_filter):
            meta={'file_id':row[1], 'filesystem_entry_id':int(row[2]),
                  'relative_path_display':row[3], 'relative_path_b64':row[4],
                  'chunk_id':row[0], 'source_level':row[5],
                  'source_index':int(row[6]), 'source_start_char':int(row[7]),
                  'source_end_char':int(row[8]),'analysis_kind':'chunk',
                  'analysis_text':''}
            count_files.add(str(row[1]))
            self._append(meta,row[11],row[9],row[10])
        for row in self.connection.execute(prefix+f"""
            SELECT r.representation_id,r.representation_kind,r.representation_text,r.embedding_dimension,
            r.embedding_dtype,r.embedding_blob,c.chunk_id,c.file_id,c.filesystem_entry_id,f.relative_path_display,
            f.relative_path_b64,c.chunk_kind,c.chunk_index,c.start_char,c.end_char
            FROM {REPRESENTATION_TABLE_NAME} r JOIN {CHUNK_TABLE_NAME} c ON c.chunk_id=r.chunk_id
            JOIN {TABLE_NAME} f ON f.id=c.filesystem_entry_id WHERE r.analysis_status='complete'""",source_filter):
            meta={'file_id':row[7], 'filesystem_entry_id':int(row[8]),
                  'relative_path_display':row[9], 'relative_path_b64':row[10],
                  'chunk_id':row[6], 'representation_id':row[0],
                  'source_level':row[11], 'source_index':int(row[12]),
                  'source_start_char':int(row[13]),'source_end_char':int(row[14]),
                  'analysis_kind':row[1],'analysis_text':row[2]}
            count_files.add(str(row[7]))
            self._append(meta,row[5],row[3],row[4])
        if not 1<=len(count_files)<=MAX_SOURCE_ORIGINALS or not self.candidates:
            raise ValueError('invalid or oversized one-batch original vector coverage')
        self.complete_originals=len(count_files)

    def rank(self,adapter,query_vector:np.ndarray):
        """Return EXACT original native search rank across the cached current rows."""
        self.require_current()
        # Never ignore an original path or restricted source collection:
        # we only prepared mode='any', original_path=None. Legacy Infra
        # _ScopedNativeConnection validates this exact per-query SQL contract.
        if (not hasattr(adapter,'parameters') or adapter.parameters!=('%:summary',)
                or not isinstance(getattr(adapter,'prefix',None),str)
                or 'collection_id NOT LIKE ?' not in adapter.prefix):
            raise ValueError('pinned source snapshot cannot ignore native query path/mode scope')
        if (not isinstance(query_vector,np.ndarray) or query_vector.shape!=(384,)
                or query_vector.dtype.kind!='f'):
            raise ValueError('invalid current semantic query embedding')
        norm=np.linalg.norm(query_vector)
        best={}
        for candidate in self.candidates:
            denominator=float(norm*candidate.vector_norm)
            score=float(np.dot(query_vector,candidate.vector)/denominator) if denominator else 0.0
            key=str(candidate.metadata['file_id'])
            previous=best.get(key)
            if previous is not None and not (score>float(previous['score'])):
                continue
            best[key]={**candidate.metadata,'score':score}
        self.require_current()
        return sorted(best.values(),key=lambda item:(-float(item['score']),str(item['relative_path_display'])))

    def close(self):
        if not self.closed:
            self.closed=True
            self.connection.close()
            self.candidates.clear()

    def __enter__(self):return self
    def __exit__(self,typ,value,tb):self.close()
