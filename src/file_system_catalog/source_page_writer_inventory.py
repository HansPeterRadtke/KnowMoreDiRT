"""Read-only observed additional native/source-map writers and timer gate.

This inventory explicitly distinguishes scheduled native direct embedding,
source-map extraction and ordinary corpus catalog refresh. The summary worker
boundary hold by itself is never sufficient for source-map + native + files
release; unknown manual writers and currently unpinned legacy readers remain.
"""
from __future__ import annotations
from datetime import datetime,timezone
from typing import Mapping

SCHEMA='kmd.private_327_pdf_additional_writer_runtime_inventory.v1'
WRITERS={
 'native_direct_embedding':('knowledge-library-semantic-worker.service',
                            'knowledge-library-semantic-worker.timer',
                            'native_filedb_chunks_and_vectors'),
 'semantic_source_extraction':('knowledge-library-semantic-extract.service',
                               'knowledge-library-semantic-extract.timer',
                               'source_map_and_derived_text_cache'),
}
CATALOG_ONLY=('knowledge-library-index.service','knowledge-library-index.timer')


def assess_other_writers(*,unit_states:Mapping[str,dict],
                         summary_pause:dict,checked_utc:str|None=None)->dict:
    """Strictly classify actual service/timer observations, not permissions."""
    expected={u for service,timer,_ in WRITERS.values() for u in (service,timer)}
    expected.update(CATALOG_ONLY)
    if not isinstance(unit_states,dict) or set(unit_states)!=expected:
        raise ValueError('every known native writer and adjacent catalog unit must be observed')
    if (not isinstance(summary_pause,dict)
         or summary_pause.get('schema')!='kmd.publisher_native_full_source_document_boundary_readiness.v1'
         or type(summary_pause.get('both_new_code_workers_confirmed')) is not bool
         or type(summary_pause.get('both_workers_quiesced_at_complete_original_boundary')) is not bool
         or summary_pause.get('safe_to_cutover_live') is not False):
        raise ValueError('missing independently observed resident KMD worker pause status')
    actual={};blockers=[]
    for label,(service,timer,target) in WRITERS.items():
        running=unit_states[service];timer_state=unit_states[timer]
        for name,row in ((service,running),(timer,timer_state)):
            if (not isinstance(row,dict) or set(row)!={'load_state','active_state','main_pid'}
                or row['load_state']!='loaded'
                or row['active_state'] not in ('active','inactive','failed')
                or type(row['main_pid']) is not int or row['main_pid']<0):
                raise ValueError(f'untrusted installed native writer/timer state: {name}')
        if timer_state['main_pid']!=0:
            raise ValueError('unexpected scheduled timer main process identity')
        running_now=running['active_state']!='inactive' or running['main_pid']>0
        scheduled=timer_state['active_state']!='inactive'
        actual[label]={
            'service':service,'service_active':running_now,
            'service_main_pid':running['main_pid'],
            'timer':timer,'timer_enabled_for_future_invocations':scheduled,
            'writes':target,'coordinated_admission_lease_present':False}
        if running_now:blockers.append(label+'_service_active_or_unknown')
        if scheduled:blockers.append(label+'_scheduled_timer_armed')
    # Catalog refresh changes generated README/catalog, not authoritative
    # source-map/native vector rows. Keep it visible but do not classify it
    # as a writer capable of repairing or mutating either SQLite index.
    catalog={}
    for name in CATALOG_ONLY:
        row=unit_states[name]
        if (not isinstance(row,dict) or set(row)!={'load_state','active_state','main_pid'}
                or row['load_state']!='loaded'
                or row['active_state'] not in ('active','inactive','failed')
                or type(row['main_pid']) is not int or row['main_pid']<0):
            raise ValueError('untrusted catalog-only refresh state')
        catalog[name]={'active':row['active_state']=='active',
                       'catalog_readme_only_no_native_or_source_map_sqlite_writes':True}
    if not summary_pause['both_new_code_workers_confirmed']:
        blockers.append('resident_summary_workers_not_running_source_boundary_hold_protocol')
    if not summary_pause['both_workers_quiesced_at_complete_original_boundary']:
        blockers.append('resident_summary_workers_not_quiesced_at_source_boundaries')
    # Not a source-lock / process census. Other manual writers may exist; even
    # zero currently running known services is NOT proof of all writer release.
    blockers.extend(('no_verified_cross_store_reader_writer_admission_lease',
                     'unregistered_manual_writers_not_excluded',
                     'legacy_readers_not_pinned_to_atomic_source_generation'))
    return {'schema':SCHEMA,'checked_utc':checked_utc or datetime.now(timezone.utc).isoformat(),
        'additional_native_source_map_writers':actual,
        'catalog_only_refresh':catalog,
        'summary_worker_readiness':{
           'both_new_code_workers_confirmed':summary_pause['both_new_code_workers_confirmed'],
           'both_workers_quiesced_at_complete_original_boundary':summary_pause['both_workers_quiesced_at_complete_original_boundary']},
        'explicit_release_blockers':blockers,
        'native_and_source_map_sqlite_use_distinct_journaling_and_writer_locks':True,
        'advisory_lock_file_existence_is_not_proof_of_no_active_lock_holder':True,
        'does_not_exclude_manual_or_unregistered_writers_or_any_legacy_reader':True,
        'all_services_processes_only_observed_no_signals_or_modifications':True,
        'safe_to_cutover_live':False}
