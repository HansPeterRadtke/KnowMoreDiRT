"""Independent writer-timer source inventory must never authorize native cutover."""
from __future__ import annotations
import unittest
from file_system_catalog.source_page_writer_inventory import (
    SCHEMA,WRITERS,CATALOG_ONLY,assess_other_writers,
)


class AdditionalNativeWriterInventoryTests(unittest.TestCase):
    def setUp(self):
        self.units={u:{'load_state':'loaded','active_state':'inactive','main_pid':0}
                    for role in WRITERS.values() for u in role[:2]}
        self.units.update({u:{'load_state':'loaded','active_state':'inactive','main_pid':0}
                     for u in CATALOG_ONLY})
        self.summary={'schema':'kmd.publisher_native_full_source_document_boundary_readiness.v1',
            'both_new_code_workers_confirmed':True,
            'both_workers_quiesced_at_complete_original_boundary':True,
            'safe_to_cutover_live':False}

    def check(self):
        return assess_other_writers(unit_states=self.units,summary_pause=self.summary)

    def test_even_all_idle_known_writers_are_not_a_cross_store_release(self):
        out=self.check()
        self.assertEqual(SCHEMA,out['schema'])
        self.assertFalse(out['safe_to_cutover_live'])
        self.assertTrue(out['does_not_exclude_manual_or_unregistered_writers_or_any_legacy_reader'])
        self.assertIn('no_verified_cross_store_reader_writer_admission_lease',out['explicit_release_blockers'])
        self.assertIn('legacy_readers_not_pinned_to_atomic_source_generation',out['explicit_release_blockers'])
        self.assertFalse(out['explicit_release_blockers'][0].startswith('native_direct_embedding_service_active'))

    def test_timer_armed_but_unit_idle_is_not_a_safe_quiescence(self):
        for writer,timer,_ in WRITERS.values():
            self.units[timer]['active_state']='active'
        out=self.check()
        self.assertFalse(out['safe_to_cutover_live'])
        for label in WRITERS:
            self.assertIn(label+'_scheduled_timer_armed',out['explicit_release_blockers'])
            self.assertFalse(out['additional_native_source_map_writers'][label]['service_active'])

    def test_direct_native_writer_running_exposes_correct_blocker(self):
        self.units[WRITERS['native_direct_embedding'][0]].update(active_state='active',main_pid=1234)
        out=self.check()
        self.assertTrue(out['additional_native_source_map_writers']['native_direct_embedding']['service_active'])
        self.assertIn('native_direct_embedding_service_active_or_unknown',out['explicit_release_blockers'])

    def test_source_map_extractor_failed_is_not_interpreted_as_idle(self):
        self.units[WRITERS['semantic_source_extraction'][0]]['active_state']='failed'
        self.assertIn('semantic_source_extraction_service_active_or_unknown',self.check()['explicit_release_blockers'])

    def test_missing_or_wrong_writer_units_fail_closed(self):
        for bad in ('bogus',):
            with self.assertRaisesRegex(ValueError,'every known'):
                assess_other_writers(unit_states={**self.units,bad:self.units[CATALOG_ONLY[0]]},
                    summary_pause=self.summary)
        missing=dict(self.units);missing.pop(WRITERS['native_direct_embedding'][1])
        with self.assertRaisesRegex(ValueError,'every known'):
            assess_other_writers(unit_states=missing,summary_pause=self.summary)

    def test_malformed_status_from_systemd_fails_closed(self):
        unit=WRITERS['semantic_source_extraction'][0]
        for bad in ({'active_state':'unknown','main_pid':0,'load_state':'loaded'},
                    {'active_state':'inactive','main_pid':0,'load_state':'not-found'},
                    {'active_state':'inactive','main_pid':'0','load_state':'loaded'},
                    {'active_state':'inactive','main_pid':0},
                    {'active_state':'inactive','main_pid':-1,'load_state':'loaded'}):
            with self.subTest(bad=bad):
                broken={key:dict(value) for key,value in self.units.items()}
                broken[unit]=bad
                with self.assertRaises(ValueError):
                    assess_other_writers(unit_states=broken,summary_pause=self.summary)

    def test_catalog_refresh_does_not_impersonate_source_map_or_native_writer(self):
        self.units[CATALOG_ONLY[0]]['active_state']='active'
        self.units[CATALOG_ONLY[0]]['main_pid']=77
        out=self.check()
        self.assertTrue(out['catalog_only_refresh'][CATALOG_ONLY[0]]['active'])
        self.assertTrue(out['catalog_only_refresh'][CATALOG_ONLY[0]][
            'catalog_readme_only_no_native_or_source_map_sqlite_writes'])
        self.assertFalse(any('catalog' in error for error in out['explicit_release_blockers']))

    def test_stale_or_missing_summary_worker_ack_not_accepted(self):
        self.summary['both_workers_quiesced_at_complete_original_boundary']=False
        self.assertIn('resident_summary_workers_not_quiesced_at_source_boundaries',
                      self.check()['explicit_release_blockers'])
        self.summary['both_new_code_workers_confirmed']=False
        self.assertIn('resident_summary_workers_not_running_source_boundary_hold_protocol',
                      self.check()['explicit_release_blockers'])
        self.summary['safe_to_cutover_live']=True
        with self.assertRaisesRegex(ValueError,'missing independently observed'):
            self.check()

    def test_timer_invalid_mainpid_rejected(self):
        timer=WRITERS['native_direct_embedding'][1]
        self.units[timer].update(active_state='active',main_pid=55)
        with self.assertRaisesRegex(ValueError,'unexpected scheduled timer'):
            self.check()

if __name__=='__main__':unittest.main()
