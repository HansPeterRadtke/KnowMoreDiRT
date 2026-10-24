"""A scoped, actual-measurement hypothesis check; no LLM or toy truth labels.

Uses only independently sourced 2012 observations already extracted and
identity-validated in prior V2 provenance. A 2011 speed-excess premise is a
conditional *hypothesis*, NOT a true fact or observational gold label.

Reports sensitivity to additional unspecified systematic error explicitly as
post-hoc hypothetical stress checks. Gaussian approximations are assumptions.
"""
import json,hashlib,math
from pathlib import Path
HERE=Path(__file__).parent
SOURCE=HERE/'real_neutrino_source_analysis_corrected_20261024.json'
TARGET=HERE/'opera_conditional_prediction_results_20261024.json'
if TARGET.exists(): raise RuntimeError('Prior result exists. Do not rerun an already recorded experiment.')
raw=SOURCE.read_bytes();data=json.loads(raw)
assert all(data['verification'].values())
assert data['measurements']['opera_2012']['measured_experiment']=='OPERA'
assert data['measurements']['icarus_2012']['measured_experiment']=='ICARUS'
initial=next(x for x in data['historical_cern_claims'] if x['claim_type']=='initial_early_arrival')
assert initial['date']=='2011-09-23'
assert '20 parts per million' in initial['trigger_text']
# Physically explicit proposed constant propagation-speed increase, not the
# historical OPERA paper's exact 2011 fit. Original 20 ppm is rounded.
c=299792458.0; L=730000.0; epsilon=20e-6
expected_ns=(L/c)*(epsilon/(1+epsilon))*1e9
observations=[]
for key in ('icarus_2012','opera_2012'):
    m=data['measurements'][key];stat=float(m['statistical_ns']);sys=float(m['systematic_ns']);value=float(m['value_ns'])
    sigma=math.hypot(stat,sys)
    z_h1=(expected_ns-value)/sigma
    z_h0=(0-value)/sigma
    sensitivity=[{'hypothetical_additional_systematic_ns':additional,
       'standardized_difference_from_constant_20ppm_hypothesis':round((expected_ns-value)/math.hypot(sigma,additional),4)}
       for additional in (0,10,30)]
    observations.append({'experiment':m['measured_experiment'],'author':m['paper_author'],'paper_date':m['paper_date'],'source_url':m['url'],
       'measurement_delta_ns':value,'statistical_ns':stat,'systematic_ns':sys,
       'combined_uncertainty_ns_assuming_independent_gaussian_components':round(sigma,5),
       'descriptive_95_percent_normal_approximation_ns':[round(value-1.96*sigma,4),round(value+1.96*sigma,4)],
       'standardized_difference_from_h1':round(z_h1,4),'standardized_difference_from_h0':round(z_h0,4),
       'stress_checks':sensitivity,'source_document_sha256':m['source_sha256']})
result={'source_analysis_sha256':hashlib.sha256(raw).hexdigest(),'historical_premise':{'source_url':initial['source_url'],
     'publisher':initial['publisher'],'asserted_at':initial['date'],'status_at_date':'tentative reported anomaly inviting independent scrutiny, not confirmed faster-than-light travel',
     'text_trigger':initial['trigger_text']},
  'conditional_speed_hypothesis':{'relative_speed_excess':epsilon,'path_length_m':L,'c_m_s':c,'assumption':'Same constant speed excess carries across the later beams and each laboratory','predicted_early_arrival_ns':round(expected_ns,5)},
  'independent_later_observations':observations,
  'inference':'2012 measurements near zero instead of about 49 nanoseconds challenge this specific constant-speed-excess explanation, conditional on each instrument calibration, beam comparability and error assumptions; original 2011 excess should remain a dated observation/anomaly rather than universal truth',
  'not_evidence_of_new_physical_discovery':True,
  'confounders':['20ppm and 730km are rounded historical figures: this does not reproduce exact 2011 estimated 60ns early arrival','Systematic components within and between experiments are not demonstrated independent or exactly Gaussian','New 2012 measurements used upgraded apparatus and beam protocols','Energy/particle/measurement settings may constrain physical extrapolation: neutrino versus antineutrino must not be conflated','Post-hoc additional-error sensitivity is a stress test, not estimated real error','As-of-2011 availability cannot include March, June, August or December 2012 evidence'],
  'scope_controls':['Dream/fiction or quotation of superluminal claim is not a positive propagation observation','2012 OPERA and ICARUS are distinct authors/experiments; no automatic name merging','February 2012 possible timing faults are not yet independently confirmed diagnoses as of that date']}
TARGET.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print('PREDICTED_EARLY_NS',round(expected_ns,5))
for m in observations: print('SOURCE',m['experiment'],'OBS_NS',m['measurement_delta_ns'],'ERR_NS',m['combined_uncertainty_ns_assuming_independent_gaussian_components'],'DISTANCE_IN_ASSUMED_SIGMA',m['standardized_difference_from_h1'],'SENSITIVITY',[(x['hypothetical_additional_systematic_ns'],x['standardized_difference_from_constant_20ppm_hypothesis']) for x in m['stress_checks']])
