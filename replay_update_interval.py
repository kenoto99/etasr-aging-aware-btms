"""Reproduce a 300-s vs 600-s paired outer-interval uncertainty check.

Uses exactly the 30 pre-sampled parameter sets from outputs_ext/monte_carlo_uncertainty.csv.
The original 600-s outputs are retained unchanged; 300-s results are newly recomputed.
Run from the repository root: python replay_update_interval.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from BTMS_AgingAware_Predictive_Extension import simulate

ROOT=Path(__file__).resolve().parent
STORED=ROOT/'outputs_ext'/'monte_carlo_uncertainty.csv'
OUT=ROOT/'interval_replay';OUT.mkdir(exist_ok=True)
SAMPLED=['age_index','ambient_shift_C','fast_charge_scale','load_scale','res_mult','rsa_mult','cth_mult']
source=pd.read_csv(STORED)
rows=[]
for sid, grp in source.groupby('scenario_id',sort=True):
    r=grp.iloc[0]
    scenario={k:float(r[k]) for k in SAMPLED}
    scenario.update({'horizon_steps':6,'n_candidates':4})
    for controller in ['P-control','Aging-Robust']:
        old=grp.loc[grp.controller==controller].iloc[0]
        new=simulate(controller,days=1,dt_s=300.0,scenario=scenario,record=False)
        rows.append({'scenario_id':int(sid),'controller':controller,
          'max_core_600':float(old.max_core_C),'max_core_300':new['max_core_C'],
          'energy_600':float(old.btms_energy_kWh),'energy_300':new['btms_energy_kWh'],
          'exposure_600':float(old.degradation_exposure_au),'exposure_300':new['degradation_exposure_au']})
comparison=pd.DataFrame(rows)
comparison.to_csv(OUT/'mc_300_600_comparison.csv',index=False)
summ=[]
for sec in [300,600]:
    for controller,g in comparison.groupby('controller'):
        peaks=g[f'max_core_{sec}'].to_numpy()
        summ.append({'outer_interval_s':sec,'controller':controller,'n':len(peaks),
           'mean_peak_C':float(np.mean(peaks)),'p95_peak_C':float(np.quantile(peaks,.95)),
           'n_above_45':int(np.sum(peaks>45)),'fraction_above_45':float(np.mean(peaks>45)),
           'mean_btms_kWh_per_day':float(g[f'energy_{sec}'].mean()),
           'mean_exposure_au':float(g[f'exposure_{sec}'].mean())})
summary=pd.DataFrame(summ)
summary.to_csv(OUT/'mc_300_600_summary.csv',index=False)
print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
