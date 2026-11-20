"""Verify primary ETASR #22085 results against the accompanying CSV files.

Run from this repository directory: python verify_results.py
For a faster check of the four nominal cases: python verify_results.py --nominal-only
Execution speed and runtime_s are machine-dependent and are not compared.
"""
import argparse
import numpy as np
import pandas as pd
from BTMS_AgingAware_Predictive_Extension import CONTROLLERS, simulate, monte_carlo


def check_close(name, got, expected, tol=1e-9):
    err = float(np.max(np.abs(np.asarray(got, dtype=float) - np.asarray(expected, dtype=float))))
    if err > tol:
        raise AssertionError(f"{name}: max absolute error {err:.3g} > {tol:.3g}")
    print(f"PASS {name}: max absolute error = {err:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nominal-only', action='store_true')
    args = ap.parse_args()
    base = pd.read_csv('outputs_ext/controller_summary.csv')
    nominal_keys = ('max_core_C', 'max_surface_C', 'max_surface_spread_C',
                    'time_core_over_40_h', 'time_core_over_45_h',
                    'btms_energy_kWh', 'degradation_exposure_au')
    for controller in CONTROLLERS:
        fresh = simulate(controller, days=3, record=False)
        ref = base.loc[base.controller == controller].iloc[0]
        check_close(f'nominal: {controller}',
                    [fresh[k] for k in nominal_keys],
                    [ref[k] for k in nominal_keys])
    if not args.nominal_only:
        expected = pd.read_csv('outputs_ext/monte_carlo_uncertainty.csv')
        actual = monte_carlo(30)
        assert len(actual) == len(expected) == 60, 'Scenario count mismatch'
        assert (actual.controller.values == expected.controller.values).all(), 'Pair ordering mismatch'
        columns = ('max_core_C', 'max_surface_C', 'btms_energy_kWh',
                   'degradation_exposure_au', 'ambient_shift_C',
                   'fast_charge_scale', 'load_scale', 'res_mult',
                   'rsa_mult', 'cth_mult', 'age_index')
        for k in columns:
            check_close('paired uncertainty: ' + k, actual[k].values, expected[k].values)
        print('PASS all 30 paired uncertainty scenarios (60 controller runs)')
    print('Verification completed.')


if __name__ == '__main__':
    main()
