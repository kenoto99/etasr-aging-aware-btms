# Aging-Aware Predictive Thermal Management of Lithium-Ion Battery Packs under Tropical Fast Charging and Uncertainty: Simulation Code and Results

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24365113.svg)](https://doi.org/10.5281/zenodo.24365113)

Praphat Phuangchuen, Prakasit Prabpal, Somchat Sonasang, Kritphon Phanrattanachai, and Wirasak Chotikankun (corresponding author, wirasak@techsakon.ac.th)

This repository contains the Python code, simulation outputs, and figure scripts used in the article "Aging-Aware Predictive Thermal Management of Lithium-Ion Battery Packs under Tropical Fast Charging and Uncertainty", submitted to *Engineering, Technology & Applied Science Research* (ETASR), manuscript no. 22085.

## What the code does

The script `BTMS_AgingAware_Predictive_Extension.py` simulates a reduced-order eight-cell lithium-ion pack with a 2RC electrical model, a two-node core/surface thermal network per cell, inter-cell thermal coupling, age-dependent capacity and resistance, and a temperature-accelerated degradation-exposure index. It compares four cooling strategies:

1. Passive operation (no active cooling)
2. Proportional control (P-control)
3. Finite-horizon predictive control
4. Aging-aware robust predictive control

Numerical scheme:

- Outer control interval: 120 s in the nominal case.
- The 2RC states use the exact exponential solution for a constant current over each interval.
- The thermal network uses explicit Euler sub-steps of no more than 10 s.
- Predictive strategies test five candidate cooling commands: 0, 53.3, 106.7, and 160 W, plus the previous command.
- Each candidate is evaluated over six 60-s prediction steps, which gives a 6-min horizon.
- Cooling-command changes are rate-limited to 35 W per minute.
- The Monte Carlo scenarios use the fixed random seed 20261031.

This is a model-based screening study. The parameters are a generic reference set, not values identified from a specific commercial cell. The long-term retained-capacity curves are comparative screening indices, not experimentally validated state-of-health predictions.

## License

See the `LICENSE` file in this repository.

## Requirements

- Python 3 (tested with Python 3.13)
- NumPy, pandas, Matplotlib (see `requirements.txt`)

## How to run

```
pip install -r requirements.txt
python BTMS_AgingAware_Predictive_Extension.py
```

The full run takes a few minutes on a standard laptop. All outputs are written to `outputs_ext/`, and the figures are written to `outputs_ext/figures/`.

## Simulation run sets

| Run set | Duration; outer interval | Output file |
|---|---|---|
| Nominal comparison, four strategies | 72 h; 120 s | `controller_summary.csv` |
| Battery age (a = 1.0, 0.9, 0.8) | 24 h; 300 s | `sensitivity_age.csv` |
| Ambient shift (-4, 0, +4 °C) | 24 h; 300 s | `sensitivity_ambient.csv` |
| Fast-charge rate (about 0.8C, 1.2C, 1.6C) | 24 h; 300 s | `sensitivity_fastcharge.csv` |
| Heat wave (+3.5 °C on day 2) | 72 h; 300 s | `sensitivity_heatwave.csv` |
| Paired uncertainty, 30 scenarios | 24 h; 600 s | `monte_carlo_uncertainty.csv`, `monte_carlo_summary.csv` |
| Long-term screening, 10 years | derived from nominal runs | `long_term_screening.csv` |
| Numerical checks (outer interval 60 to 300 s; horizon 3 to 12 min) | 72 h | `numerical_checks.csv` |

All sensitivity, heat-wave, and uncertainty runs use the same 6-min prediction horizon as the nominal case.

## Correspondence with the article

| Article item | Repository file |
|---|---|
| Table I (model parameters) | `Params` class in the script |
| Fig. 1 | `figures/Fig01_boundary_conditions.png` |
| Fig. 2 (simulation procedure) | flowchart drawn from the procedure in the script |
| Fig. 3 | `figures/Fig04_core_temperature_controllers.png` |
| Fig. 4 | `figures/Fig19_pareto_energy_exposure.png` |
| Fig. 5 | `figures/Fig22_age_state_temperature.png` |
| Fig. 6 | `figures/Fig24_ambient_sensitivity_temperature.png` |
| Fig. 7 | `figures/Fig25_fastcharge_temperature.png` |
| Fig. 8 | `figures/Fig29_heatwave_controller_comparison.png` |
| Fig. 9 | `figures/Fig27_mc_peak_temperature.png` |
| Fig. 10 | `figures/Fig20_longterm_retained_capacity.png` |
| Table III | `controller_summary.csv` |
| Table IV | `monte_carlo_summary.csv` |
| Section IV.E | `numerical_checks.csv` |

The other figures in `figures/` are additional diagnostics that are not shown in the article.

## Verifying the results

`verify_results.py` reruns the nominal cases and the 30 paired uncertainty scenarios and compares them with the stored CSV files (`python verify_results.py`, or `python verify_results.py --nominal-only` for a quick check).

## Outer-interval sensitivity replay added for reviewer response

The original 30 paired uncertainty simulations use a 600-s outer interval and a 360-s prediction horizon. This means that the cooling command remains fixed for 240 s beyond the forecast window. The previously deposited numerical-check CSV only covers 60–300 s for the 72-h nominal simulations.

The new `replay_update_interval.py` replays **the same original 30 sampled parameter sets** at a 300-s outer interval with the same 360-s horizon. It produces `interval_replay/mc_300_600_comparison.csv` and `interval_replay/mc_300_600_summary.csv`. All original simulation outputs remain unchanged.

| Outer update | Controller | Mean peak | P95 peak | Runs over 45 °C |
|---|---|---:|---:|---:|
| 600 s | P-control | 43.43 °C | 48.00 °C | 9/30 (30%) |
| 600 s | Aging-Robust | 41.38 °C | 47.77 °C | 3/30 (10%) |
| 300 s | P-control | 41.36 °C | 46.50 °C | 3/30 (10%) |
| 300 s | Aging-Robust | 40.35 °C | 45.66 °C | 3/30 (10%) |

The exceedance-frequency difference at 600 s is not present at 300 s. Results are specific to the test settings, not converged reliability probabilities. As the model's outer interval also governs electrical/input sampling and state updates, this replay does **not** isolate controller refresh rate from other discretization effects. The cooling ceiling remains 160 W.

## Versions

- v1.0: code and outputs for the original submission.
- v1.1: adds `verify_results.py`, `replay_update_interval.py`, and `interval_replay/`. All v1.0 files are unchanged.

## How to cite

P. Phuangchuen, P. Prabpal, S. Sonasang, K. Phanrattanachai, and W. Chotikankun, "Python code and simulation data for aging-aware predictive thermal management of lithium-ion battery packs under tropical fast charging," Zenodo, 2026, doi: 10.5281/zenodo.24365113.
