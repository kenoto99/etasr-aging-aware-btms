"""
Model-based extension of the published electro-thermal BTMS framework.

Purpose
-------
This script extends the original coupled 2RC electro-thermal model into a
reduced-order 8-cell pack study with four cooling strategies:
    1) Passive (no active cooling)
    2) Proportional control (P-control)
    3) Finite-horizon predictive control (MPC-like discrete optimizer)
    4) Aging-aware robust predictive control (AR-MPC)

The extension adds:
- cell-to-cell electrical/thermal variability,
- fast-charging and tropical heat-wave scenarios,
- progressive age-state feedback (capacity loss + resistance growth),
- uncertainty stress testing / Monte Carlo analysis,
- long-term retained-capacity screening index,
- >20 publication-style figures.

IMPORTANT SCIENTIFIC LIMITATION
-------------------------------
The numerical parameters are a transparent reference set derived from the
previous model-based study. They are NOT identified from a specific commercial
cell and the long-term retained-capacity quantity is a scenario-based screening
index, not experimentally validated State of Health (SoH). The model is intended
for hypothesis testing, controller comparison, and research design until public
or laboratory data are used for parameter calibration/validation.
"""

from __future__ import annotations
import os, json, time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------- output ---------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs_ext")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

RNG = np.random.default_rng(20261031)

# ------------------------------ parameters ------------------------------------
@dataclass
class Params:
    n_cells: int = 8
    Q_Ah: float = 50.0
    T_ref_K: float = 298.15
    R0_ref: float = 2.5e-3
    R1_ref: float = 1.2e-3
    C1_ref: float = 2.4e4
    R2_ref: float = 0.8e-3
    C2_ref: float = 9.0e4
    C_core: float = 760.0
    C_surf: float = 320.0
    R_cs: float = 0.12
    R_sa: float = 0.90
    R_neighbor: float = 1.8
    T_set_C: float = 32.0
    T_soft_C: float = 35.0
    T_limit_C: float = 45.0
    Qcool_max_pack: float = 160.0  # W heat removal, pack equivalent
    COP_ref: float = 3.0
    P_pump_W: float = 6.0
    Kp_pack: float = 28.0          # W/K pack command
    dt_s: float = 120.0
    horizon_steps: int = 6        # 6 prediction steps x 60 s = 6-min horizon
    days: int = 3
    # aging stress (dimensionless, screening only)
    Ea_cal: float = 31_700.0
    Ea_cyc: float = 20_000.0
    R_gas: float = 8.314
    k_soc: float = 1.6
    # long-term screening coefficients
    annual_calendar_loss_ref: float = 0.0045
    cycle_loss_per_efc_ref: float = 1.8e-5

P = Params()

CONTROLLERS = ["Passive", "P-control", "Predictive", "Aging-Robust"]

# fixed cell-to-cell variations (reproducible)
cell_idx = np.arange(P.n_cells)
CAP_VAR = np.array([0.985, 1.012, 0.998, 1.020, 0.975, 1.008, 0.992, 1.015])
RES_VAR = np.array([1.035, 0.982, 1.012, 0.970, 1.055, 1.000, 1.025, 0.990])
RSA_VAR = np.array([1.03, 0.97, 1.05, 0.95, 1.02, 0.98, 1.04, 0.96])
CTH_VAR = np.array([0.98, 1.03, 1.00, 1.02, 0.97, 1.01, 1.04, 0.99])

# ---------------------------- model functions ---------------------------------
def ocv_generic(soc: np.ndarray | float):
    s = np.clip(soc, 0.0, 1.0)
    return 3.0 + 1.18*s - 0.08*np.sin(np.pi*s)


def param_scale(soc, T_K):
    T_use = np.clip(np.asarray(T_K), 250.0, 380.0)
    temp = np.exp(np.clip(1800.0*(1.0/T_use - 1.0/P.T_ref_K), -10, 10))
    socfac = 1.0 + 0.9*(np.asarray(soc)-0.5)**2
    return temp * socfac


def ecm_params(soc, T_K, resistance_multiplier=1.0):
    s = param_scale(soc, T_K) * resistance_multiplier
    R0 = P.R0_ref*s
    R1 = P.R1_ref*s
    R2 = P.R2_ref*s
    C1 = P.C1_ref/np.maximum(s, 0.25)
    C2 = P.C2_ref/np.maximum(s, 0.25)
    return R0, R1, C1, R2, C2


def ambient_profile(hour, heatwave=False, shift_C=0.0):
    # Tropical profile: about 28–42 C. Day 2 can be a heat-wave day.
    h = np.asarray(hour)
    base = 35.0 + 7.0*np.sin(2*np.pi*(h-8.0)/24.0)
    if np.ndim(h) == 0:
        day = int(np.floor(float(h)/24.0))
        if heatwave and day == 1:
            base += 3.5
    else:
        day = np.floor(h/24.0).astype(int)
        if heatwave:
            base = base + 3.5*(day == 1)
    return base + shift_C


def current_crate(hour, fast_charge_scale=1.0, load_scale=1.0):
    # Positive = discharge; negative = charge.
    h = float(hour % 24.0)
    c = 0.0
    if 7.0 <= h < 8.5:
        c = 0.30 + 0.08*np.sin(2*np.pi*(h-7.0)/1.5)
    elif 12.0 <= h < 12.6:
        c = -1.20*fast_charge_scale
    elif 17.0 <= h < 19.0:
        c = 0.435 + 0.09*np.sin(2*np.pi*(h-17.0)/2.0)
    elif 22.0 <= h < 22.8:
        c = -0.75
    return c*load_scale


def cop_from_ambient(Tamb_C):
    return float(np.clip(P.COP_ref - 0.035*(Tamb_C-32.0), 2.15, 3.35))


def age_modifiers(age_index):
    """age_index=1.0 at BOL; 0.8 means 80% retained-capacity screening state."""
    ai = float(np.clip(age_index, 0.70, 1.0))
    cap_mult = ai
    # Resistance growth: +45% by 80% retained capacity (screening assumption)
    res_mult = 1.0 + 2.25*(1.0-ai)
    return cap_mult, res_mult


def degradation_rates(Tc_K, soc, I_A, cap_Ah):
    crate = np.abs(I_A)/np.maximum(cap_Ah, 1e-9)
    T_safe = np.clip(Tc_K, 250.0, 360.0)
    cal = np.exp(np.clip(-P.Ea_cal/P.R_gas*(1.0/T_safe - 1.0/P.T_ref_K), -20, 20))
    cal *= np.exp(np.clip(P.k_soc*(soc-0.5), -5, 5))
    cyc = (crate**1.3)*np.exp(np.clip(-P.Ea_cyc/P.R_gas*(1.0/T_safe - 1.0/P.T_ref_K), -20, 20))
    return cal, cyc


def pack_heat_predict(I_A, soc_mean, Tcore_C, age_index, n_cells=P.n_cells):
    cap_mult, res_mult = age_modifiers(age_index)
    R0, R1, _, R2, _ = ecm_params(soc_mean, Tcore_C+273.15, res_mult)
    # quasi-steady conservative resistive heat estimate for horizon prediction
    R_eff = float(R0 + 0.65*R1 + 0.45*R2)
    return n_cells*(I_A**2)*R_eff


def predictive_command(state, t_s, controller, prev_q, scenario):
    """Finite-horizon discrete receding-horizon optimizer.

    The prediction model is intentionally reduced-order: pack-average core/surface
    temperature with forecast heat and ambient. Aging-Robust evaluates nominal and
    stressed forecasts and includes an aging penalty. This is a model-predictive
    screening controller, not a production-grade NMPC implementation.
    """
    if controller == "Passive":
        return 0.0

    Ts_C = state["ts_C"]
    Tc_C = state["tc_C"]
    soc = state["soc"]
    age_index = scenario.get("age_index", 1.0)
    Tamb_C = state["Tamb_C"]

    if controller == "P-control":
        q = P.Kp_pack*(float(np.max(Ts_C))-P.T_set_C)
        return float(np.clip(q, 0.0, P.Qcool_max_pack))

    n_candidates = int(scenario.get("n_candidates", 4))
    hsteps = int(scenario.get("horizon_steps", P.horizon_steps))
    candidates = np.linspace(0.0, P.Qcool_max_pack, n_candidates)
    # Include previous command to reduce discretization chatter.
    candidates = np.unique(np.clip(np.r_[candidates, prev_q], 0, P.Qcool_max_pack))
    costs = []
    t0_h = t_s/3600.0
    pred_dt = float(scenario.get("prediction_dt_s", 60.0))
    soc_m = float(np.mean(soc))
    tc0 = float(np.mean(Tc_C))
    ts0 = float(np.mean(Ts_C))
    spread0 = float(np.ptp(Ts_C))

    robust_cases = [(0.0, 1.0)]
    if controller == "Aging-Robust":
        robust_cases = [(0.0,1.0),(1.5,1.05)]

    for q in candidates:
        case_costs = []
        for amb_bias, cur_mult in robust_cases:
            tc = tc0; ts = ts0; s = soc_m
            cost = 0.0
            for j in range(hsteps):
                hf = t0_h + j*pred_dt/3600.0
                Ta = ambient_profile(hf, heatwave=scenario.get("heatwave",False),
                                     shift_C=scenario.get("ambient_shift_C",0.0)) + amb_bias
                crate = current_crate(hf,
                    fast_charge_scale=scenario.get("fast_charge_scale",1.0),
                    load_scale=scenario.get("load_scale",1.0))*cur_mult
                cap_mult, _ = age_modifiers(age_index)
                I = crate*P.Q_Ah
                qgen = pack_heat_predict(I, s, tc, age_index)
                # pack-average thermal model. Cooling command q is pack heat removal.
                Cc = P.n_cells*P.C_core
                Cs = P.n_cells*P.C_surf
                Rcs = P.R_cs/P.n_cells
                Rsa = P.R_sa/P.n_cells
                dTc = (qgen - (tc-ts)/Rcs)/Cc
                dTs = ((tc-ts)/Rcs - (ts-Ta)/Rsa - q)/Cs
                tc += dTc*pred_dt
                ts += dTs*pred_dt
                s = np.clip(s - I*pred_dt/(3600*P.Q_Ah*cap_mult),0.05,0.95)

                thermal = 1.8*max(tc-P.T_set_C, 0.0)**2 + 0.55*max(ts-P.T_set_C,0.0)**2
                limit = 90.0*max(tc-P.T_soft_C,0.0)**2
                eweight = 0.0020 if controller == "Predictive" else 0.0035
                energy = eweight*(q/max(cop_from_ambient(Ta),1e-6))**2
                slew = 0.00035*(q-prev_q)**2 if j == 0 else 0.0
                aging = 0.0
                if controller == "Aging-Robust":
                    # relative aging pressure; larger at hot temperature and high C-rate
                    aging = 0.30*np.exp(np.clip((tc-32.0)/10.0,-5,5))*(abs(crate)**1.25 + 0.12)
                cost += thermal + limit + energy + slew + aging + 0.6*spread0**2
            case_costs.append(cost)
        costs.append(max(case_costs) if controller == "Aging-Robust" else case_costs[0])

    return float(candidates[int(np.argmin(costs))])


def simulate(controller="P-control", days=None, dt_s=None, scenario=None, record=True):
    scenario = {} if scenario is None else dict(scenario)
    days = P.days if days is None else int(days)
    dt = P.dt_s if dt_s is None else float(dt_s)
    # MPC horizon is defined for 60-s base step; in reduced MC runs we keep command logic but
    # prediction model still uses the reference 60-s horizon step.
    n = int(days*86400/dt)
    t = np.arange(n)*dt
    h = t/3600.0
    nc = P.n_cells

    age_index = scenario.get("age_index", 1.0)
    cap_mult, res_age_mult = age_modifiers(age_index)
    param_res_mult = scenario.get("res_mult", 1.0)
    rsa_mult = scenario.get("rsa_mult", 1.0)
    cth_mult = scenario.get("cth_mult", 1.0)

    cap_Ah = P.Q_Ah*cap_mult*CAP_VAR
    res_cell_mult = res_age_mult*RES_VAR*param_res_mult
    rsa_cell = P.R_sa*RSA_VAR*rsa_mult
    ccore = P.C_core*CTH_VAR*cth_mult
    csurf = P.C_surf*CTH_VAR*cth_mult

    soc = np.full(nc, 0.80)
    v1 = np.zeros(nc); v2 = np.zeros(nc)
    tc = np.full(nc, 32.0+273.15)
    ts = np.full(nc, 31.5+273.15)

    # histories
    H = {}
    if record:
        for key in ["soc_mean","soc_min","soc_max","vt_mean","tc_mean","tc_max","ts_mean","ts_max",
                    "tspread","Tamb_C","I_A","crate","qgen_W","qcool_W","pbtms_W","cal_rate","cyc_rate"]:
            H[key] = np.zeros(n)
        H["tc_cells_C"] = np.zeros((n,nc)); H["ts_cells_C"] = np.zeros((n,nc))

    prev_q = 0.0
    cal_int = 0.0; cyc_int = 0.0; e_btms_J = 0.0; e_heat_J = 0.0; throughput_Ah = 0.0
    max_core = -1e9; max_surf = -1e9; max_spread = 0.0; max_qgen=0.0
    time_over_40_s = 0.0; time_over_45_s = 0.0

    tic = time.perf_counter()
    for k in range(n):
        hr = h[k]
        Tamb_C = float(ambient_profile(hr, heatwave=scenario.get("heatwave",False),
                                      shift_C=scenario.get("ambient_shift_C",0.0)))
        crate = current_crate(hr,
            fast_charge_scale=scenario.get("fast_charge_scale",1.0),
            load_scale=scenario.get("load_scale",1.0))
        I = crate*P.Q_Ah

        # Enforce SOC safety by suppressing current when any cell would violate limits.
        if I > 0 and np.min(soc) <= 0.08:
            I = 0.0; crate = 0.0
        if I < 0 and np.max(soc) >= 0.94:
            I = 0.0; crate = 0.0

        # controller uses current state
        q_cmd = predictive_command({
            "ts_C":ts-273.15, "tc_C":tc-273.15, "soc":soc, "Tamb_C":Tamb_C
        }, t[k], controller, prev_q, scenario)
        # Simple actuator slew limit (W per model step)
        dq_lim = 35.0*(dt/60.0)
        q_cmd = float(np.clip(q_cmd, prev_q-dq_lim, prev_q+dq_lim))
        q_cmd = float(np.clip(q_cmd,0,P.Qcool_max_pack))
        prev_q = q_cmd

        R0,R1,C1,R2,C2 = ecm_params(soc, tc, res_cell_mult)
        # exact RC update using current input over interval
        a1 = np.exp(-dt/np.maximum(R1*C1,1e-9)); a2=np.exp(-dt/np.maximum(R2*C2,1e-9))
        v1 = a1*v1 + R1*(1-a1)*I
        v2 = a2*v2 + R2*(1-a2)*I
        OCV = ocv_generic(soc)
        vt = OCV - I*R0 - v1 - v2
        # Always-positive irreversible heat decomposition
        qgen_cells = I**2*R0 + (v1**2)/np.maximum(R1,1e-9) + (v2**2)/np.maximum(R2,1e-9)
        qgen_pack = float(np.sum(qgen_cells))

        # distribute cooling: 70% uniform + 30% biased to warmer surfaces
        if q_cmd > 0:
            exc = np.maximum((ts-273.15)-np.min(ts-273.15)+0.15, 0.05)
            w = 0.70*np.ones(nc)/nc + 0.30*exc/np.sum(exc)
            qcool_cells = q_cmd*w
        else:
            qcool_cells = np.zeros(nc)

        # thermal coupling among adjacent surfaces
        neigh = np.zeros(nc)
        for i in range(nc):
            if i>0: neigh[i] += ((ts[i-1]-ts[i])/P.R_neighbor)
            if i<nc-1: neigh[i] += ((ts[i+1]-ts[i])/P.R_neighbor)

        Ta_K = Tamb_C+273.15
        # Thermal network is integrated with stable internal substeps because the
        # core-surface time constant is shorter than the outer control interval.
        nsub = max(1, int(np.ceil(dt/10.0)))
        hth = dt/nsub
        for _ in range(nsub):
            neigh = np.zeros(nc)
            for i in range(nc):
                if i>0: neigh[i] += ((ts[i-1]-ts[i])/P.R_neighbor)
                if i<nc-1: neigh[i] += ((ts[i+1]-ts[i])/P.R_neighbor)
            dTc = (qgen_cells - (tc-ts)/P.R_cs)/ccore
            dTs = ((tc-ts)/P.R_cs - (ts-Ta_K)/rsa_cell + neigh - qcool_cells)/csurf
            tc = tc + dTc*hth
            ts = ts + dTs*hth

        # SoC update with cell capacity variation
        soc = np.clip(soc - I*dt/(3600.0*cap_Ah), 0.05, 0.95)

        cal,cyc = degradation_rates(tc,soc,I,cap_Ah)
        cal_mean=float(np.mean(cal)); cyc_mean=float(np.mean(cyc))
        cal_int += cal_mean*dt/86400.0
        cyc_int += cyc_mean*dt/3600.0
        cop=cop_from_ambient(Tamb_C)
        pbtms = q_cmd/cop + (P.P_pump_W if q_cmd>1e-6 else 0.0)
        e_btms_J += pbtms*dt
        e_heat_J += q_cmd*dt
        throughput_Ah += abs(I)*dt/3600.0

        tc_C = tc-273.15; ts_C=ts-273.15
        max_core=max(max_core,float(np.max(tc_C))); max_surf=max(max_surf,float(np.max(ts_C)))
        max_spread=max(max_spread,float(np.ptp(ts_C))); max_qgen=max(max_qgen,qgen_pack)
        time_over_40_s += dt if np.max(tc_C)>40 else 0.0
        time_over_45_s += dt if np.max(tc_C)>45 else 0.0

        if record:
            H["soc_mean"][k]=np.mean(soc); H["soc_min"][k]=np.min(soc); H["soc_max"][k]=np.max(soc)
            H["vt_mean"][k]=np.mean(vt); H["tc_mean"][k]=np.mean(tc_C); H["tc_max"][k]=np.max(tc_C)
            H["ts_mean"][k]=np.mean(ts_C); H["ts_max"][k]=np.max(ts_C); H["tspread"][k]=np.ptp(ts_C)
            H["Tamb_C"][k]=Tamb_C; H["I_A"][k]=I; H["crate"][k]=crate; H["qgen_W"][k]=qgen_pack
            H["qcool_W"][k]=q_cmd; H["pbtms_W"][k]=pbtms; H["cal_rate"][k]=cal_mean; H["cyc_rate"][k]=cyc_mean
            H["tc_cells_C"][k]=tc_C; H["ts_cells_C"][k]=ts_C

    runtime=time.perf_counter()-tic
    combined_exposure = 0.60*cal_int + 0.40*cyc_int
    # EFC based on cell throughput relative to nominal pack-cell capacity
    efc = throughput_Ah/(2*P.Q_Ah*cap_mult)
    metrics = {
        "controller": controller,
        "days": days,
        "age_index": age_index,
        "max_core_C": max_core,
        "max_surface_C": max_surf,
        "max_surface_spread_C": max_spread,
        "time_core_over_40_h": time_over_40_s/3600.0,
        "time_core_over_45_h": time_over_45_s/3600.0,
        "btms_energy_kWh": e_btms_J/3.6e6,
        "heat_removed_kWh": e_heat_J/3.6e6,
        "degradation_exposure_au": combined_exposure,
        "calendar_exposure_au": cal_int,
        "cycling_exposure_au": cyc_int,
        "throughput_Ah": throughput_Ah,
        "efc": efc,
        "peak_heat_generation_W": max_qgen,
        "runtime_s": runtime,
        "final_soc_mean": float(np.mean(soc)),
    }
    if record:
        H["time_h"] = h
        H["metrics"] = metrics
        return H
    return metrics

# ------------------------------- plotting -------------------------------------
def savefig(num, name):
    path=os.path.join(FIG,f"Fig{num:02d}_{name}.png")
    plt.savefig(path); plt.close()
    return path


def make_plots(results, metrics_df, age_df, sensitivity, mc_df):
    nfig=0
    rAR=results["Aging-Robust"]
    t=rAR["time_h"]

    # 1 ambient and current profile
    plt.figure(figsize=(6.6,4.1)); ax=plt.gca(); ax2=ax.twinx()
    ax.plot(t,rAR["Tamb_C"],label="Ambient temperature")
    ax2.plot(t,rAR["crate"],"--",label="Pack current")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Ambient temperature (°C)"); ax2.set_ylabel("Current (C-rate)")
    ax.set_title("Tropical heat-soak boundary condition and EV duty cycle"); ax.grid(True,ls=":",alpha=.45)
    lines=ax.lines+ax2.lines; ax.legend(lines,[x.get_label() for x in lines],loc="upper right")
    nfig+=1; savefig(nfig,"boundary_conditions")

    # 2 OCV-SOC
    s=np.linspace(.05,.95,200); plt.figure(figsize=(6.6,4.1)); plt.plot(s*100,ocv_generic(s))
    plt.xlabel("State of charge (%)"); plt.ylabel("Open-circuit voltage (V)"); plt.title("Generic OCV–SOC relation used by the model")
    plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"ocv_soc")

    # 3 R0 maps
    plt.figure(figsize=(6.6,4.1))
    for Tc in [25,35,45]: plt.plot(s*100,[ecm_params(x,Tc+273.15)[0]*1e3 for x in s],label=f"{Tc} °C")
    plt.xlabel("State of charge (%)"); plt.ylabel(r"$R_0$ (m$\Omega$)"); plt.title("Reference ohmic-resistance map")
    plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"r0_map")

    # 4 max core temperature time series
    plt.figure(figsize=(6.6,4.1))
    for c in CONTROLLERS: plt.plot(results[c]["time_h"],results[c]["tc_max"],label=c)
    plt.axhline(P.T_limit_C,ls=":",label="45 °C limit"); plt.xlabel("Time (h)"); plt.ylabel("Maximum cell-core temperature (°C)")
    plt.title("Maximum pack core temperature under four strategies"); plt.legend(); plt.grid(True,ls=":",alpha=.45)
    nfig+=1; savefig(nfig,"core_temperature_controllers")

    # 5 max surface temp
    plt.figure(figsize=(6.6,4.1))
    for c in CONTROLLERS: plt.plot(results[c]["time_h"],results[c]["ts_max"],label=c)
    plt.xlabel("Time (h)"); plt.ylabel("Maximum surface temperature (°C)"); plt.title("Maximum cell-surface temperature")
    plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"surface_temperature_controllers")

    # 6 cell spread
    plt.figure(figsize=(6.6,4.1))
    for c in CONTROLLERS: plt.plot(results[c]["time_h"],results[c]["tspread"],label=c)
    plt.xlabel("Time (h)"); plt.ylabel("Surface temperature spread (°C)"); plt.title("Pack temperature non-uniformity")
    plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"temperature_spread")

    # 7 individual cell cores AR
    plt.figure(figsize=(6.6,4.1))
    for i in range(P.n_cells): plt.plot(t,rAR["tc_cells_C"][:,i],lw=.9,label=f"Cell {i+1}")
    plt.xlabel("Time (h)"); plt.ylabel("Core temperature (°C)"); plt.title("Individual cell-core temperatures: Aging-Robust control")
    plt.legend(ncol=2); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"individual_core_AR")

    # 8 SOC envelope
    plt.figure(figsize=(6.6,4.1)); plt.fill_between(t,rAR["soc_min"]*100,rAR["soc_max"]*100,alpha=.25,label="Cell range")
    plt.plot(t,rAR["soc_mean"]*100,label="Pack mean SOC"); plt.xlabel("Time (h)"); plt.ylabel("SOC (%)")
    plt.title("SOC evolution and cell-to-cell envelope"); plt.legend(); plt.grid(True,ls=":",alpha=.45)
    nfig+=1; savefig(nfig,"soc_envelope")

    # 9 voltage
    plt.figure(figsize=(6.6,4.1)); plt.plot(t,rAR["vt_mean"]); plt.xlabel("Time (h)"); plt.ylabel("Mean terminal voltage per cell (V)")
    plt.title("Mean terminal voltage under Aging-Robust control"); plt.grid(True,ls=":",alpha=.45)
    nfig+=1; savefig(nfig,"terminal_voltage")

    # 10 heat generation
    plt.figure(figsize=(6.6,4.1))
    for c in CONTROLLERS: plt.plot(results[c]["time_h"],results[c]["qgen_W"],label=c)
    plt.xlabel("Time (h)"); plt.ylabel("Pack irreversible heat generation (W)"); plt.title("Electro-thermal heat generation")
    plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"heat_generation")

    # 11 cooling heat
    plt.figure(figsize=(6.6,4.1))
    for c in CONTROLLERS[1:]: plt.plot(results[c]["time_h"],results[c]["qcool_W"],label=c)
    plt.xlabel("Time (h)"); plt.ylabel("Cooling heat removal command (W)"); plt.title("BTMS cooling command")
    plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"cooling_command")

    # 12 electrical power
    plt.figure(figsize=(6.6,4.1))
    for c in CONTROLLERS[1:]: plt.plot(results[c]["time_h"],results[c]["pbtms_W"],label=c)
    plt.xlabel("Time (h)"); plt.ylabel("BTMS electrical power (W)"); plt.title("Parasitic BTMS electrical demand")
    plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"btms_power")

    # 13 cumulative energy
    plt.figure(figsize=(6.6,4.1))
    for c in CONTROLLERS[1:]:
        cum=np.cumsum(results[c]["pbtms_W"])*P.dt_s/3.6e6; plt.plot(results[c]["time_h"],cum,label=c)
    plt.xlabel("Time (h)"); plt.ylabel("Cumulative BTMS energy (kWh)"); plt.title("Cumulative cooling energy demand")
    plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"cumulative_energy")

    # 14 calendar stress
    plt.figure(figsize=(6.6,4.1))
    for c in CONTROLLERS: plt.plot(results[c]["time_h"],results[c]["cal_rate"],label=c)
    plt.xlabel("Time (h)"); plt.ylabel("Relative calendar-aging stress rate (a.u.)"); plt.title("Temperature/SOC-driven calendar-aging stress")
    plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"calendar_stress")

    # 15 cycling stress
    plt.figure(figsize=(6.6,4.1))
    for c in CONTROLLERS: plt.plot(results[c]["time_h"],results[c]["cyc_rate"],label=c)
    plt.xlabel("Time (h)"); plt.ylabel("Relative cycling-aging stress rate (a.u.)"); plt.title("C-rate/temperature-driven cycling stress")
    plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"cycling_stress")

    # 16 controller Tmax bar
    plt.figure(figsize=(6.6,4.1)); plt.bar(metrics_df.controller,metrics_df.max_core_C)
    plt.ylabel("Maximum core temperature (°C)"); plt.title("Controller comparison: peak temperature"); plt.xticks(rotation=12)
    plt.grid(True,axis="y",ls=":",alpha=.4); nfig+=1; savefig(nfig,"controller_peak_temperature")

    # 17 energy bar
    plt.figure(figsize=(6.6,4.1)); plt.bar(metrics_df.controller,metrics_df.btms_energy_kWh)
    plt.ylabel("3-day BTMS energy (kWh)"); plt.title("Controller comparison: cooling energy"); plt.xticks(rotation=12)
    plt.grid(True,axis="y",ls=":",alpha=.4); nfig+=1; savefig(nfig,"controller_energy")

    # 18 degradation exposure bar
    plt.figure(figsize=(6.6,4.1)); plt.bar(metrics_df.controller,metrics_df.degradation_exposure_au)
    plt.ylabel("3-day degradation exposure (a.u.)"); plt.title("Controller comparison: degradation exposure"); plt.xticks(rotation=12)
    plt.grid(True,axis="y",ls=":",alpha=.4); nfig+=1; savefig(nfig,"controller_exposure")

    # 19 Pareto
    plt.figure(figsize=(6.6,4.1))
    for _,r in metrics_df.iterrows():
        plt.scatter(r.btms_energy_kWh,r.degradation_exposure_au,s=45); plt.annotate(r.controller,(r.btms_energy_kWh,r.degradation_exposure_au),xytext=(5,4),textcoords="offset points")
    plt.xlabel("BTMS energy (kWh/3 days)"); plt.ylabel("Degradation exposure (a.u.)"); plt.title("Energy–degradation trade-off")
    plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"pareto_energy_exposure")

    # 20 long-term retained capacity
    plt.figure(figsize=(6.6,4.1))
    for c in CONTROLLERS: plt.plot(age_df[age_df.controller==c].year,age_df[age_df.controller==c].retained_capacity_pct,label=c)
    plt.axhline(80,ls=":",label="80% reference"); plt.xlabel("Repeated-duty year"); plt.ylabel("Retained-capacity screening index (%)")
    plt.title("Long-term retained-capacity screening projection"); plt.legend(); plt.grid(True,ls=":",alpha=.45)
    nfig+=1; savefig(nfig,"longterm_retained_capacity")

    # 21 resistance growth
    plt.figure(figsize=(6.6,4.1))
    for c in CONTROLLERS: plt.plot(age_df[age_df.controller==c].year,age_df[age_df.controller==c].resistance_multiplier,label=c)
    plt.xlabel("Repeated-duty year"); plt.ylabel("Resistance multiplier (-)"); plt.title("Age-feedback resistance growth assumption")
    plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"resistance_growth")

    # 22 age-state peak temp sensitivity
    ad=sensitivity["age"]
    plt.figure(figsize=(6.6,4.1))
    for c in ["P-control","Aging-Robust"]:
        d=ad[ad.controller==c]; plt.plot(d.age_index*100,d.max_core_C,"o-",label=c)
    plt.xlabel("Retained-capacity state (%)"); plt.ylabel("Maximum core temperature (°C)"); plt.title("Thermal sensitivity to battery age state")
    plt.gca().invert_xaxis(); plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"age_state_temperature")

    # 23 age-state energy sensitivity
    plt.figure(figsize=(6.6,4.1))
    for c in ["P-control","Aging-Robust"]:
        d=ad[ad.controller==c]; plt.plot(d.age_index*100,d.btms_energy_kWh,"o-",label=c)
    plt.xlabel("Retained-capacity state (%)"); plt.ylabel("BTMS energy (kWh/day)"); plt.title("Cooling-energy sensitivity to battery age state")
    plt.gca().invert_xaxis(); plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"age_state_energy")

    # 24 ambient sensitivity
    amb=sensitivity["ambient"]
    plt.figure(figsize=(6.6,4.1))
    for c in ["P-control","Aging-Robust"]:
        d=amb[amb.controller==c]; plt.plot(d.ambient_shift_C,d.max_core_C,"o-",label=c)
    plt.xlabel("Ambient profile shift (°C)"); plt.ylabel("Maximum core temperature (°C)"); plt.title("Ambient-temperature sensitivity")
    plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"ambient_sensitivity_temperature")

    # 25 fast-charge sensitivity
    fc=sensitivity["fastcharge"]
    plt.figure(figsize=(6.6,4.1))
    for c in ["P-control","Aging-Robust"]:
        d=fc[fc.controller==c]; plt.plot(d.fast_charge_scale*1.2,d.max_core_C,"o-",label=c)
    plt.xlabel("Fast-charge rate (C)"); plt.ylabel("Maximum core temperature (°C)"); plt.title("Fast-charging thermal sensitivity")
    plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"fastcharge_temperature")

    # 26 fast charge energy
    plt.figure(figsize=(6.6,4.1))
    for c in ["P-control","Aging-Robust"]:
        d=fc[fc.controller==c]; plt.plot(d.fast_charge_scale*1.2,d.btms_energy_kWh,"o-",label=c)
    plt.xlabel("Fast-charge rate (C)"); plt.ylabel("BTMS energy (kWh/day)"); plt.title("Fast-charging cooling-energy sensitivity")
    plt.legend(); plt.grid(True,ls=":",alpha=.45); nfig+=1; savefig(nfig,"fastcharge_energy")

    # 27 MC histogram / empirical distribution
    plt.figure(figsize=(6.6,4.1))
    for c in ["P-control","Aging-Robust"]:
        vals=mc_df.loc[mc_df.controller==c,"max_core_C"].values
        plt.hist(vals,bins=16,alpha=.45,label=c)
    plt.xlabel("Maximum core temperature (°C)"); plt.ylabel("Scenario count"); plt.title("Monte Carlo uncertainty: peak-temperature distribution")
    plt.legend(); plt.grid(True,axis="y",ls=":",alpha=.4); nfig+=1; savefig(nfig,"mc_peak_temperature")

    # 28 MC scatter energy vs Tmax
    plt.figure(figsize=(6.6,4.1))
    for c in ["P-control","Aging-Robust"]:
        d=mc_df[mc_df.controller==c]; plt.scatter(d.btms_energy_kWh,d.max_core_C,s=16,alpha=.65,label=c)
    plt.xlabel("BTMS energy (kWh/day)"); plt.ylabel("Maximum core temperature (°C)"); plt.title("Monte Carlo energy–temperature uncertainty cloud")
    plt.legend(); plt.grid(True,ls=":",alpha=.4); nfig+=1; savefig(nfig,"mc_energy_temperature")

    # 29 heatwave controller comparison
    hw=sensitivity["heatwave"]
    plt.figure(figsize=(6.6,4.1)); plt.bar(hw.controller,hw.max_core_C)
    plt.ylabel("Maximum core temperature (°C)"); plt.title("Heat-wave scenario: controller peak temperature"); plt.xticks(rotation=12)
    plt.grid(True,axis="y",ls=":",alpha=.4); nfig+=1; savefig(nfig,"heatwave_controller_comparison")

    # 30 runtime
    plt.figure(figsize=(6.6,4.1)); plt.bar(metrics_df.controller,metrics_df.runtime_s)
    plt.ylabel("Simulation runtime (s)"); plt.title("Computational cost for the 72-h simulation"); plt.xticks(rotation=12)
    plt.grid(True,axis="y",ls=":",alpha=.4); nfig+=1; savefig(nfig,"runtime")

    return nfig

# ---------------------------- study workflows ---------------------------------
def long_term_projection(base_results, years=10):
    rows=[]
    for c in CONTROLLERS:
        retained=1.0
        # use nominal 3-day stress as initial basis; update stress multiplier with age feedback
        base=base_results[c]["metrics"]
        cal_day=base["calendar_exposure_au"]/base["days"]
        efc_day=base["efc"]/base["days"]
        for y in range(years+1):
            _,resm=age_modifiers(retained)
            rows.append({"controller":c,"year":y,"retained_capacity_pct":100*retained,
                         "resistance_multiplier":resm})
            if y==years: break
            # screening update: hotter controller increases calendar term; cycling scales with EFC.
            thermal_mult=max(cal_day,0.25)
            annual_loss=P.annual_calendar_loss_ref*thermal_mult + P.cycle_loss_per_efc_ref*(efc_day*365)
            # aging feedback: higher resistance raises incremental thermal burden modestly
            annual_loss *= 1.0 + 0.35*(resm-1.0)
            retained=max(0.70,retained-annual_loss)
    return pd.DataFrame(rows)


def sensitivity_runs():
    sens={}
    rows=[]
    for age in [1.0,.9,.8]:
        for c in ["P-control","Aging-Robust"]:
            m=simulate(c,days=1,dt_s=300.0,scenario={"age_index":age,"horizon_steps":6,"n_candidates":4},record=False); m["age_index"]=age; rows.append(m)
    sens["age"]=pd.DataFrame(rows)

    rows=[]
    for sh in [-4,0,4]:
        for c in ["P-control","Aging-Robust"]:
            m=simulate(c,days=1,dt_s=300.0,scenario={"ambient_shift_C":sh,"horizon_steps":6,"n_candidates":4},record=False); m["ambient_shift_C"]=sh; rows.append(m)
    sens["ambient"]=pd.DataFrame(rows)

    rows=[]
    for sc in [.67,1.0,1.33]:  # 0.8C, 1.2C, 1.6C
        for c in ["P-control","Aging-Robust"]:
            m=simulate(c,days=1,dt_s=300.0,scenario={"fast_charge_scale":sc,"horizon_steps":6,"n_candidates":4},record=False); m["fast_charge_scale"]=sc; rows.append(m)
    sens["fastcharge"]=pd.DataFrame(rows)

    rows=[]
    for c in CONTROLLERS:
        m=simulate(c,days=3,dt_s=300.0,scenario={"heatwave":True,"horizon_steps":6,"n_candidates":4},record=False); rows.append(m)
    sens["heatwave"]=pd.DataFrame(rows)
    return sens


def numerical_checks():
    """Outer-interval sweep (60-300 s) and prediction-horizon sweep (3-12 min) for the
    nominal 72-h case, as reported in Section IV.E of the manuscript."""
    rows=[]
    for dt in [60.0,120.0,180.0,300.0]:
        for c in ["P-control","Predictive","Aging-Robust"]:
            m=simulate(c,days=3,dt_s=dt,scenario={},record=False)
            m.update({"check":"outer_interval","outer_interval_s":dt,"horizon_min":P.horizon_steps}); rows.append(m)
    for hs in [3,6,9,12]:
        for c in ["Predictive","Aging-Robust"]:
            m=simulate(c,days=3,dt_s=P.dt_s,scenario={"horizon_steps":hs},record=False)
            m.update({"check":"prediction_horizon","outer_interval_s":P.dt_s,"horizon_min":hs}); rows.append(m)
    return pd.DataFrame(rows)


def monte_carlo(n_scenarios=30):
    rows=[]
    # 24-h runs with a 10-min outer interval; the 6-min prediction horizon is retained.
    for i in range(n_scenarios):
        sc={
            "ambient_shift_C": float(RNG.normal(0,1.8)),
            "fast_charge_scale": float(np.clip(RNG.normal(1.0,.10),.75,1.30)),
            "load_scale": float(np.clip(RNG.normal(1.0,.07),.82,1.18)),
            "res_mult": float(np.clip(RNG.normal(1.0,.08),.82,1.22)),
            "rsa_mult": float(np.clip(RNG.normal(1.0,.08),.82,1.22)),
            "cth_mult": float(np.clip(RNG.normal(1.0,.05),.88,1.12)),
            "age_index": float(np.clip(RNG.normal(.90,.04),.80,1.00)),
        }
        for c in ["P-control","Aging-Robust"]:
            sc.update({"horizon_steps":6,"n_candidates":4})
            m=simulate(c,days=1,dt_s=600.0,scenario=sc,record=False)
            m.update({"scenario_id":i,**sc}); rows.append(m)
    return pd.DataFrame(rows)


def main():
    print("Running nominal 72-h simulations...")
    results={}
    for c in CONTROLLERS:
        results[c]=simulate(c,days=3,scenario={"heatwave":False},record=True)
        print(c, results[c]["metrics"])

    metrics_df=pd.DataFrame([results[c]["metrics"] for c in CONTROLLERS])
    metrics_df.to_csv(os.path.join(OUT,"controller_summary.csv"),index=False)

    print("Running long-term screening projection...")
    age_df=long_term_projection(results,years=10)
    age_df.to_csv(os.path.join(OUT,"long_term_screening.csv"),index=False)

    print("Running sensitivity studies...")
    sens=sensitivity_runs()
    for k,d in sens.items(): d.to_csv(os.path.join(OUT,f"sensitivity_{k}.csv"),index=False)

    print("Running Monte Carlo uncertainty study...")
    mc=monte_carlo(30)
    mc.to_csv(os.path.join(OUT,"monte_carlo_uncertainty.csv"),index=False)

    # uncertainty summary
    mc_summary=(mc.groupby("controller").agg(
        scenarios=("scenario_id","count"),
        Tmax_mean_C=("max_core_C","mean"),
        Tmax_p95_C=("max_core_C",lambda x:np.quantile(x,.95)),
        energy_mean_kWh=("btms_energy_kWh","mean"),
        exposure_mean_au=("degradation_exposure_au","mean"),
        violation_45_prob=("max_core_C",lambda x:np.mean(x>45.0)),
    ).reset_index())
    mc_summary.to_csv(os.path.join(OUT,"monte_carlo_summary.csv"),index=False)

    print("Running numerical checks...")
    numerical_checks().to_csv(os.path.join(OUT,"numerical_checks.csv"),index=False)

    print("Creating figures...")
    nfig=make_plots(results,metrics_df,age_df,sens,mc)

    summary={
        "model_type":"model-based reduced-order 8-cell electro-thermal-aging screening model",
        "controllers":CONTROLLERS,
        "figures_generated":nfig,
        "nominal_metrics":metrics_df.to_dict(orient="records"),
        "monte_carlo_summary":mc_summary.to_dict(orient="records"),
        "limitations":[
            "No laboratory calibration or validation is claimed.",
            "Long-term retained capacity is a scenario-based screening index, not measured SoH.",
            "Predictive controllers use a reduced-order finite-horizon discrete optimizer.",
            "Public or laboratory battery data should be used before quantitative lifetime claims."
        ]
    }
    with open(os.path.join(OUT,"simulation_summary.json"),"w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)

    # concise README
    with open(os.path.join(OUT,"README.txt"),"w",encoding="utf-8") as f:
        f.write("BTMS Aging-Aware Predictive Extension\n")
        f.write("====================================\n\n")
        f.write("Run: python BTMS_AgingAware_Predictive_Extension.py\n\n")
        f.write(f"Generated figures: {nfig}\n")
        f.write("Main outputs: controller_summary.csv, long_term_screening.csv, sensitivity_*.csv, monte_carlo_*.csv, simulation_summary.json\n\n")
        f.write("Scientific note: this is a model-based screening study. It does not claim experimental validation or chemistry-specific SoH prediction.\n")

    print("Figures generated:",nfig)
    print(metrics_df.to_string(index=False))
    print("\nMonte Carlo summary:\n",mc_summary.to_string(index=False))

if __name__ == "__main__":
    main()
