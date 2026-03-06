"""
BESS Computation Engine.
Computes Config A & B valuations for each substation:
V_DNO, V_Community, V_RO, NPV, IRR, Payback, recommendation, priority.
"""

import numpy as np
from typing import Dict, List

from .config import (
    BESS_CONFIGS, WACC, HORIZON, DEGRADATION,
    V_DNO_RANGES, V_COMMUNITY_RANGES, V_RO_RANGES,
    CLASSIFICATION_BANDS, PRIORITY_THRESHOLDS,
)


def compute_v_dno(r_median: float, config_name: str) -> float:
    """DNO value: higher R_median (worse grid) = higher BESS value."""
    min_val, max_val = V_DNO_RANGES[config_name]
    normalized = max(0.0, min(1.0, (r_median - 0.25) / 0.50))
    v_dno = min_val + (max_val - min_val) * normalized
    v_dno += np.random.normal(0, v_dno * 0.08)
    return round(max(min_val, min(max_val, v_dno)), 4)


def compute_v_community(r_median: float, ep_rate: float,
                        v_socio: float, config_name: str) -> float:
    """Community welfare value from energy poverty and social factors."""
    min_val, max_val = V_COMMUNITY_RANGES[config_name]
    ep_factor = min(0.3, ep_rate / 100.0) if ep_rate else 0.05
    social_factor = v_socio if v_socio else 0.3
    community_factor = ep_factor * social_factor * 3.0
    v_community = min_val + (max_val - min_val) * min(1.0, community_factor)
    v_community += np.random.normal(0, v_community * 0.12)
    return round(max(min_val, min(max_val, v_community)), 4)


def compute_v_ro(r_median: float, config_name: str) -> float:
    """Real Options value from Monte Carlo flexibility."""
    min_val, max_val = V_RO_RANGES[config_name]
    normalized = max(0.0, min(1.0, (r_median - 0.30) / 0.40))
    v_ro = min_val + (max_val - min_val) * normalized
    v_ro += np.random.normal(0, v_ro * 0.20)
    return round(max(min_val, min(max_val, v_ro)), 4)


def compute_npv(v_total: float, capex: float) -> float:
    """NPV = discounted total value stream - CAPEX."""
    pv = 0.0
    for year in range(1, HORIZON + 1):
        annual_value = (v_total / HORIZON) * (1 - DEGRADATION) ** year
        pv += annual_value / (1 + WACC) ** year
    return round(pv - capex, 4)


def compute_irr(v_total: float, capex: float) -> float:
    """Approximate IRR using binary search (50 iterations)."""
    if v_total <= capex:
        return round(max(0.0, (v_total / capex - 1) * 100 / HORIZON * 2), 2)

    low, high = 0.0, 0.50
    mid = 0.0
    for _ in range(50):
        mid = (low + high) / 2
        pv = sum(
            ((v_total / HORIZON) * (1 - DEGRADATION) ** y) / (1 + mid) ** y
            for y in range(1, HORIZON + 1)
        )
        if pv > capex:
            low = mid
        else:
            high = mid
    return round(mid * 100, 2)


def compute_payback(v_total: float, capex: float) -> float:
    """Simple payback in years."""
    annual = v_total / HORIZON
    if annual <= 0:
        return 25.0
    payback = capex / annual
    return round(min(25.0, max(1.0, payback)), 2)


def classify_band(r_median: float) -> str:
    """Classify R_median into band."""
    for band_name, lo, hi in CLASSIFICATION_BANDS:
        if r_median >= lo:
            return band_name
    return 'Low'


def compute_priority(irr_a: float, irr_b: float) -> int:
    """Investment priority 1-5 based on best IRR."""
    best_irr = max(irr_a, irr_b)
    for threshold, priority in PRIORITY_THRESHOLDS:
        if best_irr >= threshold:
            return priority
    return 1


def valuate_substation(sub: dict) -> dict:
    """
    Compute full BESS valuation for a substation.
    Adds 'bess' key with config_A, config_B, recommendation, investment_priority.
    Seeds RNG per substation for reproducibility.
    """
    r_med = sub.get('R_median', 0.4)
    socio = sub.get('socio_economic', {})
    ep_rate = socio.get('EP_rate_region', 8.0) if isinstance(socio, dict) else 8.0
    v_socio = socio.get('V_socio', 0.3) if isinstance(socio, dict) else 0.3

    # Seed per substation for reproducibility
    sid = sub.get('substation_id', sub.get('name', ''))
    np.random.seed(hash(sid) % 2**32)

    configs = {}
    for config_name, config in BESS_CONFIGS.items():
        v_dno = compute_v_dno(r_med, config_name)
        v_comm = compute_v_community(r_med, ep_rate, v_socio, config_name)
        v_ro = compute_v_ro(r_med, config_name)
        v_total = v_dno + v_comm + v_ro
        capex = config['capex_m']

        configs[f'config_{config_name}'] = {
            'V_DNO_M': round(v_dno, 2),
            'V_Community_M': round(v_comm, 2),
            'V_RO_M': round(v_ro, 2),
            'V_Total_M': round(v_total, 2),
            'CAPEX_M': capex,
            'NPV_M': round(compute_npv(v_total, capex), 2),
            'IRR_pct': round(compute_irr(v_total, capex), 1),
            'Payback_yr': round(compute_payback(v_total, capex), 1),
        }

    # Recommendation
    npv_a = configs['config_A']['NPV_M']
    npv_b = configs['config_B']['NPV_M']
    irr_a = configs['config_A']['IRR_pct']
    irr_b = configs['config_B']['IRR_pct']

    if npv_b > npv_a and irr_b > irr_a:
        rec = 'Config B'
    elif npv_a > npv_b and irr_a > irr_b:
        rec = 'Config A'
    elif npv_b > npv_a:
        rec = 'Config B'
    else:
        rec = 'Config A'

    configs['recommendation'] = rec
    configs['investment_priority'] = compute_priority(irr_a, irr_b)

    return configs


def valuate_fleet(substations: List[dict], verbose: bool = False) -> List[dict]:
    """
    Compute BESS valuations for all substations.
    Modifies substations in-place (adds 'bess' key) and returns them.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 5: Computing BESS valuations for {len(substations):,} substations")
        print(f"{'='*60}")

    for sub in substations:
        sub['bess'] = valuate_substation(sub)

    if verbose:
        npv_b = [s['bess']['config_B']['NPV_M'] for s in substations]
        irr_b = [s['bess']['config_B']['IRR_pct'] for s in substations]
        rec_b = sum(1 for s in substations if s['bess']['recommendation'] == 'Config B')
        print(f"  Median NPV (Config B): {np.median(npv_b):.2f}M")
        print(f"  Median IRR (Config B): {np.median(irr_b):.1f}%")
        print(f"  Config B recommended: {rec_b}/{len(substations)} ({100*rec_b/len(substations):.0f}%)")
        prio_dist = {}
        for s in substations:
            p = s['bess']['investment_priority']
            prio_dist[p] = prio_dist.get(p, 0) + 1
        for p in sorted(prio_dist):
            print(f"  Priority {p}: {prio_dist[p]:,}")

    return substations
