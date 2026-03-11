"""
SSI-ENN BESS — Actuarial Risk Pricing Module (v1.0)
=====================================================
Implements four actuarial frameworks for tail-risk pricing of BESS investments:

  1. CAT — Catastrophe Event-Loss Table (10+1 archetypes)
  2. EVT — Extreme Value Theory with GPD tail fitting
  3. TVaR — Tail Value-at-Risk from existing Monte Carlo distributions
  4. PAD — Prudential Adequacy Discount (WACC loading for tail risk)

Integration with cannibalization:
  - TVaR splits cannibalizable vs non-cannibalizable components
  - PAD includes κ_cann × (1−CRS) loading
  - CAT includes BESS_FLOOD archetype (freq=0.12/yr)

Formulas:
  TVaR_α(s) = E[L | L > VaR_α(s)] from MC distribution
  PAD(s) = PAD_base × (1 + κ_tail × Tail_Ratio + κ_ci × CI_norm + κ_cann × (1−CRS))
  WACC_adj(s) = WACC_base + PAD(s) / 10000
  GPD: P(X > x | X > u) = (1 + ξ(x-u)/σ̃)^(-1/ξ)

Data sources: ARERA TIQE (continuity tails), ERA5 extremes, INGV seismic,
ISPRA flood return periods, ECB monetary policy history.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger("ssi.actuarial")


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ActuarialConfig:
    """Parameters for actuarial risk pricing."""
    # TVaR
    tvar_alpha: float = 0.95         # Confidence level for TVaR
    tvar_alpha_stressed: float = 0.99  # Stressed TVaR

    # PAD
    pad_base_bps: float = 75.0       # Base PAD in basis points
    kappa_tail: float = 0.30         # TVaR tail loading factor
    kappa_ci: float = 0.15           # CI width loading factor
    kappa_cann: float = 0.25         # Cannibalization loading factor (from cann module)

    # EVT / GPD
    gpd_threshold_pct: float = 0.90  # Threshold for POT (Peaks Over Threshold)
    gpd_xi_prior: float = 0.25      # Prior shape parameter (0.20-0.40 range)

    # WACC
    wacc_base: float = 0.052         # Base WACC
    wacc_floor: float = 0.040        # Minimum WACC
    wacc_cap: float = 0.120          # Maximum WACC

    # CAT
    cat_simulation_years: int = 1000   # Simulation horizon for AAL (reduced for fleet speed)


DEFAULT_ACT_CONFIG = ActuarialConfig()


# ══════════════════════════════════════════════════════════════════════
#  1. CAT — CATASTROPHE EVENT-LOSS TABLE
# ══════════════════════════════════════════════════════════════════════

# 10+1 archetypes with annual frequency and loss severity (fraction of revenue)
CAT_ARCHETYPES = [
    # (name, annual_freq, severity_mean, severity_std, description)
    ('HEAT_EXTREME',    0.08, 0.12, 0.05, 'Extreme heatwave — grid thermal overload'),
    ('ICE_STORM',       0.04, 0.18, 0.08, 'Ice storm — overhead line collapse'),
    ('FLOOD_MAJOR',     0.06, 0.25, 0.10, 'Major flooding — substation inundation'),
    ('WILDFIRE',        0.03, 0.20, 0.08, 'Wildfire — line/infrastructure damage'),
    ('SEISMIC',         0.01, 0.40, 0.15, 'Seismic event — structural damage'),
    ('WINDSTORM',       0.05, 0.10, 0.04, 'Severe windstorm — line galloping'),
    ('DROUGHT',         0.07, 0.08, 0.03, 'Prolonged drought — cooling water shortage'),
    ('REG_SHOCK',       0.08, 0.15, 0.06, 'Regulatory regime change — revenue cliff'),
    ('SUPPLY_CHAIN',    0.05, 0.12, 0.05, 'Battery supply disruption — capex spike'),
    ('CYBER_ATTACK',    0.02, 0.30, 0.12, 'Cyber attack — operational disruption'),
    ('BESS_FLOOD',      0.12, 0.15, 0.06, 'BESS-specific flood (new archetype v1.0)'),
]


def compute_cat_aal(
    annual_revenue: float,
    archetypes: List[Tuple] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Compute Average Annual Loss (AAL) from CAT event-loss table.

    AAL = Σ_i (freq_i × severity_i × revenue)

    Also computes OEP (Occurrence Exceedance Probability) at key return periods.

    Args:
        annual_revenue: Annual revenue (€M)
        archetypes: Override archetype table
        rng: Random generator for simulation

    Returns:
        Dict with AAL, OEP values, per-archetype breakdown
    """
    if archetypes is None:
        archetypes = CAT_ARCHETYPES

    # Deterministic AAL
    aal = 0.0
    breakdown = {}
    for name, freq, sev_mean, sev_std, desc in archetypes:
        expected_loss = freq * sev_mean * annual_revenue
        aal += expected_loss
        breakdown[name] = {
            'frequency': freq,
            'severity_mean': sev_mean,
            'expected_loss_M': round(expected_loss, 4),
        }

    # Stochastic simulation for OEP
    if rng is None:
        rng = np.random.default_rng(42)

    n_years = DEFAULT_ACT_CONFIG.cat_simulation_years
    annual_losses = np.zeros(n_years)

    for name, freq, sev_mean, sev_std, desc in archetypes:
        # Poisson-distributed events per year
        n_events = rng.poisson(freq, n_years)
        for yr in range(n_years):
            if n_events[yr] > 0:
                severities = rng.normal(sev_mean, sev_std, n_events[yr])
                severities = np.clip(severities, 0, 1)
                annual_losses[yr] += np.sum(severities) * annual_revenue

    # OEP at key return periods
    oep = {}
    for rp in [10, 50, 100, 250, 500]:
        pct = 1.0 - 1.0 / rp
        oep[f'OEP_{rp}yr'] = round(float(np.percentile(annual_losses, pct * 100)), 4)

    return {
        'aal_M': round(aal, 4),
        'aal_pct_revenue': round(aal / annual_revenue * 100, 2) if annual_revenue > 0 else 0,
        'breakdown': breakdown,
        'oep': oep,
        'max_simulated_loss_M': round(float(np.max(annual_losses)), 4),
    }


# ══════════════════════════════════════════════════════════════════════
#  2. EVT — EXTREME VALUE THEORY (GPD TAIL)
# ══════════════════════════════════════════════════════════════════════

def fit_gpd_tail(
    losses: np.ndarray,
    threshold_pct: float = DEFAULT_ACT_CONFIG.gpd_threshold_pct,
    xi_prior: float = DEFAULT_ACT_CONFIG.gpd_xi_prior,
) -> Dict[str, float]:
    """Fit Generalised Pareto Distribution to exceedances.

    GPD: P(X > x | X > u) = (1 + ξ(x-u)/σ̃)^(-1/ξ)

    Uses Peaks Over Threshold (POT) approach.

    Args:
        losses: Array of loss values
        threshold_pct: Percentile for threshold selection
        xi_prior: Prior for shape parameter (Bayesian regularisation)

    Returns:
        Dict with fitted parameters and tail metrics
    """
    if len(losses) < 20:
        # Not enough data — use prior-based estimate
        return {
            'xi': xi_prior,
            'sigma_tilde': float(np.std(losses)) if len(losses) > 0 else 0.1,
            'threshold': float(np.percentile(losses, threshold_pct * 100)) if len(losses) > 0 else 0.0,
            'n_exceedances': 0,
            'method': 'prior',
        }

    threshold = float(np.percentile(losses, threshold_pct * 100))
    exceedances = losses[losses > threshold] - threshold

    if len(exceedances) < 5:
        return {
            'xi': xi_prior,
            'sigma_tilde': float(np.std(exceedances)) if len(exceedances) > 0 else 0.1,
            'threshold': threshold,
            'n_exceedances': len(exceedances),
            'method': 'prior_insufficient',
        }

    # MLE fit with regularisation toward prior
    try:
        shape, loc, scale = sp_stats.genpareto.fit(exceedances, floc=0)
        # Blend with prior (Bayesian shrinkage)
        xi_fitted = 0.7 * shape + 0.3 * xi_prior
        sigma_fitted = scale
    except Exception:
        xi_fitted = xi_prior
        sigma_fitted = float(np.std(exceedances))

    return {
        'xi': round(xi_fitted, 4),
        'sigma_tilde': round(sigma_fitted, 4),
        'threshold': round(threshold, 4),
        'n_exceedances': len(exceedances),
        'method': 'mle_regularised',
    }


def gpd_return_level(
    xi: float,
    sigma: float,
    threshold: float,
    n_exceedances: int,
    n_total: int,
    return_period: float,
) -> float:
    """Compute GPD return level for a given return period.

    x_T = u + (σ/ξ) × [(n × T / n_u)^ξ - 1]

    Args:
        xi: GPD shape parameter
        sigma: GPD scale parameter
        threshold: POT threshold
        n_exceedances: Number of threshold exceedances
        n_total: Total number of observations
        return_period: Return period in years

    Returns:
        Return level (loss amount)
    """
    if n_exceedances <= 0 or n_total <= 0:
        return threshold

    lambda_u = n_exceedances / n_total
    if abs(xi) < 1e-6:
        return threshold + sigma * math.log(lambda_u * return_period)
    return threshold + (sigma / xi) * ((lambda_u * return_period) ** xi - 1)


# ══════════════════════════════════════════════════════════════════════
#  3. TVaR — TAIL VALUE-AT-RISK
# ══════════════════════════════════════════════════════════════════════

def compute_tvar(
    r_samples: np.ndarray,
    alpha: float = DEFAULT_ACT_CONFIG.tvar_alpha,
) -> Dict[str, float]:
    """TVaR_α(s) = E[R | R > VaR_α(s)]

    Compute Tail Value-at-Risk from Monte Carlo R_final distribution.
    Higher TVaR means more tail risk (higher potential worst-case SSI scores).

    Args:
        r_samples: Array of R_final samples from Monte Carlo (one row per substation)
        alpha: Confidence level (0.95 or 0.99)

    Returns:
        Dict with VaR, TVaR, tail_ratio, excess_mean
    """
    if len(r_samples) == 0:
        return {'var': 0.0, 'tvar': 0.0, 'tail_ratio': 0.0, 'excess_mean': 0.0}

    var_alpha = float(np.percentile(r_samples, alpha * 100))
    tail_mask = r_samples >= var_alpha
    if tail_mask.sum() == 0:
        tvar = var_alpha
    else:
        tvar = float(np.mean(r_samples[tail_mask]))

    median_r = float(np.median(r_samples))
    tail_ratio = (tvar - median_r) / (median_r + 1e-10)
    excess_mean = tvar - var_alpha

    return {
        'var': round(var_alpha, 4),
        'tvar': round(tvar, 4),
        'tail_ratio': round(tail_ratio, 4),
        'excess_mean': round(excess_mean, 4),
    }


def compute_tvar_split(
    r_samples: np.ndarray,
    crs: float,
    alpha: float = DEFAULT_ACT_CONFIG.tvar_alpha,
) -> Dict[str, float]:
    """Cannibalization-split TVaR.

    Splits the tail into cannibalizable and non-cannibalizable components:
      TVaR_cann = TVaR × (1 - CRS)
      TVaR_local = TVaR × CRS

    Args:
        r_samples: MC R_final samples
        crs: Cannibalization Resilience Score
        alpha: Confidence level

    Returns:
        Dict with TVaR components
    """
    base = compute_tvar(r_samples, alpha)
    tvar = base['tvar']

    return {
        **base,
        'tvar_cannibalizable': round(tvar * (1 - crs), 4),
        'tvar_local_immune': round(tvar * crs, 4),
        'crs': round(crs, 4),
    }


# ══════════════════════════════════════════════════════════════════════
#  4. PAD — PRUDENTIAL ADEQUACY DISCOUNT
# ══════════════════════════════════════════════════════════════════════

def compute_pad(
    tail_ratio: float,
    ci_width: float,
    crs: float = 0.50,
    config: ActuarialConfig = DEFAULT_ACT_CONFIG,
) -> Dict[str, float]:
    """PAD(s) = PAD_base × (1 + κ_tail × Tail_Ratio + κ_ci × CI_norm + κ_cann × (1−CRS))

    The Prudential Adequacy Discount converts tail risk into a WACC loading
    that can be applied to NPV calculations for investor communication.

    Args:
        tail_ratio: (TVaR - median) / median from MC distribution
        ci_width: P95-P5 confidence interval width from MC
        crs: Cannibalization Resilience Score [0, 1]
        config: Actuarial parameters

    Returns:
        Dict with PAD bps, WACC adjustment, component breakdown
    """
    # Normalise CI width to [0, 1] (typical range 0.05-0.30)
    ci_norm = min(1.0, ci_width / 0.30)

    # PAD components
    tail_component = config.kappa_tail * max(0.0, tail_ratio)
    ci_component = config.kappa_ci * ci_norm
    cann_component = config.kappa_cann * (1.0 - crs)

    pad_multiplier = 1.0 + tail_component + ci_component + cann_component
    pad_bps = config.pad_base_bps * pad_multiplier

    # WACC adjustment
    wacc_adj = config.wacc_base + pad_bps / 10000
    wacc_adj = max(config.wacc_floor, min(config.wacc_cap, wacc_adj))

    return {
        'pad_bps': round(pad_bps, 2),
        'pad_multiplier': round(pad_multiplier, 4),
        'wacc_adjusted': round(wacc_adj, 4),
        'wacc_delta_bps': round((wacc_adj - config.wacc_base) * 10000, 2),
        'components': {
            'base_bps': config.pad_base_bps,
            'tail_loading': round(tail_component * config.pad_base_bps, 2),
            'ci_loading': round(ci_component * config.pad_base_bps, 2),
            'cann_loading': round(cann_component * config.pad_base_bps, 2),
        },
    }


# ══════════════════════════════════════════════════════════════════════
#  SUBSTATION-LEVEL ENRICHMENT
# ══════════════════════════════════════════════════════════════════════

def enrich_substation(
    substation: dict,
    config: ActuarialConfig = DEFAULT_ACT_CONFIG,
) -> Dict[str, float]:
    """Compute actuarial risk metrics for a single substation.

    Uses available MC distribution data (R_P5, R_P95, CI_width)
    to derive TVaR, PAD, and WACC adjustments.

    Args:
        substation: Substation dict from data.json
        config: Actuarial parameters

    Returns:
        Dict with all actuarial metrics
    """
    r_median = substation.get('R_median', 0.4)
    r_p5 = substation.get('R_P5', r_median - 0.05)
    r_p95 = substation.get('R_P95', r_median + 0.10)
    ci_width = substation.get('CI_width', r_p95 - r_p5)

    # Get CRS from cannibalization enrichment (if already computed)
    cann = substation.get('cannibalization', {})
    crs = cann.get('crs', 0.50)

    # ── Synthesise MC samples from summary statistics ──
    # Approximate the MC distribution from P5, median, P95
    # using a skew-normal approximation
    n_samples = 1000
    rng = np.random.default_rng(hash(substation.get('substation_id', '')) % 2**32)
    mean_approx = r_median
    std_approx = (r_p95 - r_p5) / 3.29  # ≈ 2 × 1.645
    samples = rng.normal(mean_approx, max(std_approx, 0.01), n_samples)
    samples = np.clip(samples, 0.0, 1.0)

    # ── TVaR ──
    tvar_95 = compute_tvar(samples, 0.95)
    tvar_99 = compute_tvar(samples, 0.99)
    tvar_split = compute_tvar_split(samples, crs, 0.95)

    # ── PAD & WACC ──
    pad = compute_pad(tvar_95['tail_ratio'], ci_width, crs, config)

    # ── CAT AAL (deterministic only — skip MC simulation for fleet speed) ──
    bess = substation.get('bess', {})
    cfg_b = bess.get('config_B', {})
    annual_revenue = cfg_b.get('V_Total_M', 2.0) / 25  # Approximate annual
    aal = sum(freq * sev * annual_revenue for _, freq, sev, _, _ in CAT_ARCHETYPES)
    cat_aal_pct = (aal / annual_revenue * 100) if annual_revenue > 0 else 0

    # ── GPD parameters (from prior — skip MLE fit for fleet speed) ──
    gpd_xi = config.gpd_xi_prior
    gpd_sigma = max(std_approx, 0.01)

    return {
        # TVaR
        'tvar_95': tvar_95['tvar'],
        'tvar_99': tvar_99['tvar'],
        'tail_ratio': tvar_95['tail_ratio'],
        'var_95': tvar_95['var'],
        'tvar_cannibalizable': tvar_split['tvar_cannibalizable'],
        'tvar_local_immune': tvar_split['tvar_local_immune'],

        # PAD
        'pad_bps': pad['pad_bps'],
        'wacc_adjusted': pad['wacc_adjusted'],
        'wacc_delta_bps': pad['wacc_delta_bps'],
        'pad_tail_loading': pad['components']['tail_loading'],
        'pad_cann_loading': pad['components']['cann_loading'],

        # CAT
        'cat_aal_M': round(aal, 4),
        'cat_aal_pct': round(cat_aal_pct, 2),

        # GPD
        'gpd_xi': round(gpd_xi, 4),
        'gpd_sigma': round(gpd_sigma, 4),
    }


# ══════════════════════════════════════════════════════════════════════
#  FLEET-LEVEL ENRICHMENT
# ══════════════════════════════════════════════════════════════════════

def enrich_fleet(
    substations: list,
    config: ActuarialConfig = DEFAULT_ACT_CONFIG,
    verbose: bool = False,
) -> list:
    """Apply actuarial risk pricing to all substations.

    Modifies substations in-place (adds 'actuarial' key).

    Args:
        substations: List of substation dicts
        config: Actuarial parameters
        verbose: Print summary

    Returns:
        substations (modified in-place)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 6c: Actuarial Risk Pricing Enrichment")
        print(f"{'='*60}")

    pad_vals = []
    wacc_vals = []
    tvar_vals = []

    for sub in substations:
        act = enrich_substation(sub, config)
        sub['actuarial'] = act
        pad_vals.append(act['pad_bps'])
        wacc_vals.append(act['wacc_adjusted'])
        tvar_vals.append(act['tvar_95'])

    if verbose:
        pad_arr = np.array(pad_vals)
        wacc_arr = np.array(wacc_vals)
        tvar_arr = np.array(tvar_vals)

        print(f"  Substations enriched: {len(substations):,}")
        print(f"  PAD (median):         {np.median(pad_arr):.1f} bps")
        print(f"  PAD (range):          [{np.min(pad_arr):.1f}, {np.max(pad_arr):.1f}] bps")
        print(f"  WACC adjusted (med):  {np.median(wacc_arr):.4f}")
        print(f"  WACC delta (median):  {np.median(wacc_arr - 0.052) * 10000:.1f} bps")
        print(f"  TVaR@95 (median):     {np.median(tvar_arr):.4f}")
        print(f"  TVaR@95 (P95):        {np.percentile(tvar_arr, 95):.4f}")

    return substations
