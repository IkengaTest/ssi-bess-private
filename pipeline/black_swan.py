"""
SSI-ENN BESS — Black Swan Risk Module (v1.0)
===============================================
Implements the three Black Swan families (12 enrichments: BS-F1-1 → BS-F3-4)
plus their interaction with the SSI scoring engine.

Family 1 — Extreme Weather (4 enrichments)
  BS-F1-1: Compound climate stress (simultaneous heat + drought + fire)
  BS-F1-2: Return-period scaling for IRI metrics
  BS-F1-3: Non-linear grid cascade from climate clustering
  BS-F1-4: Climate-driven demand spike (cooling load / heating load)

Family 2 — Change of Law (4 enrichments)
  BS-F2-1: Regulatory regime shift (capacity market redesign)
  BS-F2-2: Retroactive tariff adjustment (FiT clawback)
  BS-F2-3: Environmental permitting shock (new siting constraints)
  BS-F2-4: Network code revision (connection cost reallocation)

Family 3 — Geopolitical (4 enrichments)
  BS-F3-1: Supply chain disruption (battery cell shortage)
  BS-F3-2: Commodity price shock (lithium / cobalt spike)
  BS-F3-3: Currency / financing shock (ECB rate spike)
  BS-F3-4: Cross-border interconnector disruption

Each enrichment produces a multiplicative modifier on R_final or on
specific revenue streams. The module computes both point estimates
and stochastic distributions for the Monte Carlo engine.

Data sources: Copernicus ERA5/CMIP6, ARERA regulatory archive,
ECB/Eurostat macro series, ISPRA extreme events catalogue.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ssi.black_swan")


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

@dataclass
class BlackSwanConfig:
    """Parameters for black swan risk calculations."""
    # Family 1 — Extreme Weather
    compound_climate_weight: float = 0.15    # Max amplification from compound events
    return_period_base: float = 50.0         # Base return period (years) for design events
    cascade_threshold: float = 0.70          # Grid stress threshold for cascade
    demand_spike_factor: float = 0.25        # Max demand spike from extreme weather

    # Family 2 — Change of Law
    regime_shift_prob: float = 0.08          # Annual probability of major regulatory change
    tariff_clawback_severity: float = 0.15   # Revenue impact of retroactive tariff adjustment
    permitting_delay_months: float = 6.0     # Expected delay from new environmental rules
    network_code_cost_shift: float = 0.10    # Connection cost reallocation impact

    # Family 3 — Geopolitical
    supply_disruption_prob: float = 0.05     # Annual probability of supply chain disruption
    commodity_spike_multiplier: float = 2.0  # Lithium price multiplier in spike scenario
    rate_shock_bps: float = 200.0            # ECB rate increase in shock scenario
    interconnector_loss_pct: float = 0.20    # Cross-border capacity loss in disruption

    # General
    correlation_climate_grid: float = 0.40   # ρ(climate event, grid cascade)
    correlation_geo_commodity: float = 0.55  # ρ(geopolitical event, commodity spike)


DEFAULT_BS_CONFIG = BlackSwanConfig()


# ══════════════════════════════════════════════════════════════════════
#  FAMILY 1: EXTREME WEATHER
# ══════════════════════════════════════════════════════════════════════

def compute_compound_climate_stress(
    heat_score: float,
    drought_score: float,
    fire_score: float,
    weight: float = DEFAULT_BS_CONFIG.compound_climate_weight,
) -> float:
    """BS-F1-1: Compound climate stress multiplier.

    When multiple extreme weather conditions co-occur, the impact
    is super-additive. This captures the non-linear interaction.

    compound_stress = 1 + w × (heat × drought + heat × fire + drought × fire)

    Args:
        heat_score: Normalised heat wave intensity [0, 1]
        drought_score: Normalised drought severity [0, 1]
        fire_score: Normalised wildfire risk [0, 1]
        weight: Amplification weight

    Returns:
        Multiplicative stress factor ≥ 1.0
    """
    interaction = (heat_score * drought_score +
                   heat_score * fire_score +
                   drought_score * fire_score)
    return 1.0 + weight * interaction


def compute_return_period_scaling(
    iri_metric: float,
    current_return_period: float,
    target_return_period: float = 100.0,
    base_return_period: float = DEFAULT_BS_CONFIG.return_period_base,
) -> float:
    """BS-F1-2: Return-period scaling for IRI metrics.

    Scale IRI metrics from their baseline return period to a more
    extreme scenario. Uses GEV scaling: IRI_scaled = IRI × (T/T0)^ξ
    where ξ is the shape parameter (≈0.10 for Italian climate extremes).

    Returns:
        Scaled IRI metric value
    """
    xi = 0.10  # GEV shape parameter
    if current_return_period <= 0 or base_return_period <= 0:
        return iri_metric
    scaling = (target_return_period / base_return_period) ** xi
    return iri_metric * scaling


def compute_grid_cascade_probability(
    grid_stress: float,
    n_critical_nodes: int = 3,
    threshold: float = DEFAULT_BS_CONFIG.cascade_threshold,
) -> float:
    """BS-F1-3: Non-linear grid cascade probability.

    P(cascade) = 1 - (1 - p_node)^n  where p_node = σ((stress - threshold) × 10)

    When grid stress exceeds the cascade threshold, the probability
    of cascading failure increases non-linearly with the number of
    critical nodes.

    Returns:
        Cascade probability [0, 1]
    """
    z = 10.0 * (grid_stress - threshold)
    p_node = 1.0 / (1.0 + math.exp(-z))
    return 1.0 - (1.0 - p_node) ** n_critical_nodes


def compute_demand_spike(
    base_load: float,
    temperature_anomaly: float,
    spike_factor: float = DEFAULT_BS_CONFIG.demand_spike_factor,
) -> float:
    """BS-F1-4: Climate-driven demand spike.

    Load_spike = Load_base × (1 + spike_factor × max(0, T_anomaly / 10))

    Returns:
        Spiked demand (MW)
    """
    anomaly_norm = max(0.0, temperature_anomaly / 10.0)
    return base_load * (1.0 + spike_factor * min(1.0, anomaly_norm))


# ══════════════════════════════════════════════════════════════════════
#  FAMILY 2: CHANGE OF LAW
# ══════════════════════════════════════════════════════════════════════

def compute_regime_shift_impact(
    annual_revenue: float,
    prob: float = DEFAULT_BS_CONFIG.regime_shift_prob,
    severity_range: Tuple[float, float] = (0.05, 0.30),
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """BS-F2-1: Regulatory regime shift expected loss.

    Models the expected annual loss from capacity market redesign,
    ancillary service rule changes, or licensing regime overhaul.

    Returns:
        Dict with expected_loss, worst_case_loss, annual_probability
    """
    severity_mid = (severity_range[0] + severity_range[1]) / 2
    expected_loss = annual_revenue * prob * severity_mid
    worst_case = annual_revenue * severity_range[1]

    return {
        'expected_loss': round(expected_loss, 4),
        'worst_case_loss': round(worst_case, 4),
        'annual_probability': prob,
        'severity_mid': severity_mid,
    }


def compute_tariff_clawback(
    cumulative_revenue: float,
    clawback_severity: float = DEFAULT_BS_CONFIG.tariff_clawback_severity,
) -> float:
    """BS-F2-2: Retroactive tariff adjustment (FiT clawback).

    Returns:
        Clawback amount (€M)
    """
    return cumulative_revenue * clawback_severity


def compute_permitting_delay_cost(
    monthly_opportunity_cost: float,
    delay_months: float = DEFAULT_BS_CONFIG.permitting_delay_months,
) -> float:
    """BS-F2-3: Environmental permitting delay cost.

    Returns:
        Total delay cost (€M)
    """
    return monthly_opportunity_cost * delay_months


def compute_network_code_cost(
    connection_cost: float,
    cost_shift: float = DEFAULT_BS_CONFIG.network_code_cost_shift,
) -> float:
    """BS-F2-4: Network code connection cost reallocation.

    Returns:
        Additional connection cost (€M)
    """
    return connection_cost * cost_shift


# ══════════════════════════════════════════════════════════════════════
#  FAMILY 3: GEOPOLITICAL
# ══════════════════════════════════════════════════════════════════════

def compute_supply_chain_impact(
    capex: float,
    prob: float = DEFAULT_BS_CONFIG.supply_disruption_prob,
    delay_months: float = 12.0,
    cost_increase: float = 0.20,
) -> Dict[str, float]:
    """BS-F3-1: Supply chain disruption (battery cell shortage).

    Returns:
        Dict with expected_cost_increase, delay_months, probability
    """
    expected_increase = capex * prob * cost_increase
    return {
        'expected_cost_increase': round(expected_increase, 4),
        'worst_case_increase': round(capex * cost_increase, 4),
        'delay_months': delay_months,
        'annual_probability': prob,
    }


def compute_commodity_spike(
    capex: float,
    current_lithium_price: float = 15000.0,  # $/tonne
    spike_multiplier: float = DEFAULT_BS_CONFIG.commodity_spike_multiplier,
    lithium_fraction: float = 0.25,
) -> Dict[str, float]:
    """BS-F3-2: Commodity price shock impact.

    Returns:
        Dict with capex_increase, stressed_lithium_price
    """
    stressed_price = current_lithium_price * spike_multiplier
    capex_increase = capex * lithium_fraction * (spike_multiplier - 1.0)
    return {
        'capex_increase_M': round(capex_increase, 4),
        'stressed_lithium_price': round(stressed_price, 0),
        'lithium_fraction': lithium_fraction,
    }


def compute_rate_shock_impact(
    npv: float,
    wacc_base: float = 0.052,
    rate_shock_bps: float = DEFAULT_BS_CONFIG.rate_shock_bps,
    horizon: int = 25,
) -> Dict[str, float]:
    """BS-F3-3: ECB rate shock impact on NPV.

    Returns:
        Dict with npv_impact, wacc_stressed, npv_reduction_pct
    """
    wacc_stressed = wacc_base + rate_shock_bps / 10000
    # Simplified NPV sensitivity: ΔNPV ≈ NPV × Duration × ΔRate
    duration = horizon / 2  # Simplified modified duration
    npv_impact = -npv * duration * (rate_shock_bps / 10000)
    npv_reduction_pct = abs(npv_impact / npv) * 100 if npv != 0 else 0

    return {
        'npv_impact_M': round(npv_impact, 4),
        'wacc_stressed': round(wacc_stressed, 4),
        'npv_reduction_pct': round(npv_reduction_pct, 2),
    }


def compute_interconnector_disruption(
    import_dependency: float,
    price_premium: float = 15.0,  # €/MWh
    loss_pct: float = DEFAULT_BS_CONFIG.interconnector_loss_pct,
) -> Dict[str, float]:
    """BS-F3-4: Cross-border interconnector disruption.

    Higher import dependency → higher price impact when interconnector fails.

    Returns:
        Dict with price_spike, bess_opportunity
    """
    price_spike = price_premium * loss_pct * import_dependency
    # BESS benefits from price spikes — this is an UPSIDE tail event
    bess_opportunity = price_spike * 0.30  # Capture ~30% of spike
    return {
        'price_spike_eur_mwh': round(price_spike, 2),
        'bess_opportunity_eur_mwh': round(bess_opportunity, 2),
        'import_dependency': import_dependency,
    }


# ══════════════════════════════════════════════════════════════════════
#  COMPOSITE BLACK SWAN RISK SCORE
# ══════════════════════════════════════════════════════════════════════

def compute_black_swan_score(
    substation: dict,
    config: BlackSwanConfig = DEFAULT_BS_CONFIG,
) -> Dict[str, float]:
    """Compute composite black swan risk score for a substation.

    Aggregates across all three families into a single risk indicator.

    Returns:
        Dict with family scores and composite
    """
    r_median = substation.get('R_median', 0.4)
    components = substation.get('components', {})

    # Family 1 — use I component (infrastructure) as proxy for weather exposure
    comp_I = components.get('I', 0.3)
    comp_T = components.get('T', 0.3)
    heat_proxy = min(1.0, comp_I * 1.5)
    drought_proxy = min(1.0, comp_T * 0.8)
    fire_proxy = min(1.0, (comp_I + comp_T) * 0.5)

    f1_compound = compute_compound_climate_stress(heat_proxy, drought_proxy, fire_proxy)
    f1_cascade = compute_grid_cascade_probability(r_median)
    f1_score = (f1_compound - 1.0) + f1_cascade * 0.5  # [0, ~0.5]

    # Family 2 — regulatory risk (region-independent for now)
    f2_score = config.regime_shift_prob + config.network_code_cost_shift * 0.5  # [0, ~0.15]

    # Family 3 — geopolitical (globally correlated, not substation-specific)
    f3_score = (config.supply_disruption_prob + 0.05) * 0.5  # [0, ~0.05]

    # Composite (weighted: 50% weather, 30% regulatory, 20% geopolitical)
    composite = 0.50 * min(1.0, f1_score) + 0.30 * min(1.0, f2_score * 5) + 0.20 * min(1.0, f3_score * 10)

    return {
        'bs_f1_weather': round(min(1.0, f1_score), 4),
        'bs_f2_regulatory': round(min(1.0, f2_score * 5), 4),
        'bs_f3_geopolitical': round(min(1.0, f3_score * 10), 4),
        'bs_composite': round(min(1.0, composite), 4),
    }


# ══════════════════════════════════════════════════════════════════════
#  FLEET-LEVEL ENRICHMENT
# ══════════════════════════════════════════════════════════════════════

def enrich_fleet(
    substations: list,
    config: BlackSwanConfig = DEFAULT_BS_CONFIG,
    verbose: bool = False,
) -> list:
    """Apply black swan risk enrichment to all substations.

    Modifies substations in-place (adds 'black_swan' key).

    Args:
        substations: List of substation dicts
        config: Black swan parameters
        verbose: Print summary

    Returns:
        substations (modified in-place)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 6a: Black Swan Risk Enrichment")
        print(f"{'='*60}")

    scores = []
    for sub in substations:
        bs = compute_black_swan_score(sub, config)
        sub['black_swan'] = bs
        scores.append(bs['bs_composite'])

    if verbose:
        scores_arr = np.array(scores)
        print(f"  Substations enriched: {len(substations):,}")
        print(f"  BS Composite (median): {np.median(scores_arr):.4f}")
        print(f"  BS Composite (P95):    {np.percentile(scores_arr, 95):.4f}")
        print(f"  BS Composite (max):    {np.max(scores_arr):.4f}")

        # Family breakdown
        f1 = np.array([s['black_swan']['bs_f1_weather'] for s in substations])
        f2 = np.array([s['black_swan']['bs_f2_regulatory'] for s in substations])
        f3 = np.array([s['black_swan']['bs_f3_geopolitical'] for s in substations])
        print(f"  F1 Weather (median):   {np.median(f1):.4f}")
        print(f"  F2 Regulatory (med):   {np.median(f2):.4f}")
        print(f"  F3 Geopolitical (med): {np.median(f3):.4f}")

    return substations
