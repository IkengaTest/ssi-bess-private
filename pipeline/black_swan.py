"""
SSI-ENN BESS — Black Swan Risk Module (v1.1 — + Fuel-Electricity Nexus)
=========================================================================
Implements four Black Swan families (16 enrichments: BS-F1-1 → BS-F4-4)
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

Family 4 — Fuel-Electricity Nexus (4 enrichments, v1.1)
  BS-F4-1: Geopolitical fuel-price shock scenarios (gas/oil/LNG triggers)
  BS-F4-2: Fuel-to-electricity price transmission with R_conv(t) decay
  BS-F4-3: BESS arbitrage upside from fuel-driven price volatility
  BS-F4-4: Decarbonisation trajectory (EU 2030/2035/2050 conventional ratio)

  Transmission mechanism:
    Geopolitical shock → primary fuel pressure → merit order ΔP_elec →
    wider day/night spreads → BESS arbitrage upside

  Key formulas:
    R_conv(t) = piecewise-linear interpolation of EU decarbonisation targets:
                2024→0.58, 2030→0.42, 2035→0.25, 2050→0.05
    ΔP_elec(t) = R_conv(t) × β_pass × (M_fuel − 1) × P_elec_base
    BESS_upside = φ_spread × ΔP_elec(t) × capacity_factor
    Effective exposure = Σ_t [R_conv(t) × w_cashflow(t)] over horizon

  Data sources (Phase 2):
    BS-DS.1 — GME MGP (day-ahead zonal prices, actual spreads)
    BS-DS.2 — Terna Thermoelectric Statistics (conventional gen share)
    BS-DS.3 — TTF/PSV gas futures (ICIS / ICE Endex)
    BS-DS.4 — PNIEC / EU Fit-for-55 decarbonisation pathway

Each enrichment produces a multiplicative modifier on R_final or on
specific revenue streams. The module computes both point estimates
and stochastic distributions for the Monte Carlo engine.

Data sources: Copernicus ERA5/CMIP6, ARERA regulatory archive,
ECB/Eurostat macro series, ISPRA extreme events catalogue,
GME/TTF/PNIEC fuel-electricity nexus data.
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

    # Family 4 — Fuel-Electricity Nexus
    # Fuel shock scenarios (calibrated from 2021-22 European energy crisis)
    gas_crisis_prob: float = 0.04            # Annual probability of major gas disruption
    gas_spike_multiplier: float = 4.0        # TTF spike ratio (cf. 2022: €25→€340 ≈ 13.6×)
    oil_embargo_prob: float = 0.02           # Annual probability of oil supply shock
    oil_spike_multiplier: float = 2.5        # Brent spike ratio
    lng_disruption_prob: float = 0.03        # Annual probability of LNG supply disruption
    lng_spike_multiplier: float = 3.0        # Asian LNG diversion premium

    # Fuel-to-electricity transmission
    beta_pass_through: float = 0.85          # Merit order gas→electricity pass-through
    p_elec_base: float = 120.0              # Base electricity price (€/MWh, Italian zonal avg)
    phi_spread_capture: float = 0.40         # BESS spread capture fraction during spikes

    # Decarbonisation trajectory (Italy PNIEC + EU Fit-for-55)
    r_conv_2024: float = 0.58               # 2024 thermoelectric share (Terna 2024)
    r_conv_2030: float = 0.42               # PNIEC 2030 target
    r_conv_2035: float = 0.25               # EU 2035 interim (interpolated)
    r_conv_2050: float = 0.05               # EU net-zero 2050

    # General
    correlation_climate_grid: float = 0.40   # ρ(climate event, grid cascade)
    correlation_geo_commodity: float = 0.55  # ρ(geopolitical event, commodity spike)
    correlation_fuel_geo: float = 0.70       # ρ(geopolitical event, fuel spike)


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
#  FAMILY 4: FUEL-ELECTRICITY NEXUS (v1.1)
# ══════════════════════════════════════════════════════════════════════

# Regional thermoelectric dependency — higher values = more gas/oil generation
# Source: Terna 2024 "Dati Statistici — Produzione per fonte e zona"
_ZONE_THERMAL_DEPENDENCY = {
    'Nord': 0.55,          # Large CCGT fleet (Lombardia, Veneto, Piemonte)
    'Centro-Nord': 0.50,   # Toscana, Emilia-Romagna
    'Centro-Sud': 0.45,    # Lazio, Abruzzo, Campania
    'Sud': 0.65,           # Puglia, Calabria — high gas dependence
    'Sicilia': 0.75,       # Island, heavily thermal
    'Sardegna': 0.70,      # Coal + oil + gas (limited interconnection)
}

# Region → zone mapping for thermal dependency lookup
_REGION_TO_ZONE_THERMAL = {
    'Lombardia': 'Nord', 'Piemonte': 'Nord', 'Veneto': 'Nord',
    'Liguria': 'Nord', 'Friuli Venezia Giulia': 'Nord',
    "Valle d'Aosta": 'Nord', 'Trentino-Alto Adige': 'Nord',
    'Emilia-Romagna': 'Centro-Nord', 'Toscana': 'Centro-Nord',
    'Lazio': 'Centro-Sud', 'Marche': 'Centro-Sud', 'Umbria': 'Centro-Sud',
    'Abruzzo': 'Centro-Sud', 'Molise': 'Centro-Sud',
    'Campania': 'Centro-Sud',
    'Puglia': 'Sud', 'Calabria': 'Sud', 'Basilicata': 'Sud',
    'Sicilia': 'Sicilia',
    'Sardegna': 'Sardegna',
}


def _r_conv(year: float, config: BlackSwanConfig = DEFAULT_BS_CONFIG) -> float:
    """Conventional generation ratio R_conv(t) — piecewise linear interpolation.

    Declines per EU/PNIEC targets:
        2024 → 0.58 (current Italian thermoelectric share)
        2030 → 0.42 (PNIEC target)
        2035 → 0.25 (EU 2035 interim)
        2050 → 0.05 (EU net-zero)

    Returns:
        R_conv ∈ [0.05, 0.58]
    """
    milestones = [
        (2024, config.r_conv_2024),
        (2030, config.r_conv_2030),
        (2035, config.r_conv_2035),
        (2050, config.r_conv_2050),
    ]
    if year <= milestones[0][0]:
        return milestones[0][1]
    if year >= milestones[-1][0]:
        return milestones[-1][1]

    for i in range(len(milestones) - 1):
        y0, r0 = milestones[i]
        y1, r1 = milestones[i + 1]
        if y0 <= year <= y1:
            t = (year - y0) / (y1 - y0)
            return r0 + t * (r1 - r0)
    return milestones[-1][1]


def compute_fuel_shock_scenarios(
    config: BlackSwanConfig = DEFAULT_BS_CONFIG,
) -> Dict[str, Dict[str, float]]:
    """BS-F4-1: Geopolitical fuel-price shock scenarios.

    Three independent trigger scenarios calibrated from the 2021-22
    European energy crisis and historical commodity shocks:

    1. Gas crisis (TTF): Russia-style supply cut, LNG terminal failure,
       or Middle East pipeline disruption. Probability 4%/yr, multiplier 4×.
    2. Oil embargo: Major producer embargo or Strait of Hormuz closure.
       Probability 2%/yr, multiplier 2.5×.
    3. LNG diversion: Asian demand surge diverts LNG cargoes from Europe.
       Probability 3%/yr, multiplier 3×.

    Combined probability of ANY fuel shock ≈ 1-(1-p1)(1-p2)(1-p3) ≈ 8.7%/yr

    Returns:
        Dict of scenario name → {probability, multiplier, fuel_type}
    """
    scenarios = {
        'gas_crisis': {
            'probability': config.gas_crisis_prob,
            'multiplier': config.gas_spike_multiplier,
            'fuel_type': 'natural_gas',
            'trigger': 'Pipeline disruption / geopolitical cut-off',
            'reference': 'TTF 2022: €25 → €340/MWh',
        },
        'oil_embargo': {
            'probability': config.oil_embargo_prob,
            'multiplier': config.oil_spike_multiplier,
            'fuel_type': 'oil',
            'trigger': 'OPEC+ embargo / Strait of Hormuz closure',
            'reference': '1973 oil crisis: 4× price increase',
        },
        'lng_disruption': {
            'probability': config.lng_disruption_prob,
            'multiplier': config.lng_spike_multiplier,
            'fuel_type': 'lng',
            'trigger': 'Asian demand surge / terminal outage',
            'reference': 'JKM-TTF spread divergence 2021',
        },
    }

    # Combined probability of any shock
    p_none = 1.0
    for s in scenarios.values():
        p_none *= (1.0 - s['probability'])
    combined_prob = 1.0 - p_none

    # Probability-weighted expected multiplier
    expected_multiplier = sum(
        s['probability'] * s['multiplier'] for s in scenarios.values()
    ) / max(combined_prob, 1e-9)

    return {
        'scenarios': scenarios,
        'combined_annual_prob': round(combined_prob, 4),
        'expected_multiplier': round(expected_multiplier, 2),
    }


def compute_fuel_electricity_transmission(
    year: float,
    fuel_multiplier: float,
    region: str = '',
    config: BlackSwanConfig = DEFAULT_BS_CONFIG,
) -> Dict[str, float]:
    """BS-F4-2: Fuel-to-electricity price transmission.

    Geopolitical fuel shock → electricity price spike via merit order.
    The transmission is modulated by:
      1. R_conv(t): declining conventional share weakens the coupling
      2. β_pass: merit order pass-through coefficient
      3. Regional thermal dependency: zones with more gas CCGT get higher exposure

    ΔP_elec(t) = R_conv(t) × ζ_zone × β_pass × (M_fuel − 1) × P_elec_base

    where ζ_zone is the regional thermal dependency factor.

    Args:
        year: Assessment year (e.g. 2025, 2030, 2040)
        fuel_multiplier: Fuel price spike multiplier (e.g. 4.0 for gas crisis)
        region: Italian region name for zone thermal dependency
        config: Configuration parameters

    Returns:
        Dict with price_spike, r_conv, zone_factor, transmission details
    """
    r_conv_t = _r_conv(year, config)

    # Zone thermal dependency
    zone = _REGION_TO_ZONE_THERMAL.get(region, 'Centro-Sud')
    zeta_zone = _ZONE_THERMAL_DEPENDENCY.get(zone, 0.55)

    # Transmission formula
    fuel_shock_pct = max(0.0, fuel_multiplier - 1.0)
    delta_p_elec = (r_conv_t * zeta_zone * config.beta_pass_through *
                    fuel_shock_pct * config.p_elec_base)

    # Stressed electricity price
    p_elec_stressed = config.p_elec_base + delta_p_elec

    return {
        'delta_p_elec': round(delta_p_elec, 2),
        'p_elec_stressed': round(p_elec_stressed, 2),
        'r_conv': round(r_conv_t, 4),
        'zone_thermal_dependency': round(zeta_zone, 2),
        'zone': zone,
        'transmission_coefficient': round(r_conv_t * zeta_zone * config.beta_pass_through, 4),
    }


def compute_bess_fuel_upside(
    delta_p_elec: float,
    bess_mwh: float = 19.6,
    cycles_per_year: float = 300.0,
    config: BlackSwanConfig = DEFAULT_BS_CONFIG,
) -> Dict[str, float]:
    """BS-F4-3: BESS arbitrage upside from fuel-driven price volatility.

    Fuel price spikes widen day-night electricity price spreads.
    BESS captures a fraction φ of the incremental spread:

    BESS_upside_annual = φ × ΔP_elec × MWh × cycles × round-trip_eff

    This is an UPSIDE tail event — geopolitical fuel shocks are
    structurally beneficial for BESS owners who can cycle more
    profitably during high-spread periods.

    Args:
        delta_p_elec: Electricity price spike (€/MWh) from BS-F4-2
        bess_mwh: BESS energy capacity (MWh)
        cycles_per_year: Annual discharge cycles
        config: Configuration parameters

    Returns:
        Dict with annual upside, per-cycle gain, spread capture
    """
    round_trip_eff = 0.87  # Li-ion round-trip efficiency

    # Incremental spread captured
    spread_gain = config.phi_spread_capture * delta_p_elec

    # Annual upside revenue
    annual_upside = spread_gain * bess_mwh * cycles_per_year * round_trip_eff / 1e6

    # Per-cycle gain
    per_cycle = spread_gain * bess_mwh * round_trip_eff

    return {
        'spread_gain_eur_mwh': round(spread_gain, 2),
        'annual_upside_M': round(annual_upside, 4),
        'per_cycle_eur': round(per_cycle, 2),
        'phi_capture': config.phi_spread_capture,
    }


def compute_decarb_trajectory(
    base_year: int = 2025,
    horizon: int = 25,
    config: BlackSwanConfig = DEFAULT_BS_CONFIG,
) -> Dict[str, float]:
    """BS-F4-4: Decarbonisation trajectory — effective conventional exposure.

    Computes the cashflow-weighted average R_conv over the project horizon.
    Earlier years dominate because of discounting (WACC=5.2%), and R_conv
    is higher in earlier years, so effective exposure > simple average.

    effective_R_conv = Σ_t [R_conv(t) × w(t)] / Σ_t w(t)
    where w(t) = 1 / (1 + WACC)^t  (PV weight)

    Also provides milestone snapshots and a "fuel-shock sensitivity
    half-life" — the year at which R_conv drops below 50% of today's value.

    Args:
        base_year: Project start year
        horizon: Analysis horizon (years)
        config: Configuration parameters

    Returns:
        Dict with effective_r_conv, milestones, half_life, yearly profile
    """
    wacc = 0.052  # From config
    r_conv_base = _r_conv(base_year, config)

    weighted_sum = 0.0
    weight_sum = 0.0
    yearly = []
    half_life_year = None

    for t in range(horizon):
        year = base_year + t
        r_conv_t = _r_conv(year, config)
        w = 1.0 / (1.0 + wacc) ** t
        weighted_sum += r_conv_t * w
        weight_sum += w
        yearly.append({'year': year, 'r_conv': round(r_conv_t, 4), 'pv_weight': round(w, 4)})

        if half_life_year is None and r_conv_t <= r_conv_base * 0.5:
            half_life_year = year

    effective = weighted_sum / weight_sum if weight_sum > 0 else r_conv_base

    # Decay ratio: how much less exposed at end vs start
    r_conv_end = _r_conv(base_year + horizon, config)
    decay_ratio = r_conv_end / r_conv_base if r_conv_base > 0 else 0

    return {
        'effective_r_conv': round(effective, 4),
        'r_conv_base': round(r_conv_base, 4),
        'r_conv_end': round(r_conv_end, 4),
        'decay_ratio': round(decay_ratio, 4),
        'half_life_year': half_life_year,
        'milestones': {
            '2024': round(_r_conv(2024, config), 2),
            '2030': round(_r_conv(2030, config), 2),
            '2035': round(_r_conv(2035, config), 2),
            '2050': round(_r_conv(2050, config), 2),
        },
    }


def compute_fuel_nexus_composite(
    substation: dict,
    config: BlackSwanConfig = DEFAULT_BS_CONFIG,
    base_year: int = 2025,
    horizon: int = 25,
) -> Dict[str, object]:
    """Compute the full Family 4 fuel-electricity nexus for a substation.

    Chains BS-F4-1 → BS-F4-4 and produces:
      - fuel_shock_exposure: probability-weighted electricity price impact
      - bess_fuel_upside: BESS arbitrage revenue from fuel-driven spreads
      - decarb_discount: effective conventional ratio over project horizon

    Args:
        substation: Substation dict with region, BESS config, etc.
        config: Black swan configuration
        base_year: Project start year
        horizon: Project horizon (years)

    Returns:
        Dict with all Family 4 enrichment results + 3 NN features
    """
    region = substation.get('region', '')

    # BS-F4-1: Fuel shock scenarios
    scenarios = compute_fuel_shock_scenarios(config)

    # BS-F4-2: Transmission at multiple time horizons
    # Use probability-weighted expected multiplier
    expected_mult = scenarios['expected_multiplier']
    trans_now = compute_fuel_electricity_transmission(
        base_year, expected_mult, region, config
    )
    trans_2030 = compute_fuel_electricity_transmission(
        2030, expected_mult, region, config
    )
    trans_2040 = compute_fuel_electricity_transmission(
        2040, expected_mult, region, config
    )

    # BS-F4-3: BESS upside (use Config B as reference: 39.6 MWh)
    bess_b = substation.get('bess_B', {})
    bess_mwh = 39.6  # Config B
    upside = compute_bess_fuel_upside(trans_now['delta_p_elec'], bess_mwh, config=config)

    # BS-F4-4: Decarbonisation trajectory
    decarb = compute_decarb_trajectory(base_year, horizon, config)

    # ── NN Features ──
    # 1. fuel_shock_exposure: normalised [0,1] composite of probability × transmission × zonal dependency
    fuel_exposure = (scenarios['combined_annual_prob'] *
                     trans_now['transmission_coefficient'] *
                     (expected_mult - 1.0))
    fuel_exposure_norm = min(1.0, fuel_exposure / 0.80)  # Normalise to [0,1]

    # 2. bess_fuel_upside: normalised annual upside (€M, capped)
    bess_upside_norm = min(1.0, upside['annual_upside_M'] / 0.50)

    # 3. decarb_discount: effective R_conv (already in [0,1])
    decarb_discount = decarb['effective_r_conv']

    return {
        'scenarios': scenarios,
        'transmission_now': trans_now,
        'transmission_2030': trans_2030,
        'transmission_2040': trans_2040,
        'bess_upside': upside,
        'decarb_trajectory': decarb,
        # NN features
        'fuel_shock_exposure': round(fuel_exposure_norm, 4),
        'bess_fuel_upside': round(bess_upside_norm, 4),
        'decarb_discount': round(decarb_discount, 4),
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

    # Family 4 — fuel-electricity nexus
    fuel_nexus = compute_fuel_nexus_composite(substation, config)
    f4_score = fuel_nexus['fuel_shock_exposure']  # Already [0, 1]

    # Composite (weighted: 40% weather, 25% regulatory, 15% geopolitical, 20% fuel-nexus)
    composite = (0.40 * min(1.0, f1_score) +
                 0.25 * min(1.0, f2_score * 5) +
                 0.15 * min(1.0, f3_score * 10) +
                 0.20 * min(1.0, f4_score))

    return {
        'bs_f1_weather': round(min(1.0, f1_score), 4),
        'bs_f2_regulatory': round(min(1.0, f2_score * 5), 4),
        'bs_f3_geopolitical': round(min(1.0, f3_score * 10), 4),
        'bs_f4_fuel_nexus': round(min(1.0, f4_score), 4),
        'bs_composite': round(min(1.0, composite), 4),
        'fuel_nexus': fuel_nexus,
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
        f4 = np.array([s['black_swan']['bs_f4_fuel_nexus'] for s in substations])
        print(f"  F1 Weather (median):   {np.median(f1):.4f}")
        print(f"  F2 Regulatory (med):   {np.median(f2):.4f}")
        print(f"  F3 Geopolitical (med): {np.median(f3):.4f}")
        print(f"  F4 Fuel Nexus (med):   {np.median(f4):.4f}")

        # Fuel nexus summary
        fuel_exp = np.array([s['black_swan']['fuel_nexus']['fuel_shock_exposure'] for s in substations])
        bess_up = np.array([s['black_swan']['fuel_nexus']['bess_fuel_upside'] for s in substations])
        decarb = np.array([s['black_swan']['fuel_nexus']['decarb_discount'] for s in substations])
        print(f"\n  Fuel-Electricity Nexus (BS-F4):")
        print(f"    Fuel shock exposure (median): {np.median(fuel_exp):.4f}")
        print(f"    BESS fuel upside (median):    {np.median(bess_up):.4f}")
        print(f"    Decarb discount (effective):  {np.median(decarb):.4f}")
        print(f"    Combined annual shock prob:   {substations[0]['black_swan']['fuel_nexus']['scenarios']['combined_annual_prob']:.1%}")

    return substations
