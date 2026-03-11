"""
SSI-ENN BESS — Cannibalization Resilience Module (v1.0)
=========================================================
Implements the locality-adjusted cannibalization framework:

  Family 4 — Cannibalization (8 enrichments: CANN-P1-1 → CANN-P3-2)

Core insight: cannibalization is structurally asymmetric.
  - System-level active power services (arbitrage, aFRR, capacity market) are fully exposed
  - Local reactive power / PQaaS services are immune — reactive power cannot be transported

Key formulas:
  BESS_SAT(z)   = BESS_MW_installed(z) / Peak_Load(z)
  CRS(s)        = Σ(w_i × λ_i) / Σ(w_i)                     — Cannibalization Resilience Score
  R_i_adj       = R_i × [1 − (1−λ_i) × δ × f(BESS_SAT)]     — Revenue haircut
  f(x)          = min(1, (x / x_crit)^γ)                      — Saturation curve
  Spread_adj    = Spread_base × exp(−β × BESS_SAT)            — Spread compression
  PAD_CANN      = PAD_base × (1 + κ_cann × (1−CRS))          — Cannibalization PAD
  RPD_growth    = RPD_base × (1 + α_NL × NL_growth + α_CC × ClimateStress)

Data sources (zero new data for Phase 1 — all from Terna + ARERA):
  CN.1  Terna BESS Registry    → BESS_MW_installed per zone
  CN.2  ARERA Revenue Data     → Revenue stack weights
  CN.3  Terna Grid Plan        → Forward saturation (PNIEC trajectories)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ssi.cannibalization")

# ══════════════════════════════════════════════════════════════════════
#  LOCALITY FACTORS (λ_i) — Revenue stream immunity to cannibalization
# ══════════════════════════════════════════════════════════════════════
#
#  λ = 0.0 → fully cannibalizable (system-level, active power)
#  λ = 1.0 → fully immune (local, reactive power / PQaaS)
#
#  Values calibrated from Altinium BSP Valuation Paper v2 §§4-6
#  and validated against ARERA ancillary service definitions.

LOCALITY_FACTORS = {
    'R1_arbitrage':         0.05,   # Wholesale spread — fully system-level
    'R2_aFRR':              0.10,   # Automatic Frequency Restoration Reserve
    'R3_mFRR':              0.10,   # Manual Frequency Restoration Reserve
    'R4_capacity_mkt':      0.15,   # Capacity market (UM §5 contracts)
    'R5_wholesale':         0.05,   # Peak/off-peak wholesale trading
    'R6_self_consumption':  0.70,   # Behind-the-meter self-consumption
    'R7_LMP_premium':       0.50,   # Locational marginal price premium
    'R8_DSO_flexibility':   0.85,   # DSO flex procurement (Enel/Unareti)
    'R9_embedded_benefits':  0.80,   # TNUoS/DUoS avoidance credits
    'R10_PQaaS':            0.95,   # Power Quality as a Service — LOCAL
    'R11_CRS_climate':      0.90,   # Climate resilience services — LOCAL
}

# Default revenue weights when actual stack data is not available
# Calibrated for typical Italian MV BESS at high-SSI substation
DEFAULT_REVENUE_WEIGHTS = {
    'R1_arbitrage':         0.25,
    'R2_aFRR':              0.15,
    'R3_mFRR':              0.05,
    'R4_capacity_mkt':      0.10,
    'R5_wholesale':         0.05,
    'R6_self_consumption':  0.08,
    'R7_LMP_premium':       0.05,
    'R8_DSO_flexibility':   0.10,
    'R9_embedded_benefits':  0.02,
    'R10_PQaaS':            0.10,
    'R11_CRS_climate':      0.05,
}


# ══════════════════════════════════════════════════════════════════════
#  CORE PARAMETERS
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CannibalizationConfig:
    """Central configuration for cannibalization model."""
    # Saturation curve parameters
    x_crit: float = 0.20       # Critical BESS saturation threshold
    gamma: float = 2.0         # Saturation curve exponent (convex)
    delta: float = 0.60        # Maximum revenue haircut fraction at full saturation

    # Spread compression
    beta_spread: float = 4.0   # Spread compression coefficient (range 3.0–5.0)

    # PAD adjustment
    kappa_cann: float = 0.25   # Cannibalization PAD loading factor

    # Reactive power demand growth
    alpha_nonlinear: float = 0.02   # Non-linear load growth coefficient
    alpha_climate: float = 0.01     # Climate stress coefficient

    # Forward projection (years ahead for saturation forecasts)
    projection_years: int = 10

    # Zone definitions — Italian bidding zones → Terna zones
    zone_map: Dict[str, str] = field(default_factory=lambda: {
        'Nord': 'NORD',
        'Centro-Nord': 'CNOR',
        'Centro-Sud': 'CSUD',
        'Sud': 'SUD',
        'Sicilia': 'SICI',
        'Sardegna': 'SARD',
    })


DEFAULT_CONFIG = CannibalizationConfig()


# ══════════════════════════════════════════════════════════════════════
#  CANN-P1-1: BESS Zonal Saturation Ratio
# ══════════════════════════════════════════════════════════════════════

def compute_bess_saturation(
    bess_mw_installed: float,
    peak_load_mw: float,
) -> float:
    """BESS_SAT(z) = BESS_MW_installed(z) / Peak_Load(z)

    Measures how saturated a bidding zone is with BESS capacity
    relative to its peak demand.

    Args:
        bess_mw_installed: Total installed BESS MW in the zone
        peak_load_mw: Peak load in MW for the zone

    Returns:
        Saturation ratio [0, ∞) — typically 0.0 to 0.5
    """
    if peak_load_mw <= 0:
        log.warning("Peak load ≤ 0, returning 0.0 saturation")
        return 0.0
    return bess_mw_installed / peak_load_mw


def saturation_curve(
    bess_sat: float,
    x_crit: float = DEFAULT_CONFIG.x_crit,
    gamma: float = DEFAULT_CONFIG.gamma,
) -> float:
    """f(x) = min(1, (x / x_crit)^γ)

    Non-linear saturation function. Convex (γ=2.0) means impact
    accelerates as saturation approaches critical threshold.

    Returns:
        [0, 1] — 0 = no saturation effect, 1 = fully saturated
    """
    if bess_sat <= 0:
        return 0.0
    return min(1.0, (bess_sat / x_crit) ** gamma)


# ══════════════════════════════════════════════════════════════════════
#  CANN-P1-2: Cannibalization Resilience Score (CRS)
# ══════════════════════════════════════════════════════════════════════

def compute_crs(
    revenue_weights: Optional[Dict[str, float]] = None,
    locality_factors: Optional[Dict[str, float]] = None,
) -> float:
    """CRS(s) = Σ_i(w_i × λ_i) / Σ_i(w_i)

    Revenue-weighted average of locality factors. Higher CRS = more
    resilient to cannibalization.

    Interpretation:
      CRS > 0.60 → "Cannibalization-resilient" (dominated by local services)
      CRS ∈ [0.30, 0.60] → "Mixed exposure"
      CRS < 0.30 → "Cannibalization-exposed" (dominated by system services)

    Args:
        revenue_weights: Revenue stack weights per stream {R_name: weight}
        locality_factors: Override locality factors (default: LOCALITY_FACTORS)

    Returns:
        CRS ∈ [0, 1]
    """
    if revenue_weights is None:
        revenue_weights = DEFAULT_REVENUE_WEIGHTS
    if locality_factors is None:
        locality_factors = LOCALITY_FACTORS

    numerator = sum(
        w * locality_factors.get(k, 0.50)
        for k, w in revenue_weights.items()
    )
    denominator = sum(revenue_weights.values())

    if denominator <= 0:
        return 0.50  # No revenue data → assume moderate resilience
    return numerator / denominator


def classify_crs(crs: float) -> str:
    """Classify CRS into investor-friendly tier."""
    if crs >= 0.60:
        return 'Resilient'
    elif crs >= 0.30:
        return 'Mixed'
    else:
        return 'Exposed'


# ══════════════════════════════════════════════════════════════════════
#  CANN-P2-1: Revenue Stream Haircut
# ══════════════════════════════════════════════════════════════════════

def compute_revenue_haircut(
    r_base: float,
    lambda_i: float,
    bess_sat: float,
    delta: float = DEFAULT_CONFIG.delta,
    x_crit: float = DEFAULT_CONFIG.x_crit,
    gamma: float = DEFAULT_CONFIG.gamma,
) -> float:
    """R_i_adj = R_i × [1 − (1 − λ_i) × δ × f(BESS_SAT)]

    Locality-adjusted revenue haircut. Streams with high λ (local)
    are barely affected; streams with low λ (system) take full haircut.

    Args:
        r_base: Base revenue for this stream (€/MWh or annual €M)
        lambda_i: Locality factor for this stream [0, 1]
        bess_sat: Zonal BESS saturation ratio
        delta: Maximum haircut fraction at full saturation
        x_crit: Critical saturation threshold
        gamma: Saturation curve exponent

    Returns:
        Adjusted revenue (always ≤ r_base)
    """
    f_sat = saturation_curve(bess_sat, x_crit, gamma)
    haircut = (1.0 - lambda_i) * delta * f_sat
    return r_base * (1.0 - haircut)


def compute_total_revenue_adjusted(
    revenue_streams: Dict[str, float],
    bess_sat: float,
    config: CannibalizationConfig = DEFAULT_CONFIG,
) -> Tuple[float, float, Dict[str, float]]:
    """Apply haircuts to all revenue streams and return adjusted totals.

    Returns:
        (total_base, total_adjusted, stream_details)
    """
    total_base = 0.0
    total_adj = 0.0
    details = {}

    for stream, base_value in revenue_streams.items():
        lambda_i = LOCALITY_FACTORS.get(stream, 0.50)
        adj_value = compute_revenue_haircut(
            base_value, lambda_i, bess_sat,
            config.delta, config.x_crit, config.gamma,
        )
        total_base += base_value
        total_adj += adj_value
        details[stream] = {
            'base': round(base_value, 4),
            'adjusted': round(adj_value, 4),
            'haircut_pct': round((1.0 - adj_value / base_value) * 100, 2) if base_value > 0 else 0.0,
            'lambda': lambda_i,
        }

    return total_base, total_adj, details


# ══════════════════════════════════════════════════════════════════════
#  CANN-P2-2: Spread Compression
# ══════════════════════════════════════════════════════════════════════

def compute_spread_compression(
    spread_base: float,
    bess_sat: float,
    beta: float = DEFAULT_CONFIG.beta_spread,
) -> float:
    """Spread_adj = Spread_base × exp(−β × BESS_SAT)

    Exponential decay model for wholesale spread compression
    as BESS fleet grows. This is the primary mechanism through
    which arbitrage revenues erode.

    Args:
        spread_base: Base wholesale spread (€/MWh)
        bess_sat: Zonal BESS saturation ratio
        beta: Compression coefficient (3.0–5.0)

    Returns:
        Compressed spread (€/MWh)
    """
    return spread_base * math.exp(-beta * bess_sat)


# ══════════════════════════════════════════════════════════════════════
#  CANN-P2-3: Cannibalization-Adjusted PAD
# ══════════════════════════════════════════════════════════════════════

def compute_pad_cannibalization(
    pad_base_bps: float,
    crs: float,
    kappa_cann: float = DEFAULT_CONFIG.kappa_cann,
) -> float:
    """PAD_CANN = PAD_base × (1 + κ_cann × (1 − CRS))

    Substations with low CRS (exposed to cannibalization) get a PAD
    loading to reflect the additional revenue uncertainty.

    Args:
        pad_base_bps: Base Prudential Adequacy Discount in basis points
        crs: Cannibalization Resilience Score [0, 1]
        kappa_cann: Loading factor (default 0.25 = +25% PAD for CRS=0)

    Returns:
        Adjusted PAD in basis points
    """
    return pad_base_bps * (1.0 + kappa_cann * (1.0 - crs))


# ══════════════════════════════════════════════════════════════════════
#  CANN-P3-1: Forward Saturation Projection
# ══════════════════════════════════════════════════════════════════════

def project_saturation(
    bess_sat_current: float,
    annual_growth_rate: float = 0.15,
    years: int = DEFAULT_CONFIG.projection_years,
) -> List[float]:
    """BESS_SAT(z, t) stochastic forward projection from PNIEC.

    Simple deterministic trajectory for Phase 1.
    Phase 2 will add stochastic Monte Carlo around PNIEC targets.

    Args:
        bess_sat_current: Current zonal saturation
        annual_growth_rate: Expected annual BESS growth (default 15%/yr from PNIEC)
        years: Projection horizon

    Returns:
        List of saturation values for years [0, 1, ..., years]
    """
    trajectory = [bess_sat_current]
    for t in range(1, years + 1):
        next_sat = trajectory[-1] * (1.0 + annual_growth_rate)
        trajectory.append(next_sat)
    return trajectory


# ══════════════════════════════════════════════════════════════════════
#  CANN-P3-2: Reactive Power Demand Growth
# ══════════════════════════════════════════════════════════════════════

def compute_rpd_growth(
    rpd_base: float,
    nonlinear_load_growth: float = 0.03,
    climate_stress: float = 0.02,
    alpha_nl: float = DEFAULT_CONFIG.alpha_nonlinear,
    alpha_cc: float = DEFAULT_CONFIG.alpha_climate,
) -> float:
    """RPD_growth = RPD_base × (1 + α_NL × NonLinearLoad_growth + α_CC × ClimateStress)

    Reactive power demand INCREASES over time due to:
    - Non-linear loads (EV chargers, power electronics, data centres)
    - Climate stress (higher cooling loads with poor power factor)

    This is the counter-cyclical tailwind: while active power revenues
    erode through cannibalization, reactive power demand (and hence
    PQaaS revenue) grows — strengthening high-CRS substations.

    Args:
        rpd_base: Base reactive power demand (MVAr)
        nonlinear_load_growth: Annual growth in non-linear loads (fractional)
        climate_stress: Climate-driven cooling load stress (fractional)

    Returns:
        Projected reactive power demand (MVAr)
    """
    return rpd_base * (1.0 + alpha_nl * nonlinear_load_growth + alpha_cc * climate_stress)


# ══════════════════════════════════════════════════════════════════════
#  SUBSTATION-LEVEL ENRICHMENT
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CannibalizationResult:
    """Complete cannibalization assessment for one substation."""
    # Inputs
    substation_id: str
    zone: str
    bess_sat: float
    crs: float
    crs_tier: str

    # Revenue impact
    revenue_haircut_pct: float       # Total revenue haircut (%)
    spread_compression_pct: float    # Spread compression (%)
    pad_adjustment_bps: float        # Cannibalization PAD (bps)

    # Forward projections
    saturation_5yr: float
    saturation_10yr: float
    rpd_growth_factor: float

    # Derived features for NN
    crs_x_ssi: float               # CRS × R_median interaction
    exposure_index: float           # (1 - CRS) × BESS_SAT — compound exposure


def enrich_substation(
    substation: dict,
    zone_data: Optional[Dict] = None,
    config: CannibalizationConfig = DEFAULT_CONFIG,
) -> CannibalizationResult:
    """Apply full cannibalization enrichment to a single substation.

    This is the main entry point called from the ingestion pipeline.

    Args:
        substation: Substation dict from data.json
        zone_data: Optional zone-level data {zone: {bess_mw, peak_load_mw}}
        config: Cannibalization parameters

    Returns:
        CannibalizationResult with all computed fields
    """
    sid = substation.get('substation_id', '')
    region = substation.get('region', '')
    r_median = substation.get('R_median', 0.4)

    # ── Zone mapping ──
    # Map region to bidding zone (simplified Italian macro-zones)
    zone = _region_to_zone(region)

    # ── BESS saturation ──
    if zone_data and zone in zone_data:
        zd = zone_data[zone]
        bess_sat = compute_bess_saturation(
            zd.get('bess_mw', 0), zd.get('peak_load_mw', 1)
        )
    else:
        # Phase 1 default: use estimated national saturation
        bess_sat = _default_zone_saturation(zone)

    # ── CRS ──
    # If substation has revenue stack data, use it; else defaults
    revenue_weights = substation.get('revenue_weights', None)
    crs = compute_crs(revenue_weights)
    crs_tier = classify_crs(crs)

    # ── Revenue haircut (aggregate) ──
    # Use default revenue mix scaled by R_median as proxy for total revenue
    total_base = 1.0  # Normalised
    total_adj = 0.0
    for stream, weight in DEFAULT_REVENUE_WEIGHTS.items():
        lambda_i = LOCALITY_FACTORS.get(stream, 0.50)
        adj = compute_revenue_haircut(weight, lambda_i, bess_sat,
                                       config.delta, config.x_crit, config.gamma)
        total_adj += adj
    revenue_haircut_pct = (1.0 - total_adj / total_base) * 100 if total_base > 0 else 0.0

    # ── Spread compression ──
    spread_base = 40.0  # €/MWh typical Italian spread
    spread_adj = compute_spread_compression(spread_base, bess_sat, config.beta_spread)
    spread_compression_pct = (1.0 - spread_adj / spread_base) * 100

    # ── PAD adjustment ──
    pad_base = 75.0  # bps baseline
    pad_adj = compute_pad_cannibalization(pad_base, crs, config.kappa_cann)

    # ── Forward projections ──
    trajectory = project_saturation(bess_sat, 0.15, config.projection_years)
    sat_5yr = trajectory[min(5, len(trajectory) - 1)]
    sat_10yr = trajectory[min(10, len(trajectory) - 1)]

    # ── RPD growth ──
    rpd_factor = compute_rpd_growth(1.0)  # Normalised growth factor

    # ── Derived features ──
    crs_x_ssi = crs * r_median
    exposure_index = (1.0 - crs) * bess_sat

    return CannibalizationResult(
        substation_id=sid,
        zone=zone,
        bess_sat=round(bess_sat, 4),
        crs=round(crs, 4),
        crs_tier=crs_tier,
        revenue_haircut_pct=round(revenue_haircut_pct, 2),
        spread_compression_pct=round(spread_compression_pct, 2),
        pad_adjustment_bps=round(pad_adj, 2),
        saturation_5yr=round(sat_5yr, 4),
        saturation_10yr=round(sat_10yr, 4),
        rpd_growth_factor=round(rpd_factor, 4),
        crs_x_ssi=round(crs_x_ssi, 4),
        exposure_index=round(exposure_index, 6),
    )


def enrich_fleet(
    substations: list,
    zone_data: Optional[Dict] = None,
    config: CannibalizationConfig = DEFAULT_CONFIG,
    verbose: bool = False,
) -> list:
    """Apply cannibalization enrichment to all substations in the fleet.

    Modifies substations in-place (adds 'cannibalization' key).

    Args:
        substations: List of substation dicts
        zone_data: Optional zone-level data
        config: Cannibalization parameters
        verbose: Print summary statistics

    Returns:
        substations (modified in-place)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 6b: Cannibalization Resilience Enrichment")
        print(f"{'='*60}")

    results = []
    for sub in substations:
        result = enrich_substation(sub, zone_data, config)
        sub['cannibalization'] = {
            'bess_sat': result.bess_sat,
            'crs': result.crs,
            'crs_tier': result.crs_tier,
            'zone': result.zone,
            'revenue_haircut_pct': result.revenue_haircut_pct,
            'spread_compression_pct': result.spread_compression_pct,
            'pad_adjustment_bps': result.pad_adjustment_bps,
            'saturation_5yr': result.saturation_5yr,
            'saturation_10yr': result.saturation_10yr,
            'rpd_growth_factor': result.rpd_growth_factor,
            'crs_x_ssi': result.crs_x_ssi,
            'exposure_index': result.exposure_index,
        }
        results.append(result)

    if verbose:
        crs_vals = [r.crs for r in results]
        sat_vals = [r.bess_sat for r in results]
        haircut_vals = [r.revenue_haircut_pct for r in results]
        tiers = {}
        for r in results:
            tiers[r.crs_tier] = tiers.get(r.crs_tier, 0) + 1

        print(f"  Substations enriched: {len(results):,}")
        print(f"  CRS (median):         {np.median(crs_vals):.3f}")
        print(f"  CRS (range):          [{min(crs_vals):.3f}, {max(crs_vals):.3f}]")
        print(f"  BESS_SAT (median):    {np.median(sat_vals):.4f}")
        print(f"  Revenue haircut:      {np.median(haircut_vals):.2f}% (median)")
        print(f"  CRS Tiers:")
        for tier in ['Resilient', 'Mixed', 'Exposed']:
            count = tiers.get(tier, 0)
            pct = 100 * count / len(results)
            print(f"    {tier:12s} {count:5,} ({pct:5.1f}%)")

    return substations


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════

# Italian region → bidding zone mapping
_REGION_ZONE_MAP = {
    'Piemonte': 'Nord', "Valle d'Aosta": 'Nord', 'Lombardia': 'Nord',
    'Trentino-Alto Adige': 'Nord', 'Veneto': 'Nord',
    'Friuli Venezia Giulia': 'Nord', 'Liguria': 'Nord',
    'Emilia-Romagna': 'Nord',
    'Toscana': 'Centro-Nord', 'Umbria': 'Centro-Nord',
    'Marche': 'Centro-Nord',
    'Lazio': 'Centro-Sud', 'Abruzzo': 'Centro-Sud',
    'Molise': 'Centro-Sud', 'Campania': 'Centro-Sud',
    'Puglia': 'Sud', 'Basilicata': 'Sud', 'Calabria': 'Sud',
    'Sicilia': 'Sicilia',
    'Sardegna': 'Sardegna',
}

# Default zone saturation estimates (2025 baseline from Terna reports)
# These will be overridden by live CN.1 data when available
_DEFAULT_ZONE_SATURATION = {
    'Nord':        0.06,   # ~1,800 MW installed / 30,000 MW peak
    'Centro-Nord': 0.04,   # ~300 MW / 7,500 MW peak
    'Centro-Sud':  0.03,   # ~200 MW / 7,000 MW peak
    'Sud':         0.05,   # ~400 MW / 8,000 MW peak
    'Sicilia':     0.04,   # ~150 MW / 3,500 MW peak
    'Sardegna':    0.03,   # ~80 MW / 2,500 MW peak
}


def _region_to_zone(region: str) -> str:
    """Map Italian region to bidding zone."""
    return _REGION_ZONE_MAP.get(region, 'Centro-Sud')


def _default_zone_saturation(zone: str) -> float:
    """Get default BESS saturation for a zone."""
    return _DEFAULT_ZONE_SATURATION.get(zone, 0.05)
