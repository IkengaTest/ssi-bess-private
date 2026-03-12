"""
SSI-ENN BESS — Monte Carlo Revenue Engine (v1.0)
===================================================
Full stochastic simulation of BESS revenue streams over a 25-year horizon.

Architecture:
  1. ItalianMarketCalibration — Published statistics from GME, Terna, ARERA
  2. HourlyPriceGenerator     — Synthetic Italian zonal electricity prices
  3. BESSRevenueSimulator      — Converts hourly prices to 10 revenue streams
  4. MonteCarloEngine          — Runs N paths, computes NPV distributions
  5. SyntheticDataGenerator    — Fleet-wide MC with per-substation enrichments

Price Model:
  Ornstein-Uhlenbeck jump-diffusion with:
    - Seasonal mean μ(t): annual cycle (winter peak, summer secondary)
    - Intraday pattern h(t): peak (08-20) vs off-peak (20-08)
    - Weekend effect w(t): ~15% lower on Sat/Sun
    - Poisson jump process: λ_jump ≈ 10/yr, magnitude ~2-5× base
    - Mean reversion: κ ≈ 50/yr (half-life ~5 days)
    - Zone differentials: calibrated from GME published zonal data

  dP = κ(μ(t) − P)dt + σ·P·dW + J·dN(λ)

Revenue Streams (10 total):
  R1  Arbitrage         — Daily cycling: spread × MWh × η_RT
  R2  FCR               — Frequency containment: availability × price × MW
  R3  aFRR              — Automatic restoration: activation × price × MW
  R4  mFRR              — Manual restoration: activation × price × MW
  R5  Capacity Market   — Fixed annual payment × MW (ARERA auction)
  R6  Congestion Rent   — Zonal price differential × capacity
  R7  Nodal/LMP Premium — Future scenario, probability-weighted
  R8  Energy Community  — CER framework: self-consumption × local factor
  R9  DSO Services      — Voltage support + congestion management
  R10 PQaaS             — Power quality as a service

Calibration Sources:
  GME:   Average PUN €125/MWh, zonal spreads, peak/off-peak ratios
  Terna: Load profiles, thermoelectric share, BESS installed capacity
  ARERA: FCR €5-8/MW/h, Capacity Market €33k/MW/yr, VOLL €11k/MWh
  ECB:   TTF gas futures, PSV spread
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ssi.monte_carlo")


# ══════════════════════════════════════════════════════════════════════
#  ITALIAN MARKET CALIBRATION (from published GME/Terna/ARERA data)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ItalianMarketCalibration:
    """Published Italian electricity market statistics for MC calibration.

    All values calibrated from 2023-2024 published data (GME statistical
    reports, Terna "Dati Statistici", ARERA annual reports).
    """
    # ── Zonal Base Prices (€/MWh, 2024 averages from GME) ──
    zonal_base_prices: Dict[str, float] = field(default_factory=lambda: {
        'Nord': 115.0,
        'Centro-Nord': 120.0,
        'Centro-Sud': 128.0,
        'Sud': 132.0,
        'Sicilia': 148.0,
        'Sardegna': 142.0,
    })

    # ── Seasonal Price Pattern (monthly multiplier, GME 2023-24 avg) ──
    # Jan=1.15 (winter heating), Aug=1.10 (summer cooling), Apr=0.85 (shoulder)
    monthly_multipliers: List[float] = field(default_factory=lambda: [
        1.15, 1.08, 0.95, 0.85, 0.82, 0.88,   # Jan-Jun
        0.92, 1.10, 1.05, 1.00, 1.08, 1.18,    # Jul-Dec
    ])

    # ── Intraday Pattern (hourly multiplier, normalised to mean=1.0) ──
    # Peak 08:00-20:00 ≈ 1.20×, Off-peak 20:00-08:00 ≈ 0.80×
    # Solar dip 11:00-15:00 (duck curve effect in southern Italy)
    hourly_pattern: List[float] = field(default_factory=lambda: [
        0.72, 0.68, 0.65, 0.63, 0.65, 0.72,   # 00-05 (night trough)
        0.85, 1.00, 1.18, 1.25, 1.28, 1.22,   # 06-11 (morning ramp)
        1.15, 1.10, 1.08, 1.15, 1.25, 1.35,   # 12-17 (solar dip then evening peak)
        1.38, 1.30, 1.15, 1.00, 0.88, 0.78,   # 18-23 (evening decline)
    ])

    # ── Weekend Effect ──
    weekend_discount: float = 0.85       # Sat/Sun prices ≈ 85% of weekday

    # ── Volatility Parameters ──
    annual_volatility: float = 0.35      # σ annual (GME price variance)
    mean_reversion_speed: float = 50.0   # κ (half-life ≈ 5 days)
    jump_frequency: float = 12.0         # λ: ~12 spikes/year (Poisson)
    jump_mean_log: float = 0.50          # Mean log-jump size
    jump_std_log: float = 0.40           # Std of log-jump size

    # ── Zonal Spread Volatility (additional zone-specific vol) ──
    zonal_vol_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'Nord': 0.85,            # Well-meshed, lower vol
        'Centro-Nord': 0.90,
        'Centro-Sud': 1.00,
        'Sud': 1.10,
        'Sicilia': 1.35,        # Island, high vol
        'Sardegna': 1.30,       # Island, constrained
    })

    # ── BESS Revenue Stream Parameters ──
    # R2: FCR (Frequency Containment Reserve)
    fcr_price_mean: float = 6.5          # €/MW/h (ARERA/Terna 2024)
    fcr_price_std: float = 2.0           # Monthly variation
    fcr_availability: float = 0.92       # Average availability requirement

    # R3: aFRR (Automatic Frequency Restoration Reserve)
    afrr_activation_rate: float = 0.40   # Fraction of hours with activation
    afrr_premium_mean: float = 30.0      # €/MWh above spot (when activated)
    afrr_premium_std: float = 12.0

    # R4: mFRR (Manual Frequency Restoration Reserve)
    mfrr_activation_rate: float = 0.15   # Less frequent
    mfrr_premium_mean: float = 45.0      # Higher premium
    mfrr_premium_std: float = 18.0

    # R5: Capacity Market
    cm_price_mean: float = 33000.0       # €/MW/year (ARERA 2024 auction)
    cm_price_std: float = 5000.0         # Auction-to-auction variation
    cm_regulatory_shock_prob: float = 0.08  # Prob of major rule change/year

    # R6-R10: See revenue simulator for details
    congestion_rent_fraction: float = 0.15  # Fraction of zonal spread captured
    nodal_transition_annual_prob: float = 0.06
    cer_base_revenue: float = 8000.0     # €/MW/year for energy communities
    dso_service_base: float = 5000.0     # €/MW/year for DSO services
    pqaas_base: float = 2000.0          # €/MW/year for power quality

    # ── Battery Physics ──
    round_trip_efficiency: float = 0.87  # Li-ion round-trip
    degradation_rate: float = 0.008      # 0.8% annual capacity loss
    depth_of_discharge: float = 0.90     # Usable fraction of capacity
    max_daily_cycles: float = 1.5        # Average daily full-equivalent cycles
    calendar_life_years: float = 25.0    # Warranty/design life

    # ── Financial ──
    wacc: float = 0.052
    inflation_rate: float = 0.02         # Long-run Italian CPI
    opex_pct_capex: float = 0.015        # Annual OPEX as % of CAPEX

    # ── Trend Parameters (annual drift in revenue streams) ──
    arbitrage_spread_trend: float = -0.01   # Slight compression over time (more BESS)
    fcr_price_trend: float = -0.005         # Slight decline (more supply)
    cm_price_trend: float = 0.02            # Growing (decarbonisation needs)
    cer_growth_rate: float = 0.08           # Energy communities growing fast
    dso_growth_rate: float = 0.10           # DSO services emerging
    pqaas_growth_rate: float = 0.05         # PQaaS slowly growing


DEFAULT_CALIBRATION = ItalianMarketCalibration()


# ══════════════════════════════════════════════════════════════════════
#  HOURLY PRICE GENERATOR
# ══════════════════════════════════════════════════════════════════════

class HourlyPriceGenerator:
    """Generate synthetic Italian zonal electricity prices.

    Uses mean-reverting jump-diffusion (Ornstein-Uhlenbeck + Poisson jumps)
    calibrated from published GME statistics.

    Process: dP = κ(μ(t,h) − P)dt + σ_zone·P·dW + J·dN(λ)

    where:
        μ(t,h) = P_base_zone × seasonal(month) × intraday(hour) × weekend(dow)
        σ_zone = σ_annual × zone_vol_multiplier / √8760
        J ~ LogNormal(μ_jump, σ_jump)  (positive spikes only)
        N(λ) = Poisson process with rate λ jumps/year
    """

    def __init__(self, cal: ItalianMarketCalibration = DEFAULT_CALIBRATION):
        self.cal = cal
        self._hourly_vol = cal.annual_volatility / math.sqrt(8760)
        self._jump_prob_hourly = cal.jump_frequency / 8760

    def _precompute_patterns(self) -> np.ndarray:
        """Pre-compute the 8,760-element deterministic pattern array.

        Returns:
            Array of shape (8760,) with seasonal × intraday × weekend multipliers.
        """
        cal = self.cal
        pattern = np.zeros(8760)
        for h in range(8760):
            day_of_year = h // 24
            month = min(11, day_of_year * 12 // 365)
            hour = h % 24
            dow = day_of_year % 7
            seasonal = cal.monthly_multipliers[month]
            intraday = cal.hourly_pattern[hour]
            weekend = cal.weekend_discount if dow >= 5 else 1.0
            pattern[h] = seasonal * intraday * weekend
        return pattern

    def generate_paths(
        self,
        zone: str,
        n_paths: int = 1000,
        n_years: int = 25,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """Generate n_paths × n_years of hourly prices (VECTORIZED).

        Processes all paths simultaneously using numpy broadcasting.
        Uses exact OU discrete transition:
            P(t+1) = μ(t) + (P(t) - μ(t)) × exp(-κ×dt) + noise

        Args:
            zone: Italian market zone name
            n_paths: Number of Monte Carlo paths
            n_years: Number of years per path
            rng: NumPy random generator

        Returns:
            Array of shape (n_paths, n_years, 8760)
        """
        if rng is None:
            rng = np.random.default_rng()

        cal = self.cal
        p_base = cal.zonal_base_prices.get(zone, 125.0)
        vol_mult = cal.zonal_vol_multipliers.get(zone, 1.0)

        # Pre-compute pattern for one year
        pattern = self._precompute_patterns()  # (8760,)

        # OU parameters in hourly terms
        kappa_h = cal.mean_reversion_speed / 8760
        decay = math.exp(-kappa_h)
        sigma_h = cal.annual_volatility * vol_mult / math.sqrt(8760)
        # Conditional variance for exact OU: σ²_cond = σ²(1 - e^(-2κdt)) / (2κ)
        sigma_cond = sigma_h * math.sqrt((1.0 - decay**2) / max(1e-9, 2.0 * kappa_h))

        total_hours = n_years * 8760
        result = np.zeros((n_paths, n_years, 8760))

        # Generate all noise at once: (n_paths, total_hours)
        noise = rng.standard_normal((n_paths, total_hours))

        # Jump process: Bernoulli for each hour
        jump_mask = rng.random((n_paths, total_hours)) < self._jump_prob_hourly
        jump_sizes = np.exp(rng.normal(cal.jump_mean_log, cal.jump_std_log,
                                        (n_paths, total_hours)))

        # Simulate path by path through hours (vectorised across n_paths)
        p = np.full(n_paths, p_base)  # (n_paths,)

        for yr in range(n_years):
            trend = max(0.5, 1.0 + cal.arbitrage_spread_trend * yr)
            yr_offset = yr * 8760

            for h in range(8760):
                idx = yr_offset + h
                mu = p_base * pattern[h] * trend  # Scalar target

                # Exact OU step: P → μ + (P - μ) × decay + σ_cond × Z
                p = mu + (p - mu) * decay + sigma_cond * p * noise[:, idx]

                # Jumps (vectorised)
                jump_h = jump_mask[:, idx]
                p = np.where(jump_h, p * jump_sizes[:, idx], p)

                # Floor
                p = np.maximum(p, 5.0)
                result[:, yr, h] = p

        return result


# ══════════════════════════════════════════════════════════════════════
#  BESS REVENUE SIMULATOR
# ══════════════════════════════════════════════════════════════════════

class BESSRevenueSimulator:
    """Convert hourly price paths to 10 BESS revenue streams.

    Takes hourly electricity prices and computes annual revenue for each
    of the 10 revenue streams, accounting for:
    - Battery capacity and degradation
    - Round-trip efficiency losses
    - Depth of discharge limits
    - Market participation constraints
    - Revenue stream correlations (embedded in price dynamics)
    """

    def __init__(
        self,
        mw: float,
        mwh: float,
        capex_m: float,
        cal: ItalianMarketCalibration = DEFAULT_CALIBRATION,
    ):
        self.mw = mw
        self.mwh = mwh
        self.capex_m = capex_m
        self.cal = cal
        self.usable_mwh = mwh * cal.depth_of_discharge

    def compute_annual_revenues(
        self,
        hourly_prices: np.ndarray,
        year_offset: int = 0,
        rng: np.random.Generator = None,
        crs: float = 0.50,
        bess_sat: float = 0.05,
        zone: str = 'Centro-Sud',
    ) -> Dict[str, float]:
        """Compute annual revenue for all 10 streams from hourly prices.

        Args:
            hourly_prices: Array of 8,760 hourly prices (€/MWh)
            year_offset: Years from commissioning (for degradation)
            rng: Random generator for stochastic elements
            crs: Cannibalization Resilience Score [0, 1]
            bess_sat: BESS saturation ratio in zone
            zone: Market zone name

        Returns:
            Dict of {stream_name: annual_revenue_euros}
        """
        if rng is None:
            rng = np.random.default_rng()

        cal = self.cal
        # Degradation: capacity fades linearly
        capacity_factor = (1.0 - cal.degradation_rate) ** year_offset
        eff_mwh = self.usable_mwh * capacity_factor
        eff_mw = self.mw * capacity_factor

        # ── R1: ARBITRAGE ──
        r1 = self._compute_arbitrage(hourly_prices, eff_mwh, year_offset, crs, bess_sat)

        # ── R2: FCR ──
        r2 = self._compute_fcr(eff_mw, year_offset, rng)

        # ── R3: aFRR ──
        r3 = self._compute_afrr(hourly_prices, eff_mw, rng)

        # ── R4: mFRR ──
        r4 = self._compute_mfrr(hourly_prices, eff_mw, rng)

        # ── R5: Capacity Market ──
        r5 = self._compute_capacity_market(eff_mw, year_offset, rng)

        # ── R6: Congestion Rent ──
        r6 = self._compute_congestion(hourly_prices, eff_mwh, zone)

        # ── R7: Nodal/LMP Premium ──
        r7 = self._compute_nodal_premium(hourly_prices, eff_mw, year_offset, rng)

        # ── R8: Energy Community ──
        r8 = self._compute_energy_community(eff_mw, year_offset, rng)

        # ── R9: DSO Services ──
        r9 = self._compute_dso_services(eff_mw, year_offset, rng)

        # ── R10: PQaaS ──
        r10 = self._compute_pqaas(eff_mw, year_offset, rng)

        # ── OPEX deduction ──
        opex = self.capex_m * 1e6 * cal.opex_pct_capex * (1 + cal.inflation_rate) ** year_offset

        total_gross = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10
        total_net = total_gross - opex

        return {
            'R1_arbitrage': round(r1, 2),
            'R2_fcr': round(r2, 2),
            'R3_afrr': round(r3, 2),
            'R4_mfrr': round(r4, 2),
            'R5_capacity_market': round(r5, 2),
            'R6_congestion': round(r6, 2),
            'R7_nodal_lmp': round(r7, 2),
            'R8_energy_community': round(r8, 2),
            'R9_dso_services': round(r9, 2),
            'R10_pqaas': round(r10, 2),
            'opex': round(opex, 2),
            'total_gross': round(total_gross, 2),
            'total_net': round(total_net, 2),
            'capacity_factor': round(capacity_factor, 4),
        }

    def _compute_arbitrage(
        self, prices: np.ndarray, eff_mwh: float,
        year_offset: int, crs: float, bess_sat: float,
    ) -> float:
        """R1: Daily arbitrage from peak/off-peak cycling.

        Strategy: Charge during lowest-price 4h window, discharge during
        highest-price 4h window each day. Revenue = spread × MWh × η_RT.
        Apply cannibalization haircut based on CRS and saturation.
        """
        cal = self.cal
        daily_revenue = 0.0
        n_days = len(prices) // 24

        for d in range(n_days):
            day_prices = prices[d * 24:(d + 1) * 24]
            if len(day_prices) < 24:
                continue

            # Find optimal charge/discharge windows (4h each)
            sorted_idx = np.argsort(day_prices)
            charge_hours = sorted_idx[:4]
            discharge_hours = sorted_idx[-4:]

            avg_charge = np.mean(day_prices[charge_hours])
            avg_discharge = np.mean(day_prices[discharge_hours])
            spread = max(0, avg_discharge - avg_charge)

            # Revenue for one cycle
            cycle_revenue = spread * eff_mwh * cal.round_trip_efficiency
            daily_revenue += cycle_revenue

        # Cannibalization haircut: more BESS → compressed spreads
        # Haircut = (1 - λ_R1) × δ × f(BESS_SAT) where λ_R1 = 0.05 (low locality)
        lambda_r1 = 0.05
        delta = 0.60
        gamma = 2.0
        sat_effect = bess_sat ** gamma if bess_sat < 1.0 else 1.0
        haircut = (1 - lambda_r1) * delta * sat_effect
        crs_adjustment = 1.0 - haircut * (1.0 - crs)

        # Trend: slight spread compression over time
        trend = max(0.5, 1.0 + cal.arbitrage_spread_trend * year_offset)

        return daily_revenue * crs_adjustment * trend

    def _compute_fcr(self, eff_mw: float, year_offset: int, rng) -> float:
        """R2: Frequency Containment Reserve.

        FCR_annual = MW × hours_available × FCR_price × availability
        Italian BESS FCR: ~€5-8/MW/h, available ~8,000 h/year.
        """
        cal = self.cal
        fcr_price = max(1.0, rng.normal(cal.fcr_price_mean, cal.fcr_price_std))
        hours_available = 8000 * cal.fcr_availability
        trend = max(0.5, 1.0 + cal.fcr_price_trend * year_offset)
        return eff_mw * hours_available * fcr_price * trend

    def _compute_afrr(self, prices: np.ndarray, eff_mw: float, rng) -> float:
        """R3: Automatic Frequency Restoration Reserve.

        Revenue from activated aFRR: prob(activation) × premium × MW × hours.
        Correlated with price volatility (more activation when prices volatile).
        """
        cal = self.cal
        # Higher price volatility → more aFRR activation
        price_vol = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0.3
        vol_factor = min(2.0, price_vol / 0.30)  # Normalise to typical vol

        activation_rate = cal.afrr_activation_rate * vol_factor
        premium = max(5.0, rng.normal(cal.afrr_premium_mean, cal.afrr_premium_std))
        hours = 8760 * min(1.0, activation_rate)

        return eff_mw * hours * premium * 0.5  # 50% energy delivery fraction

    def _compute_mfrr(self, prices: np.ndarray, eff_mw: float, rng) -> float:
        """R4: Manual Frequency Restoration Reserve.

        Less frequent than aFRR but higher premium.
        """
        cal = self.cal
        price_vol = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0.3
        vol_factor = min(2.0, price_vol / 0.30)

        activation_rate = cal.mfrr_activation_rate * vol_factor
        premium = max(10.0, rng.normal(cal.mfrr_premium_mean, cal.mfrr_premium_std))
        hours = 8760 * min(1.0, activation_rate)

        return eff_mw * hours * premium * 0.3  # 30% energy delivery fraction

    def _compute_capacity_market(self, eff_mw: float, year_offset: int, rng) -> float:
        """R5: Capacity Market payment.

        Fixed annual €/MW from ARERA capacity market auction.
        Subject to regulatory risk (BS-F2-1 type shock).
        Growing trend as decarbonisation increases flexibility needs.
        """
        cal = self.cal
        base_price = max(10000, rng.normal(cal.cm_price_mean, cal.cm_price_std))

        # Regulatory shock: small probability of major rule change
        if rng.random() < cal.cm_regulatory_shock_prob:
            shock_severity = rng.uniform(0.5, 1.5)  # Can go up or down
            base_price *= shock_severity

        trend = 1.0 + cal.cm_price_trend * year_offset
        return eff_mw * base_price * trend

    def _compute_congestion(self, prices: np.ndarray, eff_mwh: float, zone: str) -> float:
        """R6: Congestion rent from zonal price differentials.

        BESS in congested zones can profit from price differentials.
        Revenue = fraction_captured × (P_zone - P_pun) × MWh × hours.
        """
        cal = self.cal
        p_zone_avg = np.mean(prices)
        # PUN is roughly the demand-weighted average
        p_pun = cal.zonal_base_prices.get('Nord', 115.0)  # Nord dominates PUN
        diff = max(0, p_zone_avg - p_pun)

        # Hours of positive differential
        n_positive_hours = np.sum(prices > p_pun * 1.05)
        fraction = n_positive_hours / 8760

        return diff * eff_mwh * fraction * 365 * cal.congestion_rent_fraction

    def _compute_nodal_premium(
        self, prices: np.ndarray, eff_mw: float, year_offset: int, rng,
    ) -> float:
        """R7: Nodal/LMP premium (future scenario).

        Probability-weighted: P(nodal_transition) grows over time.
        If nodal regime activated, premium = congestion_factor × base.
        """
        cal = self.cal
        # Cumulative probability of nodal transition by this year
        p_nodal = 1.0 - (1.0 - cal.nodal_transition_annual_prob) ** (year_offset + 1)

        # If nodal regime exists, premium is ~€10-25/MW/h for congested locations
        nodal_premium = rng.uniform(10, 25) if rng.random() < p_nodal else 0.0
        hours = 4000  # Estimated hours of congestion benefit

        return eff_mw * nodal_premium * hours * p_nodal

    def _compute_energy_community(self, eff_mw: float, year_offset: int, rng) -> float:
        """R8: Energy Community (CER) revenue.

        Italian CER framework: premium for local self-consumption.
        Growing rapidly from low base as regulatory framework matures.
        """
        cal = self.cal
        base = cal.cer_base_revenue * (1 + cal.cer_growth_rate) ** year_offset
        noise = rng.normal(1.0, 0.15)
        return eff_mw * base * max(0.5, noise)

    def _compute_dso_services(self, eff_mw: float, year_offset: int, rng) -> float:
        """R9: DSO voltage support and congestion management.

        Emerging market. Small today, growing as DSOs adopt flexibility platforms.
        """
        cal = self.cal
        base = cal.dso_service_base * (1 + cal.dso_growth_rate) ** year_offset
        noise = rng.normal(1.0, 0.20)
        return eff_mw * base * max(0.3, noise)

    def _compute_pqaas(self, eff_mw: float, year_offset: int, rng) -> float:
        """R10: Power Quality as a Service.

        Niche but stable: harmonic filtering, reactive power, voltage regulation.
        """
        cal = self.cal
        base = cal.pqaas_base * (1 + cal.pqaas_growth_rate) ** year_offset
        noise = rng.normal(1.0, 0.10)
        return eff_mw * base * max(0.5, noise)


# ══════════════════════════════════════════════════════════════════════
#  MONTE CARLO ENGINE
# ══════════════════════════════════════════════════════════════════════

class MonteCarloEngine:
    """Run Monte Carlo simulation for BESS project valuation (VECTORIZED).

    Generates N paths of 25-year revenue trajectories simultaneously,
    using numpy broadcasting across all paths. Computes:
    - NPV distribution (P5, P25, P50, P75, P95)
    - IRR distribution
    - Revenue stream decomposition (which streams dominate)
    - Risk metrics (VaR, CVaR/TVaR, Sharpe-like ratio)
    """

    def __init__(
        self,
        n_paths: int = 1000,
        horizon: int = 25,
        cal: ItalianMarketCalibration = DEFAULT_CALIBRATION,
    ):
        self.n_paths = n_paths
        self.horizon = horizon
        self.cal = cal
        self.price_gen = HourlyPriceGenerator(cal)
        # Pre-compute discount factors
        self._disc = np.array([1.0 / (1.0 + cal.wacc) ** (y + 1) for y in range(horizon)])

    def run_substation(
        self,
        substation: dict,
        config_name: str = 'B',
        seed: int = None,
        verbose: bool = False,
    ) -> Dict:
        """Run full MC simulation for one substation (VECTORIZED).

        Generates all price paths at once, then computes revenues
        vectorized across paths for each year.

        Args:
            substation: Substation dict with enrichments
            config_name: BESS config ('A' or 'B')
            seed: Random seed for reproducibility
            verbose: Print progress

        Returns:
            Dict with NPV distribution, revenue decomposition, risk metrics
        """
        from .config import BESS_CONFIGS

        config = BESS_CONFIGS[config_name]
        mw = config['mw']
        mwh = config['mwh']
        capex_m = config['capex_m']

        # Extract substation enrichment data
        region = substation.get('region', '')
        from .black_swan import _REGION_TO_ZONE_THERMAL
        zone = _REGION_TO_ZONE_THERMAL.get(region, 'Centro-Sud')

        cann = substation.get('cannibalization', {})
        crs = cann.get('crs', 0.50)
        bess_sat = cann.get('bess_sat', 0.05)

        cal = self.cal
        rng = np.random.default_rng(seed)
        n = self.n_paths
        H = self.horizon

        # ── STEP 1: Generate all price paths at once ──
        # Shape: (n_paths, horizon, 8760)
        all_prices = self.price_gen.generate_paths(zone, n, H, rng)

        # ── STEP 2: Compute revenues vectorized ──
        # Revenue arrays: (n_paths, horizon) per stream
        R = {i: np.zeros((n, H)) for i in range(1, 11)}

        usable_mwh = mwh * cal.depth_of_discharge

        for yr in range(H):
            cap_factor = (1.0 - cal.degradation_rate) ** yr
            eff_mwh = usable_mwh * cap_factor
            eff_mw = mw * cap_factor

            year_prices = all_prices[:, yr, :]  # (n_paths, 8760)

            # ── R1: Arbitrage (vectorized daily spread) ──
            daily = year_prices.reshape(n, 365, 24)  # (n, 365, 24)
            # Sort each day's prices
            daily_sorted = np.sort(daily, axis=2)  # sorted ascending
            avg_charge = np.mean(daily_sorted[:, :, :4], axis=2)    # cheapest 4h
            avg_discharge = np.mean(daily_sorted[:, :, -4:], axis=2)  # dearest 4h
            daily_spread = np.maximum(0, avg_discharge - avg_charge)   # (n, 365)
            annual_arb = np.sum(daily_spread, axis=1) * eff_mwh * cal.round_trip_efficiency

            # Cannibalization haircut
            sat_effect = bess_sat ** 2.0 if bess_sat < 1.0 else 1.0
            haircut = (1 - 0.05) * 0.60 * sat_effect
            crs_adj = 1.0 - haircut * (1.0 - crs)
            trend = max(0.5, 1.0 + cal.arbitrage_spread_trend * yr)
            R[1][:, yr] = annual_arb * crs_adj * trend

            # ── R2: FCR (vectorized across paths) ──
            fcr_prices = np.maximum(1.0, rng.normal(cal.fcr_price_mean, cal.fcr_price_std, n))
            fcr_trend = max(0.5, 1.0 + cal.fcr_price_trend * yr)
            R[2][:, yr] = eff_mw * 8000 * cal.fcr_availability * fcr_prices * fcr_trend

            # ── R3: aFRR (correlated with price vol) ──
            price_means = np.mean(year_prices, axis=1)  # (n,)
            price_stds = np.std(year_prices, axis=1)
            price_vol = price_stds / np.maximum(price_means, 1.0)
            vol_factor = np.minimum(2.0, price_vol / 0.30)
            afrr_act = cal.afrr_activation_rate * vol_factor
            afrr_prem = np.maximum(5.0, rng.normal(cal.afrr_premium_mean, cal.afrr_premium_std, n))
            R[3][:, yr] = eff_mw * 8760 * np.minimum(1.0, afrr_act) * afrr_prem * 0.5

            # ── R4: mFRR ──
            mfrr_act = cal.mfrr_activation_rate * vol_factor
            mfrr_prem = np.maximum(10.0, rng.normal(cal.mfrr_premium_mean, cal.mfrr_premium_std, n))
            R[4][:, yr] = eff_mw * 8760 * np.minimum(1.0, mfrr_act) * mfrr_prem * 0.3

            # ── R5: Capacity Market ──
            cm_prices = np.maximum(10000, rng.normal(cal.cm_price_mean, cal.cm_price_std, n))
            shock_mask = rng.random(n) < cal.cm_regulatory_shock_prob
            shock_severity = rng.uniform(0.5, 1.5, n)
            cm_prices = np.where(shock_mask, cm_prices * shock_severity, cm_prices)
            cm_trend = 1.0 + cal.cm_price_trend * yr
            R[5][:, yr] = eff_mw * cm_prices * cm_trend

            # ── R6: Congestion ──
            p_pun = cal.zonal_base_prices.get('Nord', 115.0)
            diff = np.maximum(0, price_means - p_pun)
            pos_hours = np.sum(year_prices > p_pun * 1.05, axis=1) / 8760
            R[6][:, yr] = diff * eff_mwh * pos_hours * 365 * cal.congestion_rent_fraction

            # ── R7: Nodal/LMP Premium ──
            p_nodal = 1.0 - (1.0 - cal.nodal_transition_annual_prob) ** (yr + 1)
            nodal_draw = rng.random(n)
            nodal_prem = np.where(nodal_draw < p_nodal, rng.uniform(10, 25, n), 0.0)
            R[7][:, yr] = eff_mw * nodal_prem * 4000 * p_nodal

            # ── R8: Energy Community ──
            cer_base = cal.cer_base_revenue * (1 + cal.cer_growth_rate) ** yr
            cer_noise = np.maximum(0.5, rng.normal(1.0, 0.15, n))
            R[8][:, yr] = eff_mw * cer_base * cer_noise

            # ── R9: DSO Services ──
            dso_base = cal.dso_service_base * (1 + cal.dso_growth_rate) ** yr
            dso_noise = np.maximum(0.3, rng.normal(1.0, 0.20, n))
            R[9][:, yr] = eff_mw * dso_base * dso_noise

            # ── R10: PQaaS ──
            pq_base = cal.pqaas_base * (1 + cal.pqaas_growth_rate) ** yr
            pq_noise = np.maximum(0.5, rng.normal(1.0, 0.10, n))
            R[10][:, yr] = eff_mw * pq_base * pq_noise

        # ── STEP 3: Aggregate cashflows ──
        # OPEX per year: (horizon,)
        opex = np.array([capex_m * 1e6 * cal.opex_pct_capex * (1 + cal.inflation_rate) ** yr
                         for yr in range(H)])

        # Total gross revenue per path per year: (n, H)
        gross = sum(R[i] for i in range(1, 11))
        net = gross - opex[np.newaxis, :]  # (n, H)

        # NPV: -CAPEX + Σ(net_yr × discount_yr)
        npv_array = -capex_m + np.sum(net * self._disc[np.newaxis, :], axis=1) / 1e6  # €M

        # IRR (vectorized binary search — batch all paths)
        irr_array = self._compute_irr_batch(net, capex_m * 1e6)

        # Stream totals across years: (n,) per stream
        stream_totals = {f'R{i}': np.sum(R[i], axis=1) for i in range(1, 11)}

        # ── Aggregate Results ──
        result = self._aggregate_results(npv_array, irr_array, stream_totals, capex_m, mw)
        result['zone'] = zone
        result['config'] = config_name
        result['crs'] = crs
        result['bess_sat'] = bess_sat
        result['n_paths'] = self.n_paths

        return result

    def _compute_irr_batch(self, cashflows: np.ndarray, capex: float) -> np.ndarray:
        """Vectorized IRR via binary search across all paths.

        Args:
            cashflows: (n_paths, horizon) annual net cashflows
            capex: Initial investment (scalar)

        Returns:
            (n_paths,) array of IRR percentages
        """
        n = cashflows.shape[0]
        H = cashflows.shape[1]
        low = np.full(n, -0.10)
        high = np.full(n, 0.50)

        years = np.arange(1, H + 1)  # (H,)

        for _ in range(40):
            mid = (low + high) / 2  # (n,)
            # PV = Σ CF_t / (1+r)^t for each path
            disc = 1.0 / (1.0 + mid[:, np.newaxis]) ** years[np.newaxis, :]  # (n, H)
            pv = np.sum(cashflows * disc, axis=1)  # (n,)
            mask = pv > capex
            low = np.where(mask, mid, low)
            high = np.where(mask, high, mid)

        return (low + high) / 2 * 100  # Percentage

    def _aggregate_results(
        self,
        npv_array: np.ndarray,
        irr_array: np.ndarray,
        stream_totals: dict,
        capex_m: float,
        mw: float,
    ) -> Dict:
        """Compute distribution statistics from MC paths."""
        npv_stats = {
            'npv_mean': round(float(np.mean(npv_array)), 4),
            'npv_median': round(float(np.median(npv_array)), 4),
            'npv_std': round(float(np.std(npv_array)), 4),
            'npv_P5': round(float(np.percentile(npv_array, 5)), 4),
            'npv_P25': round(float(np.percentile(npv_array, 25)), 4),
            'npv_P50': round(float(np.percentile(npv_array, 50)), 4),
            'npv_P75': round(float(np.percentile(npv_array, 75)), 4),
            'npv_P95': round(float(np.percentile(npv_array, 95)), 4),
            'npv_positive_pct': round(float(np.mean(npv_array > 0) * 100), 1),
        }

        irr_stats = {
            'irr_mean': round(float(np.mean(irr_array)), 2),
            'irr_median': round(float(np.median(irr_array)), 2),
            'irr_P5': round(float(np.percentile(irr_array, 5)), 2),
            'irr_P95': round(float(np.percentile(irr_array, 95)), 2),
        }

        var_95 = float(np.percentile(npv_array, 5))
        tail_mask = npv_array <= var_95
        tvar_95 = float(np.mean(npv_array[tail_mask])) if np.any(tail_mask) else var_95
        sharpe = float(np.mean(npv_array)) / max(0.01, float(np.std(npv_array)))

        risk_stats = {
            'var_95_M': round(var_95, 4),
            'tvar_95_M': round(tvar_95, 4),
            'sharpe_ratio': round(sharpe, 3),
            'max_loss_M': round(float(np.min(npv_array)), 4),
            'max_gain_M': round(float(np.max(npv_array)), 4),
        }

        total_all = sum(float(np.mean(v)) for v in stream_totals.values())
        stream_pcts = {}
        stream_means = {}
        for name, arr in stream_totals.items():
            mean_val = float(np.mean(arr))
            stream_means[f'{name}_total_mean'] = round(mean_val / 1e6, 4)
            stream_pcts[f'{name}_pct'] = round(mean_val / max(1, total_all) * 100, 1)

        annual_rev_per_mw = total_all / max(1, mw) / 25 / 1000
        capex_per_mw = capex_m / mw * 1000

        # Median annual net revenue across paths
        gross_by_path = sum(stream_totals.values())
        median_annual_net = float(np.median(gross_by_path)) / 25 / 1e6

        economics = {
            'annual_revenue_k_per_mw': round(annual_rev_per_mw, 1),
            'capex_k_per_mw': round(capex_per_mw, 0),
            'payback_median_yr': round(capex_m / max(0.01, median_annual_net), 1),
        }

        return {
            'npv': npv_stats,
            'irr': irr_stats,
            'risk': risk_stats,
            'streams': {**stream_means, **stream_pcts},
            'economics': economics,
        }


# ══════════════════════════════════════════════════════════════════════
#  FLEET-LEVEL SIMULATION
# ══════════════════════════════════════════════════════════════════════

def run_fleet_mc(
    substations: list,
    n_paths: int = 1000,
    config_name: str = 'B',
    sample_size: int = None,
    verbose: bool = True,
) -> list:
    """Run Monte Carlo simulation across fleet with zone-level price caching.

    OPTIMIZATION: Generates price paths once per zone (6 zones), then
    reuses them for all substations in that zone. Per-substation variation
    comes from CRS, BESS_SAT, and enrichment adjustments on the revenue
    calculations — not from regenerating prices.

    Args:
        substations: List of enriched substation dicts
        n_paths: Number of MC paths per substation
        config_name: BESS config ('A' or 'B')
        sample_size: If set, randomly sample this many substations
        verbose: Print progress

    Returns:
        List of substations with 'mc_results' key added
    """
    import time
    from .config import BESS_CONFIGS
    from .black_swan import _REGION_TO_ZONE_THERMAL

    if verbose:
        print(f"\n{'='*60}")
        print(f"MONTE CARLO REVENUE SIMULATION")
        print(f"{'='*60}")
        print(f"  Paths:       {n_paths:,}")
        print(f"  Horizon:     25 years")
        print(f"  Config:      {config_name}")
        print(f"  Fleet size:  {len(substations):,}")

    cal = DEFAULT_CALIBRATION
    config = BESS_CONFIGS[config_name]
    mw, mwh, capex_m = config['mw'], config['mwh'], config['capex_m']
    horizon = 25

    # Sample if requested
    if sample_size and sample_size < len(substations):
        rng_sample = np.random.default_rng(42)
        indices = rng_sample.choice(len(substations), size=sample_size, replace=False)
        targets = [(i, substations[i]) for i in indices]
        if verbose:
            print(f"  Sample:      {sample_size:,} substations")
    else:
        targets = list(enumerate(substations))

    # ── STEP 1: Group substations by zone ──
    zone_groups = {}
    for idx, (i, sub) in enumerate(targets):
        region = sub.get('region', '')
        zone = _REGION_TO_ZONE_THERMAL.get(region, 'Centro-Sud')
        zone_groups.setdefault(zone, []).append((idx, i, sub))

    if verbose:
        for z, grp in sorted(zone_groups.items()):
            print(f"  Zone {z:15s}: {len(grp):,} substations")

    # ── STEP 2: Generate prices per zone & compute revenues ──
    price_gen = HourlyPriceGenerator(cal)
    disc = np.array([1.0 / (1.0 + cal.wacc) ** (y + 1) for y in range(horizon)])

    t0 = time.time()
    results = []
    processed = 0

    for zone, group in zone_groups.items():
        # Generate zone-level prices ONCE
        rng_zone = np.random.default_rng(hash(zone) % 2**32)
        if verbose:
            print(f"\n  Generating {n_paths:,} price paths for {zone}...")

        zone_prices = price_gen.generate_paths(zone, n_paths, horizon, rng_zone)
        # zone_prices: (n_paths, horizon, 8760)

        if verbose:
            t_gen = time.time() - t0
            print(f"    Price generation: {t_gen:.1f}s")

        # Pre-compute zone-level price statistics for R3/R4/R6
        # (shared across substations in the zone)
        n = n_paths
        usable_mwh = mwh * cal.depth_of_discharge

        # Pre-compute daily sorted prices for arbitrage: (n, horizon, 365, 24)
        daily_all = zone_prices.reshape(n, horizon, 365, 24)
        daily_sorted_all = np.sort(daily_all, axis=3)
        avg_charge_all = np.mean(daily_sorted_all[:, :, :, :4], axis=3)
        avg_discharge_all = np.mean(daily_sorted_all[:, :, :, -4:], axis=3)
        daily_spread_all = np.maximum(0, avg_discharge_all - avg_charge_all)
        annual_spread_sum = np.sum(daily_spread_all, axis=2)  # (n, horizon)

        # Price stats per year
        price_means = np.mean(zone_prices, axis=2)  # (n, horizon)
        price_stds = np.std(zone_prices, axis=2)
        price_vol = price_stds / np.maximum(price_means, 1.0)
        vol_factor = np.minimum(2.0, price_vol / 0.30)

        p_pun = cal.zonal_base_prices.get('Nord', 115.0)
        pos_hours = np.sum(zone_prices > p_pun * 1.05, axis=2) / 8760  # (n, horizon)

        # Per-substation revenue computation (fast — no price regeneration)
        for idx_in_group, (target_idx, i, sub) in enumerate(group):
            cann = sub.get('cannibalization', {})
            crs = cann.get('crs', 0.50)
            bess_sat = cann.get('bess_sat', 0.05)

            # Per-substation RNG for stochastic revenue components
            sid = sub.get('substation_id', sub.get('name', str(i)))
            rng_sub = np.random.default_rng(hash(sid) % 2**32)

            R = {ri: np.zeros((n, horizon)) for ri in range(1, 11)}

            # Cannibalization haircut (substation-specific)
            sat_effect = bess_sat ** 2.0 if bess_sat < 1.0 else 1.0
            haircut = (1 - 0.05) * 0.60 * sat_effect
            crs_adj = 1.0 - haircut * (1.0 - crs)

            for yr in range(horizon):
                cap_factor = (1.0 - cal.degradation_rate) ** yr
                eff_mwh = usable_mwh * cap_factor
                eff_mw = mw * cap_factor

                # R1: Arbitrage (from pre-computed spreads)
                trend = max(0.5, 1.0 + cal.arbitrage_spread_trend * yr)
                R[1][:, yr] = annual_spread_sum[:, yr] * eff_mwh * cal.round_trip_efficiency * crs_adj * trend

                # R2: FCR
                fcr_p = np.maximum(1.0, rng_sub.normal(cal.fcr_price_mean, cal.fcr_price_std, n))
                fcr_t = max(0.5, 1.0 + cal.fcr_price_trend * yr)
                R[2][:, yr] = eff_mw * 8000 * cal.fcr_availability * fcr_p * fcr_t

                # R3: aFRR
                afrr_act = cal.afrr_activation_rate * vol_factor[:, yr]
                afrr_prem = np.maximum(5.0, rng_sub.normal(cal.afrr_premium_mean, cal.afrr_premium_std, n))
                R[3][:, yr] = eff_mw * 8760 * np.minimum(1.0, afrr_act) * afrr_prem * 0.5

                # R4: mFRR
                mfrr_act = cal.mfrr_activation_rate * vol_factor[:, yr]
                mfrr_prem = np.maximum(10.0, rng_sub.normal(cal.mfrr_premium_mean, cal.mfrr_premium_std, n))
                R[4][:, yr] = eff_mw * 8760 * np.minimum(1.0, mfrr_act) * mfrr_prem * 0.3

                # R5: Capacity Market
                cm_p = np.maximum(10000, rng_sub.normal(cal.cm_price_mean, cal.cm_price_std, n))
                shock_mask = rng_sub.random(n) < cal.cm_regulatory_shock_prob
                cm_p = np.where(shock_mask, cm_p * rng_sub.uniform(0.5, 1.5, n), cm_p)
                R[5][:, yr] = eff_mw * cm_p * (1.0 + cal.cm_price_trend * yr)

                # R6: Congestion
                diff = np.maximum(0, price_means[:, yr] - p_pun)
                R[6][:, yr] = diff * eff_mwh * pos_hours[:, yr] * 365 * cal.congestion_rent_fraction

                # R7: Nodal/LMP
                p_nod = 1.0 - (1.0 - cal.nodal_transition_annual_prob) ** (yr + 1)
                nod_draw = rng_sub.random(n)
                nod_prem = np.where(nod_draw < p_nod, rng_sub.uniform(10, 25, n), 0.0)
                R[7][:, yr] = eff_mw * nod_prem * 4000 * p_nod

                # R8-R10: Trend-driven
                R[8][:, yr] = eff_mw * cal.cer_base_revenue * (1 + cal.cer_growth_rate) ** yr * np.maximum(0.5, rng_sub.normal(1.0, 0.15, n))
                R[9][:, yr] = eff_mw * cal.dso_service_base * (1 + cal.dso_growth_rate) ** yr * np.maximum(0.3, rng_sub.normal(1.0, 0.20, n))
                R[10][:, yr] = eff_mw * cal.pqaas_base * (1 + cal.pqaas_growth_rate) ** yr * np.maximum(0.5, rng_sub.normal(1.0, 0.10, n))

            # OPEX
            opex = np.array([capex_m * 1e6 * cal.opex_pct_capex * (1 + cal.inflation_rate) ** yr for yr in range(horizon)])
            gross = sum(R[ri] for ri in range(1, 11))
            net = gross - opex[np.newaxis, :]

            # NPV
            npv_arr = -capex_m + np.sum(net * disc[np.newaxis, :], axis=1) / 1e6

            # IRR (vectorized)
            years = np.arange(1, horizon + 1)
            low = np.full(n, -0.10)
            high = np.full(n, 0.50)
            for _ in range(40):
                mid = (low + high) / 2
                d = 1.0 / (1.0 + mid[:, np.newaxis]) ** years[np.newaxis, :]
                pv = np.sum(net * d, axis=1)
                mask = pv > capex_m * 1e6
                low = np.where(mask, mid, low)
                high = np.where(mask, high, mid)
            irr_arr = (low + high) / 2 * 100

            # Stream totals
            stream_totals = {f'R{ri}': np.sum(R[ri], axis=1) for ri in range(1, 11)}

            # Build compact MC result
            total_all = sum(float(np.mean(v)) for v in stream_totals.values())
            stream_data = {}
            for ri in range(1, 11):
                mv = float(np.mean(stream_totals[f'R{ri}']))
                stream_data[f'R{ri}_total_mean'] = round(mv / 1e6, 4)
                stream_data[f'R{ri}_pct'] = round(mv / max(1, total_all) * 100, 1)

            var_95 = float(np.percentile(npv_arr, 5))
            tail_m = npv_arr[npv_arr <= var_95]
            tvar_95 = float(np.mean(tail_m)) if len(tail_m) > 0 else var_95

            mc_result = {
                'npv': {
                    'npv_mean': round(float(np.mean(npv_arr)), 4),
                    'npv_median': round(float(np.median(npv_arr)), 4),
                    'npv_std': round(float(np.std(npv_arr)), 4),
                    'npv_P5': round(float(np.percentile(npv_arr, 5)), 4),
                    'npv_P25': round(float(np.percentile(npv_arr, 25)), 4),
                    'npv_P50': round(float(np.percentile(npv_arr, 50)), 4),
                    'npv_P75': round(float(np.percentile(npv_arr, 75)), 4),
                    'npv_P95': round(float(np.percentile(npv_arr, 95)), 4),
                    'npv_positive_pct': round(float(np.mean(npv_arr > 0) * 100), 1),
                },
                'irr': {
                    'irr_mean': round(float(np.mean(irr_arr)), 2),
                    'irr_median': round(float(np.median(irr_arr)), 2),
                    'irr_P5': round(float(np.percentile(irr_arr, 5)), 2),
                    'irr_P95': round(float(np.percentile(irr_arr, 95)), 2),
                },
                'risk': {
                    'var_95_M': round(var_95, 4),
                    'tvar_95_M': round(tvar_95, 4),
                    'sharpe_ratio': round(float(np.mean(npv_arr)) / max(0.01, float(np.std(npv_arr))), 3),
                    'max_loss_M': round(float(np.min(npv_arr)), 4),
                    'max_gain_M': round(float(np.max(npv_arr)), 4),
                },
                'streams': stream_data,
                'zone': zone,
                'config': config_name,
                'crs': crs,
                'bess_sat': bess_sat,
                'n_paths': n_paths,
            }

            sub['mc_results'] = mc_result
            results.append(mc_result)
            processed += 1

            if verbose and processed % 100 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                eta = (len(targets) - processed) / rate if rate > 0 else 0
                print(f"  [{processed:,}/{len(targets):,}] "
                      f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s "
                      f"| Last NPV €{mc_result['npv']['npv_median']:.2f}M")

        # Free zone prices memory
        del zone_prices, daily_all, daily_sorted_all

    elapsed = time.time() - t0

    if verbose and results:
        npvs = np.array([r['npv']['npv_median'] for r in results])
        irrs = np.array([r['irr']['irr_median'] for r in results])
        sharpes = np.array([r['risk']['sharpe_ratio'] for r in results])

        print(f"\n  {'─'*50}")
        print(f"  FLEET MC RESULTS ({len(results):,} substations)")
        print(f"  {'─'*50}")
        print(f"  NPV median (fleet):     €{np.median(npvs):.2f}M")
        print(f"  NPV P5 (fleet):         €{np.percentile(npvs, 5):.2f}M")
        print(f"  NPV P95 (fleet):        €{np.percentile(npvs, 95):.2f}M")
        print(f"  IRR median (fleet):     {np.median(irrs):.1f}%")
        print(f"  Sharpe ratio (median):  {np.median(sharpes):.2f}")
        print(f"  Positive NPV:           {np.mean(npvs > 0)*100:.0f}%")
        print(f"  Elapsed:                {elapsed:.1f}s")
        print(f"  Rate:                   {len(results)/elapsed:.1f} subs/sec")

        # Revenue stream breakdown (fleet average)
        for ri in range(1, 11):
            pcts = [r['streams'].get(f'R{ri}_pct', 0) for r in results]
            print(f"  R{ri:2d} share:  {np.mean(pcts):5.1f}%")

    return substations
