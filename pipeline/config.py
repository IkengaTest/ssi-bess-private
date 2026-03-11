"""
Central configuration for the SSI-ENN BESS Ingestion Pipeline.
Single source of truth for data sources, BESS parameters, validation rules.
"""

from pathlib import Path

# ─────────────────────────── Paths ───────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent          # bess-private/
PIPELINE_DIR = Path(__file__).resolve().parent              # bess-private/pipeline/
OUTPUT_DIR = BASE_DIR                                       # data.json lives here
CACHE_DIR = PIPELINE_DIR / '.ingestion_cache'
VERSION_FILE = PIPELINE_DIR / 'version.txt'

# ─────────────────────────── Data Sources (28 providers) ───────────────────────────

DATA_SOURCES = {
    # ── PRIMARY (fetched live) ──
    'ssi_substations': {
        'url': 'https://ikengassiindex.github.io/ssi-data.json',
        'section': 'A',
        'critical': True,
        'description': 'Real SSI v4.0 substations (4,293 records)',
        'provider': 'Ikenga SSI Index',
    },
    'grid_geometry': {
        'url': 'https://ikengassiindex.github.io/grid-geo.json',
        'section': 'A',
        'critical': True,
        'description': 'Grid topology — 14,221 transmission/distribution lines',
        'provider': 'Ikenga SSI Index',
    },

    # ── SECTION A: Core Metrics (embedded in SSI data) ──
    'arera_continuity':       {'section': 'A', 'provider': 'ARERA', 'variables': ['C1','C2','C3','C4'], 'status': 'embedded'},
    'arera_voltage':          {'section': 'A', 'provider': 'ARERA TIQE', 'variables': ['V1','V2'], 'status': 'embedded'},
    'ispra_flood':            {'section': 'A', 'provider': 'ISPRA IdroGeo', 'variables': ['I1'], 'status': 'embedded'},
    'ispra_landslide':        {'section': 'A', 'provider': 'ISPRA IFFI', 'variables': ['I2'], 'status': 'embedded'},
    'copernicus_heatwave':    {'section': 'A', 'provider': 'Copernicus CDS ERA5', 'variables': ['I3'], 'status': 'embedded'},
    'osm_infrastructure':     {'section': 'A', 'provider': 'OpenStreetMap', 'variables': ['I4','I6'], 'status': 'embedded'},
    'arera_thermal':          {'section': 'A', 'provider': 'IEEE C57.91 / ERA5 / Terna', 'variables': ['I5'], 'status': 'embedded'},
    'terna_load':             {'section': 'A', 'provider': 'Terna Open Data', 'variables': ['I7'], 'status': 'embedded'},
    'eea_corrosion':          {'section': 'A', 'provider': 'EEA Air Quality', 'variables': ['I8'], 'status': 'embedded'},
    'ispra_hydrogeo':         {'section': 'A', 'provider': 'ISPRA Hydrogeology', 'variables': ['I9'], 'status': 'embedded'},
    'istat_gdp':              {'section': 'A', 'provider': 'Eurostat / ISTAT', 'variables': ['E1'], 'status': 'embedded'},
    'arera_voll':             {'section': 'A', 'provider': 'ARERA / CEER', 'variables': ['E2'], 'status': 'embedded'},
    'gse_der':                {'section': 'A', 'provider': 'GSE Atlaimpianti', 'variables': ['S1','S2','S3'], 'status': 'embedded'},
    'gse_transition':         {'section': 'A', 'provider': 'GSE / Terna / ENEA', 'variables': ['T1'], 'status': 'embedded'},

    # ── SECTION A.M: Modifier Inputs (Phase 2 — static lookups) ──
    'copernicus_cmip6':       {'section': 'A.M', 'provider': 'Copernicus CDS CMIP6', 'variables': ['R2'], 'status': 'phase2'},
    'oipe_energy_poverty':    {'section': 'A.M', 'provider': 'OIPE / MEF', 'variables': ['R3'], 'status': 'phase2'},
    'osm_topology':           {'section': 'A.M', 'provider': 'OSM Overpass', 'variables': ['R4'], 'status': 'phase2'},
    'arera_caidi':            {'section': 'A.M', 'provider': 'ARERA TIQE', 'variables': ['R6a'], 'status': 'phase2'},
    'ingv_seismic':           {'section': 'A.M', 'provider': 'INGV MPS04', 'variables': ['R6b'], 'status': 'phase2'},
    'desi_digital':           {'section': 'A.M', 'provider': 'EU DESI / Eurostat', 'variables': ['R7'], 'status': 'phase2'},

    # ── SECTION B: Academic Enrichment (derived from fleet) ──
    'academic_fleet':         {'section': 'B', 'provider': 'Derived (ARERA tail)', 'variables': ['entropy','DSRR','fragility','CPIndex','Jensen','DER_coupling'], 'status': 'embedded'},

    # ── SECTION C: Trajectory Inputs (embedded in SSI data) ──
    'trajectory_inputs':      {'section': 'C', 'provider': 'PNIEC / GSE / Terna / GME', 'variables': ['C.1-C.10'], 'status': 'embedded'},

    # ── SECTION D: BESS Impact (from config) ──
    'bess_specifications':    {'section': 'D', 'provider': 'Plan of Work v11', 'variables': ['D.1-D.5'], 'status': 'config'},
}

# ─────────────────────────── BESS Configuration ───────────────────────────

BESS_CONFIGS = {
    'A': {'name': 'A', 'mw': 4.9, 'mwh': 19.6, 'capex_m': 3.5},
    'B': {'name': 'B', 'mw': 9.9, 'mwh': 39.6, 'capex_m': 7.0},
}

# Financial parameters
WACC = 0.052           # Weighted average cost of capital
HORIZON = 25           # Analysis horizon (years)
DEGRADATION = 0.008    # Battery degradation 0.8% annual

# V_DNO ranges by config
V_DNO_RANGES = {
    'A': (2.0, 15.0),
    'B': (4.0, 30.0),
}
V_COMMUNITY_RANGES = {
    'A': (0.5, 5.0),
    'B': (1.0, 10.0),
}
V_RO_RANGES = {
    'A': (0.02, 1.5),
    'B': (0.02, 3.0),
}

# ─────────────────────────── Classification Bands ───────────────────────────

CLASSIFICATION_BANDS = [
    ('Critical', 0.65, 1.00),
    ('High',     0.50, 0.65),
    ('Medium',   0.35, 0.50),
    ('Low',      0.00, 0.35),
]

# Investment priority thresholds (IRR → priority 1-5)
PRIORITY_THRESHOLDS = [
    (16.0, 5),
    (13.0, 4),
    (10.0, 3),
    (7.0,  2),
    (0.0,  1),
]

# ─────────────────────────── Validation Rules ───────────────────────────

ITALY_BOUNDS = {
    'lat_min': 36.6, 'lat_max': 47.6,
    'lon_min': 6.6,  'lon_max': 18.5,
}

VALID_REGIONS = [
    'Abruzzo', 'Basilicata', 'Calabria', 'Campania', 'Emilia-Romagna',
    'Friuli Venezia Giulia', 'Lazio', 'Liguria', 'Lombardia', 'Marche',
    'Molise', 'Piemonte', 'Puglia', 'Sardegna', 'Sicilia', 'Toscana',
    'Trentino-Alto Adige', 'Umbria', "Valle d'Aosta", 'Veneto',
]

VALIDATION_RULES = {
    # Core identifiers
    'substation_id': {'type': str, 'required': True},
    'name':          {'type': str, 'required': True},
    'lat':           {'type': float, 'required': True, 'min': 36.6, 'max': 47.6},
    'lon':           {'type': float, 'required': True, 'min': 6.6, 'max': 18.5},
    'voltage_kv':    {'type': (int, float), 'required': False, 'min': 0, 'max': 500},
    'region':        {'type': str, 'required': True},
    'province':      {'type': str, 'required': True},

    # SSI scores
    'R_median':      {'type': float, 'required': True, 'min': 0.0, 'max': 1.0},
    'R_P5':          {'type': float, 'required': False, 'min': 0.0, 'max': 1.0},
    'R_P95':         {'type': float, 'required': False, 'min': 0.0, 'max': 1.0},
    'R_base_median': {'type': float, 'required': False, 'min': 0.0, 'max': 1.0},
    'CI_width':      {'type': float, 'required': False, 'min': 0.0, 'max': 1.0},
    'fleet_percentile': {'type': float, 'required': False, 'min': 0.0, 'max': 1.0},
}

COMPONENT_RULES = {
    'C': {'min': 0.0, 'max': 1.0},
    'V': {'min': 0.0, 'max': 1.0},
    'I': {'min': 0.0, 'max': 1.0},
    'E': {'min': 0.0, 'max': 1.0},
    'S': {'min': 0.0, 'max': 1.0},
    'T': {'min': 0.0, 'max': 1.0},
}

MODIFIER_BOUNDS = {
    'R3': {'min': 0.70, 'max': 1.35},
    'R4': {'min': 0.70, 'max': 1.40},
    'R6': {'min': 0.60, 'max': 1.30},
    'R7': {'min': 0.80, 'max': 1.10},
}

BESS_VALIDATION = {
    'V_DNO_M':       {'min': 0.0, 'max': 50.0},
    'V_Community_M': {'min': 0.0, 'max': 20.0},
    'V_RO_M':        {'min': 0.0, 'max': 5.0},
    'V_Total_M':     {'min': 0.0, 'max': 75.0},
    'CAPEX_M':       {'min': 1.0, 'max': 10.0},
    'NPV_M':         {'min': -20.0, 'max': 50.0},
    'IRR_pct':       {'min': 0.0, 'max': 50.0},
    'Payback_yr':    {'min': 1.0, 'max': 25.0},
}

# ─────────────────────────── Enrichment Module Config ───────────────────────────

# Cannibalization enrichment — see cannibalization.py for full documentation
CANNIBALIZATION = {
    'x_crit': 0.20,           # Critical BESS saturation threshold
    'gamma': 2.0,             # Saturation curve exponent (convex)
    'delta': 0.60,            # Maximum revenue haircut at full saturation
    'beta_spread': 4.0,       # Spread compression coefficient
    'kappa_cann': 0.25,       # PAD loading for cannibalization
    'projection_years': 10,   # Forward saturation projection horizon
}

# Black Swan enrichment — see black_swan.py for full documentation
BLACK_SWAN = {
    'compound_climate_weight': 0.15,
    'regime_shift_prob': 0.08,
    'supply_disruption_prob': 0.05,
    'commodity_spike_multiplier': 2.0,
    'rate_shock_bps': 200.0,
}

# Actuarial enrichment — see actuarial.py for full documentation
ACTUARIAL = {
    'tvar_alpha': 0.95,
    'pad_base_bps': 75.0,
    'kappa_tail': 0.30,
    'kappa_ci': 0.15,
    'kappa_cann': 0.25,
    'gpd_xi_prior': 0.25,
    'wacc_base': 0.052,
}

# ─────────────────────────── Data Sources (enrichment) ───────────────────────────

ENRICHMENT_DATA_SOURCES = {
    # CN.1 — Terna BESS Registry
    'terna_bess_registry': {
        'provider': 'Terna Open Data',
        'variables': ['BESS_MW_installed', 'zone_peak_load'],
        'status': 'phase1_defaults',
        'description': 'Installed BESS capacity per zone (defaults until API available)',
    },
    # CN.2 — ARERA Revenue Stack
    'arera_revenue_stack': {
        'provider': 'ARERA / GME',
        'variables': ['revenue_weights_per_stream'],
        'status': 'phase1_defaults',
        'description': 'Revenue stack weights per ancillary service',
    },
    # CN.3 — Terna Grid Plan (PNIEC)
    'terna_grid_plan': {
        'provider': 'Terna / PNIEC',
        'variables': ['BESS_SAT_forward', 'growth_trajectory'],
        'status': 'phase1_defaults',
        'description': 'Forward BESS saturation from national energy plan',
    },
    # BS.1 — Copernicus ERA5 Extremes
    'copernicus_extremes': {
        'provider': 'Copernicus CDS',
        'variables': ['heat_score', 'drought_score', 'fire_score'],
        'status': 'phase2',
        'description': 'Extreme weather indices for compound climate stress',
    },
    # BS.2 — ECB Monetary Policy
    'ecb_rates': {
        'provider': 'ECB / Eurostat',
        'variables': ['policy_rate', 'spread_history'],
        'status': 'phase2',
        'description': 'Interest rate history for financing shock scenarios',
    },
    # ACT.1 — ARERA TIQE Tails
    'arera_tiqe_tails': {
        'provider': 'ARERA TIQE',
        'variables': ['continuity_tail_distribution'],
        'status': 'phase2',
        'description': 'Continuity metric tail distributions for GPD fitting',
    },
}

# ─────────────────────────── Cache Settings ───────────────────────────

CACHE_TTL_HOURS = 24
HTTP_TIMEOUT = 30
HTTP_RETRIES = 3
