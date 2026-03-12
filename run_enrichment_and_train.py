#!/usr/bin/env python3
"""
Run enrichments + NN training on existing data.json.
Skips the network fetch step — uses cached/existing data directly.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.black_swan import enrich_fleet as bs_enrich_fleet
from pipeline.cannibalization import enrich_fleet as cann_enrich_fleet
from pipeline.actuarial import enrich_fleet as act_enrich_fleet

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'data.json'
MODELS_DIR = BASE_DIR / 'pipeline' / 'nn_models'

t0 = time.time()

# ── Step 1: Load existing data.json ──
print(f"\n{'='*70}")
print(f"  SSI-ENN BESS — Enrichment + Training Pipeline (v3.4 — Full L3 Upgrade)")
print(f"{'='*70}")
print(f"\n  Loading {DATA_FILE}...")
with open(DATA_FILE) as f:
    substations = json.load(f)
print(f"  Loaded {len(substations):,} substations")

# ── Step 2: Apply enrichments ──
substations = bs_enrich_fleet(substations, verbose=True)
substations = cann_enrich_fleet(substations, verbose=True)
substations = act_enrich_fleet(substations, verbose=True)

# ── Step 3: Write enriched data.json ──
print(f"\n  Writing enriched data.json...")
with open(DATA_FILE, 'w') as f:
    json.dump(substations, f)
size_mb = DATA_FILE.stat().st_size / 1024 / 1024
print(f"  data.json: {len(substations):,} records ({size_mb:.1f} MB)")

# ── Step 4: Train NN models ──
print(f"\n{'='*70}")
print(f"  Starting NN Training (v3.4 — 38 features, 7 models)")
print(f"{'='*70}")

from pipeline.nn_trainer import NeuralNetworkTrainer

trainer = NeuralNetworkTrainer(substations, verbose=True)
metrics = trainer.train_all()
model_dir = trainer.save_models()

# ── Step 5: Run inference ──
print(f"\n{'='*70}")
print(f"  Running NN Inference")
print(f"{'='*70}")

from pipeline.nn_inference import NeuralNetworkInference

inferencer = NeuralNetworkInference(
    data_file=DATA_FILE,
    models_dir=MODELS_DIR,
    verbose=True,
)
inferencer.score_all()
output_path = BASE_DIR / 'nn_predictions.json'
inferencer.save_predictions(output_path)

# ── Summary ──
duration = time.time() - t0
print(f"\n{'='*70}")
print(f"  COMPLETE — {duration:.1f}s")
print(f"{'='*70}")
print(f"  Features:      {trainer.X.shape[1]}")
print(f"  Substations:   {len(substations):,}")
print(f"  Models:        4 (recommender, NPV, band, anomaly)")
print(f"  Predictions:   {inferencer.meta.get('total_scored', 0):,}")
print(f"  Anomalies:     {inferencer.meta.get('total_anomalies', 0)}")
print(f"{'='*70}\n")
