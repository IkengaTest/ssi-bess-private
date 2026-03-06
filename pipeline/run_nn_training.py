#!/usr/bin/env python3
"""
SSI-ENN BESS — Neural Network Training Entry Point
====================================================
Usage:
    python3 -m pipeline.run_nn_training
    python3 -m pipeline.run_nn_training --test-size 0.3 --verbose
"""

import json
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # bess-private/


def main():
    parser = argparse.ArgumentParser(
        description='Train 4 NN models on SSI-ENN BESS data'
    )
    parser.add_argument(
        '--data-file', default=str(BASE_DIR / 'data.json'),
        help='Path to data.json (default: bess-private/data.json)'
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Model output directory (default: pipeline/nn_models/)'
    )
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--seed', type=int, default=42, help='Random state')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    args = parser.parse_args()

    # Load data
    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        return 1

    with open(data_path) as f:
        substations = json.load(f)
    print(f"Loaded {len(substations):,} substations from {data_path.name}")

    # Train
    from pipeline.nn_trainer import NeuralNetworkTrainer

    trainer = NeuralNetworkTrainer(
        substations,
        test_size=args.test_size,
        random_state=args.seed,
        verbose=not args.quiet,
    )
    metrics = trainer.train_all()

    # Save
    output_dir = Path(args.output_dir) if args.output_dir else None
    saved_path = trainer.save_models(output_dir)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    rec = metrics.get('bess_recommender', {})
    npv = metrics.get('npv_regressor', {})
    band = metrics.get('band_predictor', {})
    anom = metrics.get('anomaly_detector', {})
    print(f"  1. Recommender  Acc={rec.get('accuracy_test',0):.3f}  F1={rec.get('f1_test',0):.3f}  AUC={rec.get('roc_auc_test',0):.3f}")
    print(f"  2. NPV Regress  R²={npv.get('r2_test',0):.3f}  MAE=€{npv.get('mae_test',0):.3f}M  Spearman={npv.get('spearman_corr',0):.3f}")
    print(f"  3. Band Predict Acc={band.get('accuracy_test',0):.3f}  F1w={band.get('f1_weighted_test',0):.3f}")
    print(f"  4. Anomaly Det  {anom.get('n_anomalies',0)} flagged ({anom.get('pct_anomalies',0):.1f}%)")
    print(f"\n  Models → {saved_path}/")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
