"""
SSI-ENN BESS — Neural Network Inference Module (v3)
====================================================
Load trained models and score all substations with predictions.
v3: 32-feature set with BS/Actuarial/Cannibalization enrichments.

Outputs:
  - nn_predictions.json: Complete per-substation predictions with confidence scores

Models used:
  1. BESS Recommender   (MLP) → Config A/B classification + probability
  2. NPV Regressor      (MLP) → Config B NPV prediction
  3. Band Predictor     (RF)  → Low/Medium/High/Critical classification
  4. Anomaly Detector   (IF)  → Anomaly flags and types
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import joblib

from pipeline.nn_trainer import FeatureEngineer, BAND_NAMES, BAND_MAP, ALL_FEATURES

# ─────────────────────────── Paths ───────────────────────────

PIPELINE_DIR = Path(__file__).resolve().parent
BASE_DIR = PIPELINE_DIR.parent


# ═══════════════════════════════════════════════════════════════
# Inference Engine
# ═══════════════════════════════════════════════════════════════

class NeuralNetworkInference:
    """Load models and score all substations."""

    def __init__(self, data_file: Path, models_dir: Path, verbose: bool = True):
        self.verbose = verbose
        self.t0 = time.time()
        self.data_file = Path(data_file)
        self.models_dir = Path(models_dir)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  SSI-ENN BESS — Neural Network Inference (v2)")
            print(f"{'='*70}")
            print(f"  Data file       : {self.data_file}")
            print(f"  Models dir      : {self.models_dir}")

        # Load data
        with open(self.data_file) as f:
            self.substations = json.load(f)

        if self.verbose:
            print(f"  Substations     : {len(self.substations):,}")

        # Feature engineering
        self.fe = FeatureEngineer(self.substations)
        self.X = self.fe.get_feature_matrix()

        if self.verbose:
            print(f"  Features        : {self.X.shape[1]}")

        # Load models
        self._load_models()

        # Storage for results
        self.predictions = {}
        self.meta = {}

    def _load_models(self):
        """Load all trained models from disk."""
        if self.verbose:
            print(f"\n  Loading models...")

        self.scaler = joblib.load(self.models_dir / 'scaler.pkl')
        self.bess_recommender = joblib.load(self.models_dir / 'bess_recommender.pkl')
        self.npv_regressor = joblib.load(self.models_dir / 'npv_regressor.pkl')
        self.band_predictor = joblib.load(self.models_dir / 'band_predictor.pkl')
        self.anomaly_detector = joblib.load(self.models_dir / 'anomaly_detector.pkl')

        if self.verbose:
            print(f"    bess_recommender.pkl   (MLP)")
            print(f"    npv_regressor.pkl      (MLP)")
            print(f"    band_predictor.pkl     (RandomForest)")
            print(f"    anomaly_detector.pkl   (IsolationForest)")
            print(f"    scaler.pkl             (StandardScaler)")

    def score_all(self) -> dict:
        """Score all substations with all 4 models."""
        if self.verbose:
            print(f"\n  Scoring {len(self.substations):,} substations...")

        # Transform features
        X_scaled = self.scaler.transform(self.X)

        # ═══════════════════════════════════════════════════════════════
        # Model 1: BESS Recommender
        # ═══════════════════════════════════════════════════════════════

        rec_pred = self.bess_recommender.predict(X_scaled)
        rec_proba = self.bess_recommender.predict_proba(X_scaled)
        # Probability of predicted class
        rec_confidence = rec_proba[np.arange(len(rec_proba)), rec_pred]

        # ═══════════════════════════════════════════════════════════════
        # Model 2: NPV Regressor
        # ═══════════════════════════════════════════════════════════════

        npv_pred = self.npv_regressor.predict(X_scaled)

        # ═══════════════════════════════════════════════════════════════
        # Model 3: Band Predictor (uses unscaled features)
        # ═══════════════════════════════════════════════════════════════

        band_pred = self.band_predictor.predict(self.X)

        # ═══════════════════════════════════════════════════════════════
        # Model 4: Anomaly Detector (3-residual approach)
        # ═══════════════════════════════════════════════════════════════

        # Compute residuals
        y_rec = self.fe.get_target('recommendation')
        y_npv = self.fe.get_target('npv_b')
        y_band = self.fe.get_target('band')

        rec_err = np.abs(y_rec - rec_pred).astype(float)
        npv_err = np.abs(y_npv - npv_pred)
        band_err = np.abs(y_band - band_pred).astype(float)

        # Normalise NPV residuals by std dev
        npv_std = np.std(y_npv) + 1e-10
        npv_err_norm = npv_err / npv_std

        # 3-dimensional residual matrix
        residual_matrix = np.column_stack([rec_err, npv_err_norm, band_err])

        # Get anomaly scores
        anom_scores = self.anomaly_detector.score_samples(residual_matrix)
        anom_labels = self.anomaly_detector.predict(residual_matrix)

        # Normalise scores to [0, 1] where 1 = most anomalous
        anom_scores_norm = 1.0 - (anom_scores - anom_scores.min()) / (
            anom_scores.max() - anom_scores.min() + 1e-10
        )

        is_anomaly = (anom_labels == -1).astype(bool)

        # ═══════════════════════════════════════════════════════════════
        # Build per-substation predictions
        # ═══════════════════════════════════════════════════════════════

        for i, substation in enumerate(self.substations):
            substation_id = substation.get('substation_id', f'unknown_{i}')

            # Recommendation
            rec_class = 'Config B' if rec_pred[i] == 1 else 'Config A'

            # Band prediction
            band_class = BAND_NAMES[band_pred[i]]
            band_actual = substation.get('classification', 'Unknown')
            band_correct = (band_class == band_actual)

            # Anomaly type classification
            if is_anomaly[i]:
                types = []
                if rec_err[i] > 0:
                    types.append('rec_mismatch')
                if npv_err[i] > 2.0:  # > 2 std devs
                    types.append('npv_outlier')
                if band_err[i] > 0:
                    types.append('band_mismatch')

                if len(types) >= 2:
                    anomaly_type = 'multi_signal'
                elif len(types) == 1:
                    anomaly_type = types[0]
                else:
                    anomaly_type = 'statistical'
            else:
                anomaly_type = None

            # NPV residual
            cfg_b = substation.get('bess', {}).get('config_B', {})
            actual_npv = cfg_b.get('NPV_M', 0)
            npv_residual = npv_pred[i] - actual_npv

            self.predictions[substation_id] = {
                'nn_recommendation': rec_class,
                'nn_recommendation_confidence': round(float(rec_confidence[i]), 4),
                'nn_npv_predicted_M': round(float(npv_pred[i]), 4),
                'nn_npv_residual_M': round(float(npv_residual), 4),
                'nn_band_predicted': band_class,
                'nn_band_correct': bool(band_correct),
                'nn_anomaly_flag': bool(is_anomaly[i]),
                'nn_anomaly_score': round(float(anom_scores_norm[i]), 4),
                'nn_anomaly_type': anomaly_type,
            }

        # ═══════════════════════════════════════════════════════════════
        # Build metadata
        # ═══════════════════════════════════════════════════════════════

        duration = time.time() - self.t0
        n_anomalies = int(is_anomaly.sum())

        self.meta = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'version': 3,
            'total_scored': len(self.substations),
            'total_anomalies': n_anomalies,
            'anomaly_pct': round(100 * n_anomalies / len(self.substations), 2),
            'models': {
                'bess_recommender': 'MLPClassifier(128,64,32)',
                'npv_regressor': 'MLPRegressor(512,256,128)',
                'band_predictor': 'RandomForestClassifier(200)',
                'anomaly_detector': 'IsolationForest(200, contamination=0.05)',
                'scaler': 'StandardScaler',
            },
            'feature_count': self.X.shape[1],
            'enrichments': ['black_swan', 'cannibalization', 'actuarial'],
            'duration_s': round(duration, 2),
        }

        return self.predictions

    def save_predictions(self, output_file: Path) -> Path:
        """Save predictions to JSON file."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        output = {**self.predictions, '_meta': self.meta}

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        if self.verbose:
            size = output_file.stat().st_size
            n_records = len(self.predictions)
            print(f"\n  Predictions saved:")
            print(f"    File        : {output_file}")
            print(f"    Size        : {size:,} bytes")
            print(f"    Records     : {n_records:,}")

        return output_file


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Score all substations with trained neural network models'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default='data.json',
        help='Path to input data.json (default: data.json)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='pipeline/nn_models',
        help='Directory containing trained models (default: pipeline/nn_models)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='nn_predictions.json',
        help='Output file for predictions (default: nn_predictions.json)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Ensure paths are absolute (relative to base dir)
    data_file = BASE_DIR / args.data_file
    models_dir = BASE_DIR / args.models_dir
    output_file = BASE_DIR / args.output

    # Run inference
    engine = NeuralNetworkInference(
        data_file=data_file,
        models_dir=models_dir,
        verbose=not args.quiet
    )

    engine.score_all()
    engine.save_predictions(output_file)

    if not args.quiet:
        print(f"\n{'='*70}")
        print(f"  INFERENCE COMPLETE")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
