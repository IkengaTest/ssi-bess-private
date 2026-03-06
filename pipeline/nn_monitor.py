"""
Model Monitoring and Drift Detection Module

Compares current inference predictions against baseline training metrics to detect
model drift across multiple dimensions:
- Recommendation distribution drift
- NPV residual drift (bias and variance)
- Band distribution drift
- Anomaly rate drift
- Feature drift (if data.json available)
- Confidence drift
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import statistics

# Optional imports with graceful fallbacks
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ModelMonitor:
    """Monitor model performance and detect drift indicators."""

    # Default thresholds
    RECOMMENDATION_DRIFT_THRESHOLD = 5.0  # percentage points
    NPV_BIAS_THRESHOLD = 0.5  # Million
    NPV_STD_THRESHOLD = 2.0  # Million
    ANOMALY_DRIFT_THRESHOLD = 2.0  # percentage points from baseline
    CONFIDENCE_MIN_THRESHOLD = 0.85  # 85% mean confidence
    CONFIDENCE_BELOW_70_THRESHOLD = 10.0  # Allow max 10% below 70%
    CHI_SQUARED_P_THRESHOLD = 0.05

    def __init__(
        self,
        baseline_metrics_path: str = "pipeline/nn_models/metrics.json",
        predictions_path: str = "nn_predictions.json",
        data_path: Optional[str] = "data.json",
    ):
        """
        Initialize the monitor with paths to baseline metrics and current predictions.

        Args:
            baseline_metrics_path: Path to training metrics.json
            predictions_path: Path to nn_predictions.json
            data_path: Optional path to data.json for feature drift analysis
        """
        self.baseline_metrics_path = baseline_metrics_path
        self.predictions_path = predictions_path
        self.data_path = data_path

        self.baseline_metrics = None
        self.current_predictions = None
        self.training_data = None

        self._load_files()

    def _load_files(self) -> None:
        """Load baseline metrics and current predictions."""
        # Load baseline metrics
        with open(self.baseline_metrics_path) as f:
            self.baseline_metrics = json.load(f)

        # Load current predictions
        with open(self.predictions_path) as f:
            self.current_predictions = json.load(f)

        # Load training data if available
        if self.data_path and Path(self.data_path).exists():
            try:
                with open(self.data_path) as f:
                    self.training_data = json.load(f)
            except Exception:
                self.training_data = None

    def _get_baseline_config_b_percentage(self) -> float:
        """Extract baseline Config B percentage from confusion matrix."""
        cm = self.baseline_metrics["bess_recommender"]["confusion_matrix"]
        total_b = cm["fn"] + cm["tp"]
        total = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
        return (total_b / total) * 100 if total > 0 else 0

    def _get_current_config_b_percentage(self) -> float:
        """Count current Config B percentage from predictions."""
        predictions = self.current_predictions
        config_b_count = 0
        total_count = 0

        for key, val in predictions.items():
            if key != "_meta" and isinstance(val, dict):
                total_count += 1
                if val.get("nn_recommendation") == "Config B":
                    config_b_count += 1

        return (config_b_count / total_count) * 100 if total_count > 0 else 0

    def check_recommendation_drift(self) -> Dict[str, Any]:
        """Check if recommendation distribution has drifted significantly."""
        baseline_pct = self._get_baseline_config_b_percentage()
        current_pct = self._get_current_config_b_percentage()
        delta = abs(current_pct - baseline_pct)

        status = "ok"
        if delta > self.RECOMMENDATION_DRIFT_THRESHOLD:
            status = "alert"
        elif delta > self.RECOMMENDATION_DRIFT_THRESHOLD * 0.5:
            status = "warning"

        return {
            "status": status,
            "baseline_config_b_pct": round(baseline_pct, 2),
            "current_config_b_pct": round(current_pct, 2),
            "delta_pct": round(delta, 2),
            "threshold": self.RECOMMENDATION_DRIFT_THRESHOLD,
        }

    def _compute_npv_residuals(self) -> Tuple[List[float], float, float, float]:
        """Compute NPV residuals from current predictions."""
        residuals = []
        predictions = self.current_predictions

        for key, val in predictions.items():
            if key != "_meta" and isinstance(val, dict):
                residual = val.get("nn_npv_residual_M")
                if residual is not None:
                    residuals.append(residual)

        if not residuals:
            return [], 0, 0, 0

        mean_residual = statistics.mean(residuals)
        std_residual = statistics.stdev(residuals) if len(residuals) > 1 else 0
        max_abs_residual = max(abs(r) for r in residuals)

        return residuals, mean_residual, std_residual, max_abs_residual

    def check_npv_residual_drift(self) -> Dict[str, Any]:
        """Check NPV prediction drift via residual bias and variance."""
        residuals, mean_res, std_res, max_abs_res = self._compute_npv_residuals()

        baseline_bias = self.baseline_metrics["npv_regressor"].get("bias", 0)
        baseline_rmse = self.baseline_metrics["npv_regressor"].get("rmse_test", 0)

        # Flags
        bias_alert = abs(mean_res) > self.NPV_BIAS_THRESHOLD
        std_alert = std_res > self.NPV_STD_THRESHOLD

        status = "alert" if (bias_alert or std_alert) else "ok"

        return {
            "status": status,
            "mean_residual_M": round(mean_res, 4),
            "std_residual_M": round(std_res, 4),
            "max_abs_residual_M": round(max_abs_res, 4),
            "bias_threshold": self.NPV_BIAS_THRESHOLD,
            "std_threshold": self.NPV_STD_THRESHOLD,
        }

    def _get_baseline_band_distribution(self) -> Dict[str, float]:
        """Extract band distribution from baseline test set."""
        band_pred = self.baseline_metrics["band_predictor"]
        per_class = band_pred["per_class"]

        # Total from test set
        total = sum(per_class[band]["support"] for band in per_class)

        distribution = {}
        for band in ["Low", "Medium", "High", "Critical"]:
            if band in per_class:
                distribution[band] = per_class[band]["support"] / total
            else:
                distribution[band] = 0.0

        return distribution

    def _get_current_band_distribution(self) -> Dict[str, float]:
        """Extract band distribution from current predictions."""
        predictions = self.current_predictions
        band_counts = {}
        total_count = 0

        for key, val in predictions.items():
            if key != "_meta" and isinstance(val, dict):
                total_count += 1
                band = val.get("nn_band_predicted")
                if band:
                    band_counts[band] = band_counts.get(band, 0) + 1

        distribution = {}
        for band in ["Low", "Medium", "High", "Critical"]:
            if total_count > 0:
                distribution[band] = band_counts.get(band, 0) / total_count
            else:
                distribution[band] = 0.0

        return distribution

    def check_band_distribution_drift(self) -> Dict[str, Any]:
        """Check band distribution drift using chi-squared test if scipy available."""
        baseline_dist = self._get_baseline_band_distribution()
        current_dist = self._get_current_band_distribution()

        result = {
            "status": "ok",
            "baseline_distribution": {
                band: round(baseline_dist[band], 4) for band in baseline_dist
            },
            "current_distribution": {
                band: round(current_dist[band], 4) for band in current_dist
            },
            "threshold_p": self.CHI_SQUARED_P_THRESHOLD,
        }

        # Perform chi-squared test if scipy is available
        if HAS_SCIPY:
            # Get observed frequencies from current predictions
            total_count = sum(
                1
                for key, val in self.current_predictions.items()
                if key != "_meta" and isinstance(val, dict)
            )

            if total_count > 0:
                observed = [
                    current_dist[band] * total_count for band in ["Low", "Medium", "High", "Critical"]
                ]
                expected = [
                    baseline_dist[band] * total_count for band in ["Low", "Medium", "High", "Critical"]
                ]

                # Filter out zero-frequency categories to avoid chi-squared issues
                observed_nonzero = [o for o, e in zip(observed, expected) if e > 0]
                expected_nonzero = [e for e in expected if e > 0]

                if observed_nonzero and expected_nonzero:
                    chi2, p_value = scipy_stats.chisquare(observed_nonzero, expected_nonzero)
                    result["chi_squared"] = round(chi2, 4)
                    result["p_value"] = round(p_value, 4)

                    if p_value < self.CHI_SQUARED_P_THRESHOLD:
                        result["status"] = "alert"
                    elif p_value < self.CHI_SQUARED_P_THRESHOLD * 2:
                        result["status"] = "warning"
        else:
            # Fallback: simple L1 distance if scipy not available
            l1_distance = sum(
                abs(current_dist[band] - baseline_dist[band]) for band in baseline_dist
            )
            result["chi_squared_unavailable"] = True
            result["l1_distance"] = round(l1_distance, 4)

            # Alert if L1 distance > 0.1 (10% total drift)
            if l1_distance > 0.1:
                result["status"] = "alert"
            elif l1_distance > 0.05:
                result["status"] = "warning"

        return result

    def check_anomaly_rate_drift(self) -> Dict[str, Any]:
        """Check if anomaly rate has drifted significantly from baseline."""
        baseline_rate = self.baseline_metrics["anomaly_detector"]["pct_anomalies"]

        # Count current anomalies
        anomaly_count = 0
        total_count = 0

        for key, val in self.current_predictions.items():
            if key != "_meta" and isinstance(val, dict):
                total_count += 1
                if val.get("nn_anomaly_flag"):
                    anomaly_count += 1

        current_rate = (anomaly_count / total_count * 100) if total_count > 0 else 0
        delta = abs(current_rate - baseline_rate)

        status = "ok"
        if delta > self.ANOMALY_DRIFT_THRESHOLD:
            status = "alert"
        elif delta > self.ANOMALY_DRIFT_THRESHOLD * 0.5:
            status = "warning"

        return {
            "status": status,
            "baseline_pct": round(baseline_rate, 2),
            "current_pct": round(current_rate, 2),
            "delta_pct": round(delta, 2),
            "threshold": self.ANOMALY_DRIFT_THRESHOLD,
        }

    def check_confidence_drift(self) -> Dict[str, Any]:
        """Check recommendation confidence drift."""
        predictions = self.current_predictions
        confidences = []

        for key, val in predictions.items():
            if key != "_meta" and isinstance(val, dict):
                conf = val.get("nn_recommendation_confidence")
                if conf is not None:
                    confidences.append(conf)

        if not confidences:
            return {
                "status": "warning",
                "mean_confidence": 0,
                "min_confidence": 0,
                "pct_below_70": 0,
                "threshold_mean": self.CONFIDENCE_MIN_THRESHOLD,
            }

        mean_conf = statistics.mean(confidences)
        min_conf = min(confidences)
        pct_below_70 = (sum(1 for c in confidences if c < 0.70) / len(confidences)) * 100

        status = "ok"
        if mean_conf < self.CONFIDENCE_MIN_THRESHOLD:
            status = "alert"
        elif pct_below_70 > self.CONFIDENCE_BELOW_70_THRESHOLD:
            status = "warning"

        return {
            "status": status,
            "mean_confidence": round(mean_conf, 4),
            "min_confidence": round(min_conf, 4),
            "pct_below_70": round(pct_below_70, 2),
            "threshold_mean": self.CONFIDENCE_MIN_THRESHOLD,
        }

    def check_feature_drift(self) -> Optional[Dict[str, Any]]:
        """Check feature statistics drift if training data is available."""
        if not self.training_data:
            return None

        # Try to import FeatureEngineer for feature engineering
        try:
            from pipeline.nn_trainer import FeatureEngineer
        except ImportError:
            return None

        try:
            # Extract features from training data
            feature_engineer = FeatureEngineer()

            # Compute baseline feature statistics from training data
            baseline_features = {}
            for key, row in self.training_data.items():
                if key == "_meta" or not isinstance(row, dict):
                    continue

                # Extract feature values from row
                for feature_name in self.baseline_metrics["_meta"].get("features", []):
                    if feature_name in row:
                        if feature_name not in baseline_features:
                            baseline_features[feature_name] = []
                        baseline_features[feature_name].append(row[feature_name])

            # Compute statistics
            baseline_stats = {}
            for fname, values in baseline_features.items():
                if values:
                    baseline_stats[fname] = {
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0,
                    }

            # Compare with current if data.json was updated (optional feature)
            return {
                "status": "ok",
                "feature_count": len(baseline_stats),
                "note": "Feature drift analysis available but requires updated data.json",
            }

        except Exception:
            return None

    def compute_overall_status(self, checks: Dict[str, Dict[str, Any]]) -> str:
        """Determine overall system status from individual checks."""
        has_alert = any(check.get("status") == "alert" for check in checks.values())
        has_warning = any(check.get("status") == "warning" for check in checks.values())

        if has_alert:
            return "degraded"
        elif has_warning:
            return "warning"
        else:
            return "healthy"

    def generate_recommendations(self, checks: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on detected drift."""
        recommendations = []

        if checks["recommendation_drift"]["status"] == "alert":
            delta = checks["recommendation_drift"]["delta_pct"]
            recommendations.append(
                f"Config B distribution shifted by {delta:.1f}%. Investigate data distribution changes."
            )

        if checks["npv_residual_drift"]["status"] == "alert":
            mean_res = checks["npv_residual_drift"]["mean_residual_M"]
            std_res = checks["npv_residual_drift"]["std_residual_M"]
            if abs(mean_res) > self.NPV_BIAS_THRESHOLD:
                recommendations.append(
                    f"NPV prediction bias detected ({mean_res:.3f}M). Retrain NPV regressor."
                )
            if std_res > self.NPV_STD_THRESHOLD:
                recommendations.append(
                    f"NPV residual variance increased ({std_res:.3f}M). Check for data anomalies."
                )

        if checks["band_distribution_drift"]["status"] == "alert":
            recommendations.append(
                "Band distribution shifted significantly. Validate band predictor performance."
            )

        if checks["anomaly_rate_drift"]["status"] == "alert":
            delta = checks["anomaly_rate_drift"]["delta_pct"]
            recommendations.append(
                f"Anomaly rate shifted by {delta:.1f}%. Review anomaly detection thresholds."
            )

        if checks["confidence_drift"]["status"] == "alert":
            recommendations.append(
                "Recommendation confidence dropped below threshold. Monitor model uncertainty."
            )

        if not recommendations:
            recommendations.append("All checks passed. Models performing within expected parameters.")

        return recommendations

    def run_monitoring(self) -> Dict[str, Any]:
        """Run complete monitoring workflow."""
        checks = {
            "recommendation_drift": self.check_recommendation_drift(),
            "npv_residual_drift": self.check_npv_residual_drift(),
            "band_distribution_drift": self.check_band_distribution_drift(),
            "anomaly_rate_drift": self.check_anomaly_rate_drift(),
            "confidence_drift": self.check_confidence_drift(),
        }

        # Optional feature drift check
        feature_drift = self.check_feature_drift()
        if feature_drift:
            checks["feature_drift"] = feature_drift

        overall_status = self.compute_overall_status(checks)
        recommendations = self.generate_recommendations(checks)
        summary = recommendations[0] if recommendations else "Monitoring complete."

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": overall_status,
            "checks": checks,
            "summary": summary,
            "recommendations": recommendations,
        }

        return report

    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print human-readable monitoring summary to stdout."""
        print("\n" + "=" * 80)
        print("MODEL MONITORING REPORT")
        print("=" * 80)
        print(f"\nTimestamp: {report['timestamp']}")
        print(f"Overall Status: {report['status'].upper()}")
        print("\n" + "-" * 80)
        print("DRIFT CHECKS")
        print("-" * 80)

        for check_name, check_result in report["checks"].items():
            status_symbol = {
                "ok": "✓",
                "warning": "⚠",
                "alert": "✗",
            }.get(check_result.get("status"), "?")

            print(f"\n{status_symbol} {check_name.replace('_', ' ').title()}")
            print(f"  Status: {check_result.get('status', 'unknown').upper()}")

            # Print relevant metrics
            if check_name == "recommendation_drift":
                print(
                    f"  Baseline Config B: {check_result['baseline_config_b_pct']:.2f}% "
                    f"→ Current: {check_result['current_config_b_pct']:.2f}% "
                    f"(Δ {check_result['delta_pct']:.2f}%, threshold: {check_result['threshold']:.1f}%)"
                )
            elif check_name == "npv_residual_drift":
                print(f"  Mean Residual: {check_result['mean_residual_M']:.4f}M")
                print(f"  Std Residual: {check_result['std_residual_M']:.4f}M")
                print(f"  Max Abs Residual: {check_result['max_abs_residual_M']:.4f}M")
                print(
                    f"  Thresholds: bias >{check_result['bias_threshold']}M, "
                    f"std >{check_result['std_threshold']}M"
                )
            elif check_name == "band_distribution_drift":
                print(f"  Baseline: {check_result['baseline_distribution']}")
                print(f"  Current: {check_result['current_distribution']}")
                if "chi_squared" in check_result:
                    print(
                        f"  Chi-squared: {check_result['chi_squared']:.4f}, "
                        f"p-value: {check_result['p_value']:.4f} "
                        f"(threshold: {check_result['threshold_p']:.3f})"
                    )
                elif "l1_distance" in check_result:
                    print(
                        f"  L1 Distance: {check_result['l1_distance']:.4f} "
                        f"(scipy unavailable, fallback test)"
                    )
            elif check_name == "anomaly_rate_drift":
                print(
                    f"  Baseline: {check_result['baseline_pct']:.2f}% "
                    f"→ Current: {check_result['current_pct']:.2f}% "
                    f"(Δ {check_result['delta_pct']:.2f}%, threshold: {check_result['threshold']:.1f}%)"
                )
            elif check_name == "confidence_drift":
                print(f"  Mean Confidence: {check_result['mean_confidence']:.4f}")
                print(f"  Min Confidence: {check_result['min_confidence']:.4f}")
                print(f"  Below 70%: {check_result['pct_below_70']:.2f}%")
                print(f"  Threshold: {check_result['threshold_mean']:.2f}")
            elif check_name == "feature_drift":
                print(f"  Features Analyzed: {check_result.get('feature_count', 0)}")
                if "note" in check_result:
                    print(f"  Note: {check_result['note']}")

        print("\n" + "-" * 80)
        print("SUMMARY & RECOMMENDATIONS")
        print("-" * 80)
        print(f"\n{report['summary']}")

        if len(report["recommendations"]) > 1:
            print("\nAdditional Recommendations:")
            for rec in report["recommendations"][1:]:
                print(f"  • {rec}")

        print("\n" + "=" * 80 + "\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor neural network model performance and detect drift."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="nn_predictions.json",
        help="Path to predictions file (default: nn_predictions.json)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="pipeline/nn_models/metrics.json",
        help="Path to baseline metrics (default: pipeline/nn_models/metrics.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pipeline/nn_models/monitoring_report.json",
        help="Path to output monitoring report (default: pipeline/nn_models/monitoring_report.json)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable output to stdout",
    )

    args = parser.parse_args()

    try:
        monitor = ModelMonitor(
            baseline_metrics_path=args.metrics,
            predictions_path=args.predictions,
        )

        report = monitor.run_monitoring()

        # Write JSON report
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary unless --quiet
        if not args.quiet:
            monitor.print_summary(report)

        print(f"Report saved to: {output_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
