#!/usr/bin/env python3
"""
SSI-ENN BESS Full Ingestion Pipeline
======================================

Production-grade pipeline that fetches real SSI data, validates all 50 inputs,
computes BESS valuations (Config A & B), generates audit trail, and outputs
data.json + audit_report.json for the private BESS dashboard.

Usage:
    python3 -m pipeline.run_ingestion [--options]

    Options:
      --no-cache     Bypass cache, refetch all sources
      --strict       Fail on validation errors
      --no-bess      Skip BESS computation
      --dry-run      Simulate without writing outputs
      --verbose      Verbose logging (default: on)
      --quiet        Suppress output

Pipeline steps:
    1. Load config
    2. Fetch SSI data (4,293 substations) + grid geometry (14,221 lines)
    3. Parse substations from SSI nested format
    4. Remap modifier fields (R3_C_mult→R3, etc.)
    5. Validate all records (schema + business logic)
    6. Compute BESS valuations (Config A & B)
    7. Generate audit trail
    8. Write data.json + audit_report.json
    9. Increment version.txt (cache-buster)
   10. Print summary
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Allow running from bess-private/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import (
    OUTPUT_DIR, VERSION_FILE, BESS_CONFIGS,
    CLASSIFICATION_BANDS,
)
from pipeline.data_loader import MultiSourceLoader
from pipeline.data_validator import DataValidator
from pipeline.bess_valuator import valuate_fleet, classify_band
from pipeline.audit_trail import AuditTrail


def remap_modifiers(modifiers: dict) -> dict:
    """Remap public SSI modifier keys to BESS format."""
    if not modifiers or not isinstance(modifiers, dict):
        return {'R3': 1.0, 'R4': 1.0, 'R6': 1.0, 'R7': 1.0}
    return {
        'R3': modifiers.get('R3_C_mult', modifiers.get('R3', 1.0)),
        'R4': modifiers.get('R4_F_topo', modifiers.get('R4', 1.0)),
        'R6': (modifiers.get('R6_restoration', modifiers.get('R6', 1.0))
               * modifiers.get('R6_seismic', 1.0)),
        'R7': modifiers.get('R7_cyber', modifiers.get('R7', 1.0)),
    }


def parse_substations(ssi_data) -> list:
    """Extract substations from SSI data (handles both array and nested formats)."""
    if isinstance(ssi_data, list):
        return ssi_data
    if isinstance(ssi_data, dict):
        if 'substations' in ssi_data:
            return ssi_data['substations']
    raise ValueError(f"Unexpected SSI data format: {type(ssi_data)}")


def prepare_bess_record(sub: dict) -> dict:
    """Transform a public SSI substation into a BESS-format record."""
    return {
        'substation_id': sub.get('substation_id', ''),
        'name': sub.get('name', ''),
        'lat': sub.get('lat', 0),
        'lon': sub.get('lon', 0),
        'voltage_kv': sub.get('voltage_kv', 132),
        'region': sub.get('region', ''),
        'province': sub.get('province', ''),
        'R_median': sub.get('R_median', 0.4),
        'R_P5': sub.get('R_P5', 0.3),
        'R_P95': sub.get('R_P95', 0.6),
        'R_base_median': sub.get('R_base_median', 0.35),
        'CI_width': sub.get('CI_width', 0.2),
        'classification': sub.get('classification', classify_band(sub.get('R_median', 0.4))),
        'fleet_percentile': sub.get('fleet_percentile', 0.5),
        'components': sub.get('components', {}),
        'modifiers': remap_modifiers(sub.get('modifiers', {})),
        'socio_economic': sub.get('socio_economic', {}),
        'confidence_tier': sub.get('confidence_tier', 'medium'),
    }


def increment_version() -> int:
    """Increment version.txt and return new version number."""
    try:
        current = int(VERSION_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        current = 6  # Start after v=6 (current cache-buster)
    new_version = current + 1
    VERSION_FILE.write_text(str(new_version))
    return new_version


def update_html_cache_busters(version: int, verbose: bool = False) -> int:
    """Update ?v=N in all HTML files to match current version.txt."""
    import re
    html_files = sorted(OUTPUT_DIR.glob('*.html'))
    total_replacements = 0
    for html_file in html_files:
        content = html_file.read_text()
        new_content, count = re.subn(r'\?v=\d+', f'?v={version}', content)
        if count > 0:
            html_file.write_text(new_content)
            total_replacements += count
            if verbose:
                print(f"  {html_file.name}: {count} references → ?v={version}")
    return total_replacements


def run_pipeline(use_cache=True, strict=False, compute_bess=True,
                 dry_run=False, verbose=True):
    """Execute the full ingestion pipeline."""

    t0 = time.time()
    audit = AuditTrail(verbose=verbose)

    # ── STEP 1-2: Load data sources ──
    loader = MultiSourceLoader(use_cache=use_cache, verbose=verbose)
    results = loader.load_all_sources()

    # Check critical sources
    ssi_result = results.get('ssi_substations')
    geo_result = results.get('grid_geometry')

    if not ssi_result or not ssi_result.success:
        print("\nFATAL: Failed to load SSI substations data.")
        print(f"  Error: {ssi_result.error if ssi_result else 'No result'}")
        return 1

    if not geo_result or not geo_result.success:
        print("\nWARNING: Failed to load grid geometry. Power lines won't update.")

    # Record all fetches in audit
    audit.record_all_sources(results)

    # ── STEP 3: Parse substations ──
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 2-3: Parsing SSI substations")
        print(f"{'='*60}")

    ssi_subs = parse_substations(ssi_result.data)
    if verbose:
        print(f"  Parsed {len(ssi_subs):,} substations from SSI data")

    # ── STEP 4: Remap to BESS format ──
    substations = [prepare_bess_record(sub) for sub in ssi_subs]
    if verbose:
        regions = len(set(s['region'] for s in substations))
        provinces = len(set(s['province'] for s in substations))
        print(f"  Remapped to BESS format: {regions} regions, {provinces} provinces")

    # ── STEP 5: Validate ──
    validator = DataValidator(verbose=verbose)
    report = validator.validate_substations(substations, strict=strict)
    audit.record_validation(report)

    if strict and report.invalid_records > 0:
        print(f"\nFATAL: {report.invalid_records} records failed strict validation.")
        return 1

    # ── STEP 6: Compute BESS valuations ──
    if compute_bess:
        substations = valuate_fleet(substations, verbose=verbose)
    else:
        if verbose:
            print("\n  Skipping BESS computation (--no-bess)")

    # Record computation in audit
    audit.record_computation(substations)

    # ── STEP 7-8: Write outputs ──
    if dry_run:
        if verbose:
            print(f"\n  DRY RUN: Would write {len(substations):,} substations to data.json")
        version = 0
    else:
        if verbose:
            print(f"\n{'='*60}")
            print("STEP 8: Writing output files")
            print(f"{'='*60}")

        # Write data.json
        data_path = OUTPUT_DIR / 'data.json'
        with open(data_path, 'w') as f:
            json.dump(substations, f)
        size_mb = data_path.stat().st_size / 1024 / 1024
        if verbose:
            print(f"  data.json: {len(substations):,} records ({size_mb:.1f} MB)")

        # Write grid-geo.json (if freshly fetched)
        if geo_result and geo_result.success and geo_result.data and not geo_result.from_cache:
            geo_path = OUTPUT_DIR / 'grid-geo.json'
            with open(geo_path, 'w') as f:
                json.dump(geo_result.data, f)
            geo_mb = geo_path.stat().st_size / 1024 / 1024
            if verbose:
                print(f"  grid-geo.json: {geo_result.row_count:,} lines ({geo_mb:.1f} MB)")

        # Write audit report
        audit_report = audit.generate_report()
        audit_path = OUTPUT_DIR / 'pipeline' / 'audit_report.json'
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, 'w') as f:
            json.dump(audit_report, f, indent=2)
        if verbose:
            print(f"  audit_report.json: written")

        # Increment version
        version = increment_version()
        if verbose:
            print(f"  version.txt: v{version}")

        # Update HTML cache-busters to match new version
        replacements = update_html_cache_busters(version, verbose)
        if verbose:
            print(f"  HTML cache-busters: {replacements} references updated")

    # ── STEP 8b: Run NN inference (if models exist) ──
    models_dir = PIPELINE_DIR / 'nn_models'
    if (models_dir / 'scaler.pkl').exists() and not dry_run:
        if verbose:
            print(f"\n── Step 8b: Neural Network Inference ──")
        try:
            from pipeline.nn_inference import NeuralNetworkInference
            inferencer = NeuralNetworkInference(
                data_file=OUTPUT_DIR / 'data.json',
                models_dir=models_dir,
                verbose=False,
            )
            inferencer.score_all()
            output_path = OUTPUT_DIR / 'nn_predictions.json'
            inferencer.save_predictions(output_path)
            meta = inferencer.meta
            if verbose:
                print(f"  nn_predictions.json: {meta.get('total_scored', 0):,} substations scored")
                print(f"  Anomalies flagged:   {meta.get('total_anomalies', 0)} ({meta.get('anomaly_pct', 0):.1f}%)")
        except Exception as e:
            if verbose:
                print(f"  NN inference skipped: {e}")

    # ── STEP 8c: Run model monitoring (if predictions exist) ──
    predictions_path = OUTPUT_DIR / 'nn_predictions.json'
    if predictions_path.exists() and not dry_run:
        if verbose:
            print(f"\n── Step 8c: Model Monitoring ──")
        try:
            from pipeline.nn_monitor import ModelMonitor
            monitor = ModelMonitor(
                baseline_metrics_path=str(models_dir / 'metrics.json'),
                predictions_path=str(predictions_path),
                data_path=str(OUTPUT_DIR / 'data.json'),
            )
            report = monitor.run_monitoring()
            # Save report
            report_path = models_dir / 'monitoring_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            if verbose:
                status = report.get('status', 'unknown').upper()
                checks = report.get('checks', {})
                ok_count = sum(1 for c in checks.values() if c.get('status') == 'ok')
                print(f"  Model health: {status} ({ok_count}/{len(checks)} checks passed)")
        except Exception as e:
            if verbose:
                print(f"  Model monitoring skipped: {e}")

    # ── STEP 9: Print summary ──
    duration = time.time() - t0
    print_summary(substations, report, loader.get_summary(), version, duration)

    return 0


def print_summary(substations, validation_report, loader_summary, version, duration):
    """Print execution summary."""
    print(f"\n{'='*70}")
    print("  SSI-ENN BESS INGESTION PIPELINE — EXECUTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Status:    SUCCESS")
    print(f"  Duration:  {duration:.1f}s")
    print(f"  Version:   v{version}")
    print(f"")
    print(f"  Data Sources:")
    print(f"    Total registered: {loader_summary['total_sources']}")
    print(f"    Fetched live:     {loader_summary['successful'] - loader_summary.get('cache_hits', 0)}")
    print(f"    Cache hits:       {loader_summary.get('cache_hits', 0)}")
    print(f"    Failed:           {loader_summary['failed']}")
    print(f"")
    print(f"  Substations:  {len(substations):,}")
    print(f"  Validation:   {validation_report.valid_records:,}/{validation_report.total_records:,} valid "
          f"({validation_report.data_quality_score}% quality)")
    print(f"  Errors:       {len(validation_report.errors)}")
    print(f"  Warnings:     {len(validation_report.warnings)}")
    print(f"")

    if substations and 'bess' in substations[0]:
        import numpy as np
        band_dist = Counter(s['classification'] for s in substations)
        npv_b = [s['bess']['config_B']['NPV_M'] for s in substations]
        irr_b = [s['bess']['config_B']['IRR_pct'] for s in substations]
        regions = len(set(s['region'] for s in substations))
        provinces = len(set(s['province'] for s in substations))

        print(f"  Fleet Metrics:")
        print(f"    Regions:          {regions}")
        print(f"    Provinces:        {provinces}")
        print(f"    Median R:         {np.median([s['R_median'] for s in substations]):.3f}")
        print(f"    Median NPV (B):   {np.median(npv_b):.2f}M")
        print(f"    Median IRR (B):   {np.median(irr_b):.1f}%")
        print(f"    Total NPV (B):    {np.sum(npv_b):.1f}M")
        print(f"")
        print(f"  Band Distribution:")
        for band in ['Low', 'Medium', 'High', 'Critical']:
            count = band_dist.get(band, 0)
            pct = 100 * count / len(substations) if substations else 0
            print(f"    {band:10s} {count:5,} ({pct:5.1f}%)")

    print(f"\n  Outputs:")
    print(f"    {OUTPUT_DIR / 'data.json'}")
    print(f"    {OUTPUT_DIR / 'pipeline' / 'audit_report.json'}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='SSI-ENN BESS Full Ingestion Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--no-cache', action='store_true', help='Bypass cache')
    parser.add_argument('--strict', action='store_true', help='Fail on validation errors')
    parser.add_argument('--no-bess', action='store_true', help='Skip BESS computation')
    parser.add_argument('--dry-run', action='store_true', help='Simulate without output')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()
    verbose = not args.quiet

    return run_pipeline(
        use_cache=not args.no_cache,
        strict=args.strict,
        compute_bess=not args.no_bess,
        dry_run=args.dry_run,
        verbose=verbose,
    )


if __name__ == '__main__':
    sys.exit(main())
