"""
Provenance and lineage tracking for the ingestion pipeline.
Records source fetches, validation results, BESS computations,
and generates a comprehensive audit report.
"""

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DATA_SOURCES, BESS_CONFIGS, WACC, HORIZON


class AuditTrail:
    """Orchestrates comprehensive audit logging for the pipeline."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = datetime.utcnow()
        self.source_records: List[dict] = []
        self.validation_record: Optional[dict] = None
        self.computation_summary: Optional[dict] = None
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def record_source_fetch(self, result) -> None:
        """Log completion of a data source fetch."""
        self.source_records.append(result.to_dict())

    def record_all_sources(self, results: dict) -> None:
        """Log all data source fetch results."""
        for name, result in results.items():
            self.source_records.append(result.to_dict())

    def record_validation(self, report) -> None:
        """Log validation results."""
        self.validation_record = report.to_dict()

    def record_computation(self, substations: list) -> None:
        """Log BESS computation summary statistics."""
        import numpy as np

        if not substations:
            self.computation_summary = {'total': 0}
            return

        npv_a = [s['bess']['config_A']['NPV_M'] for s in substations if 'bess' in s]
        npv_b = [s['bess']['config_B']['NPV_M'] for s in substations if 'bess' in s]
        irr_b = [s['bess']['config_B']['IRR_pct'] for s in substations if 'bess' in s]
        rec_b = sum(1 for s in substations if s.get('bess', {}).get('recommendation') == 'Config B')

        from collections import Counter
        band_dist = dict(Counter(s.get('classification', 'Unknown') for s in substations))
        region_dist = dict(Counter(s.get('region', 'Unknown') for s in substations))
        prio_dist = dict(Counter(s.get('bess', {}).get('investment_priority', 0) for s in substations))

        self.computation_summary = {
            'total_computed': len(substations),
            'config_a_median_npv': round(float(np.median(npv_a)), 2) if npv_a else 0,
            'config_b_median_npv': round(float(np.median(npv_b)), 2) if npv_b else 0,
            'config_b_median_irr': round(float(np.median(irr_b)), 1) if irr_b else 0,
            'config_b_total_npv': round(float(np.sum(npv_b)), 2) if npv_b else 0,
            'config_b_recommended_pct': round(100 * rec_b / len(substations), 1) if substations else 0,
            'band_distribution': band_dist,
            'region_distribution': region_dist,
            'priority_distribution': {str(k): v for k, v in sorted(prio_dist.items())},
            'seed_base': 42,
        }

    def generate_report(self) -> dict:
        """Generate the comprehensive audit report JSON."""
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()

        # Git commit (best effort)
        git_commit = self._get_git_commit()

        # Config hash
        config_str = json.dumps({
            'bess_configs': BESS_CONFIGS,
            'wacc': WACC,
            'horizon': HORIZON,
        }, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        report = {
            'execution_metadata': {
                'pipeline_version': '1.0.0',
                'start_time': self.start_time.isoformat() + 'Z',
                'end_time': end_time.isoformat() + 'Z',
                'duration_sec': round(duration, 1),
                'python_version': platform.python_version(),
                'platform': platform.platform(),
            },
            'data_sources': {
                'total_registered': len(DATA_SOURCES),
                'fetched_live': sum(
                    1 for r in self.source_records
                    if r.get('hash_sha256') and r['hash_sha256'] != 'embedded' and r['success']
                ),
                'embedded': sum(
                    1 for r in self.source_records
                    if r.get('hash_sha256') == 'embedded'
                ),
                'failed': sum(1 for r in self.source_records if not r['success']),
                'per_source': self.source_records,
            },
            'validation': self.validation_record or {},
            'bess_computation': self.computation_summary or {},
            'reproducibility': {
                'git_commit': git_commit,
                'config_hash': config_hash,
                'numpy_seed': 42,
                'deterministic': True,
            },
            'data_source_registry': self._build_source_registry(),
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print("STEP 7: Audit report generated")
            print(f"{'='*60}")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Sources: {report['data_sources']['fetched_live']} live, "
                  f"{report['data_sources']['embedded']} embedded")
            if self.validation_record:
                print(f"  Validation: {self.validation_record.get('valid_records', 0)}/"
                      f"{self.validation_record.get('total_records', 0)} valid")
            if self.computation_summary:
                print(f"  BESS: {self.computation_summary.get('total_computed', 0)} substations computed")

        return report

    def _build_source_registry(self) -> list:
        """Build summary of all 28 data source providers."""
        registry = []
        for name, src in DATA_SOURCES.items():
            entry = {
                'name': name,
                'section': src.get('section', ''),
                'provider': src.get('provider', ''),
                'status': src.get('status', 'live'),
            }
            if 'url' in src:
                entry['url'] = src['url']
            if 'variables' in src:
                entry['variables'] = src['variables']
            registry.append(entry)
        return registry

    def _get_git_commit(self) -> str:
        """Get current git commit hash (best effort)."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return 'unknown'
