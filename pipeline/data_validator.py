"""
Comprehensive schema and business logic validation for SSI-ENN BESS data.
Validates all 50 input variables across 5 sections (A, A.M, B, C, D).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import (
    VALIDATION_RULES, COMPONENT_RULES, MODIFIER_BOUNDS,
    BESS_VALIDATION, ITALY_BOUNDS, VALID_REGIONS,
    CLASSIFICATION_BANDS,
)


@dataclass
class ValidationError:
    """A single validation issue."""
    field: str
    value: Any
    rule: str
    message: str
    severity: str  # 'error' or 'warning'
    row_index: Optional[int] = None
    substation_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            'field': self.field,
            'value': str(self.value)[:100],
            'rule': self.rule,
            'message': self.message,
            'severity': self.severity,
        }
        if self.row_index is not None:
            d['row_index'] = self.row_index
        if self.substation_id:
            d['substation_id'] = self.substation_id
        return d


@dataclass
class ValidationReport:
    """Summary of all validation checks."""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    completeness: Dict[str, float] = field(default_factory=dict)
    data_quality_score: float = 100.0

    def to_dict(self) -> dict:
        return {
            'total_records': self.total_records,
            'valid_records': self.valid_records,
            'invalid_records': self.invalid_records,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'data_quality_score': round(self.data_quality_score, 2),
            'completeness': {k: round(v, 4) for k, v in self.completeness.items()},
            'errors': [e.to_dict() for e in self.errors[:50]],  # Cap at 50 for readability
            'warnings': [w.to_dict() for w in self.warnings[:50]],
        }


class DataValidator:
    """Validates substations against the full SSI-ENN schema."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def validate_substations(self, substations: list, strict: bool = False) -> ValidationReport:
        """Validate all substations. Returns ValidationReport."""
        report = ValidationReport(total_records=len(substations))
        invalid_ids = set()

        self._log(f"\n{'='*60}")
        self._log(f"STEP 4: Validating {len(substations):,} substations")
        self._log(f"{'='*60}")

        # Per-record validation
        for idx, sub in enumerate(substations):
            errors = self._validate_record(sub, idx)
            for e in errors:
                if e.severity == 'error':
                    report.errors.append(e)
                    invalid_ids.add(idx)
                else:
                    report.warnings.append(e)

        # Referential integrity
        ref_errors = self._validate_referential_integrity(substations)
        for e in ref_errors:
            if e.severity == 'error':
                report.errors.append(e)
            else:
                report.warnings.append(e)

        # Completeness
        report.completeness = self._compute_completeness(substations)

        # Compute quality score
        report.valid_records = report.total_records - len(invalid_ids)
        report.invalid_records = len(invalid_ids)
        if report.total_records > 0:
            error_rate = len(report.errors) / (report.total_records * 20)  # ~20 checks per record
            report.data_quality_score = max(0, round(100 * (1 - error_rate), 2))

        self._log(f"  Valid: {report.valid_records:,} / {report.total_records:,}")
        self._log(f"  Errors: {len(report.errors)}, Warnings: {len(report.warnings)}")
        self._log(f"  Quality score: {report.data_quality_score}%")

        return report

    def _validate_record(self, sub: dict, idx: int) -> List[ValidationError]:
        """Validate a single substation record."""
        errors = []
        sid = sub.get('substation_id', f'row_{idx}')

        # 1. Required fields & type checks
        for field_name, rules in VALIDATION_RULES.items():
            value = sub.get(field_name)

            if rules.get('required') and value is None:
                errors.append(ValidationError(
                    field=field_name, value=None, rule='required',
                    message=f'Required field missing',
                    severity='error', row_index=idx, substation_id=sid,
                ))
                continue

            if value is None:
                continue

            # Type check
            expected_type = rules.get('type')
            if expected_type and not isinstance(value, expected_type):
                # Allow int where float expected
                if expected_type == float and isinstance(value, (int, float)):
                    pass
                elif expected_type == str and not isinstance(value, str):
                    errors.append(ValidationError(
                        field=field_name, value=value, rule='type',
                        message=f'Expected {expected_type.__name__}, got {type(value).__name__}',
                        severity='error', row_index=idx, substation_id=sid,
                    ))
                    continue

            # Range check
            if isinstance(value, (int, float)):
                min_val = rules.get('min')
                max_val = rules.get('max')
                if min_val is not None and value < min_val:
                    errors.append(ValidationError(
                        field=field_name, value=value, rule='range',
                        message=f'Below minimum {min_val}',
                        severity='error', row_index=idx, substation_id=sid,
                    ))
                if max_val is not None and value > max_val:
                    errors.append(ValidationError(
                        field=field_name, value=value, rule='range',
                        message=f'Above maximum {max_val}',
                        severity='error', row_index=idx, substation_id=sid,
                    ))

        # 2. Component scores
        components = sub.get('components', {})
        if isinstance(components, dict):
            for comp, rules in COMPONENT_RULES.items():
                val = components.get(comp)
                if val is not None and isinstance(val, (int, float)):
                    if val < rules['min'] or val > rules['max']:
                        errors.append(ValidationError(
                            field=f'components.{comp}', value=val, rule='range',
                            message=f'Component {comp} out of [{rules["min"]},{rules["max"]}]',
                            severity='warning', row_index=idx, substation_id=sid,
                        ))

        # 3. Modifier bounds
        modifiers = sub.get('modifiers', {})
        if isinstance(modifiers, dict):
            for mod, bounds in MODIFIER_BOUNDS.items():
                val = modifiers.get(mod)
                if val is not None and isinstance(val, (int, float)):
                    if val < bounds['min'] or val > bounds['max']:
                        errors.append(ValidationError(
                            field=f'modifiers.{mod}', value=val, rule='range',
                            message=f'Modifier {mod} out of [{bounds["min"]},{bounds["max"]}]',
                            severity='warning', row_index=idx, substation_id=sid,
                        ))

        # 4. Classification consistency
        r_med = sub.get('R_median')
        classification = sub.get('classification')
        if r_med is not None and classification:
            expected_band = None
            for band_name, lo, hi in CLASSIFICATION_BANDS:
                if lo <= r_med < hi or (band_name == 'Critical' and r_med >= lo):
                    expected_band = band_name
                    break
            if expected_band and expected_band != classification:
                errors.append(ValidationError(
                    field='classification', value=classification, rule='consistency',
                    message=f'R_median={r_med:.3f} should be "{expected_band}", got "{classification}"',
                    severity='warning', row_index=idx, substation_id=sid,
                ))

        # 5. BESS valuation (if present)
        bess = sub.get('bess', {})
        if isinstance(bess, dict):
            for config_key in ('config_A', 'config_B'):
                cfg = bess.get(config_key, {})
                if isinstance(cfg, dict):
                    for field_name, rules in BESS_VALIDATION.items():
                        val = cfg.get(field_name)
                        if val is not None and isinstance(val, (int, float)):
                            if val < rules['min'] or val > rules['max']:
                                errors.append(ValidationError(
                                    field=f'bess.{config_key}.{field_name}',
                                    value=val, rule='range',
                                    message=f'Out of [{rules["min"]},{rules["max"]}]',
                                    severity='warning', row_index=idx, substation_id=sid,
                                ))

        return errors

    def _validate_referential_integrity(self, substations: list) -> List[ValidationError]:
        """Cross-record validation: region/province consistency."""
        errors = []
        regions_seen = set()
        region_province_map: Dict[str, set] = {}

        for idx, sub in enumerate(substations):
            region = sub.get('region', '')
            province = sub.get('province', '')
            sid = sub.get('substation_id', f'row_{idx}')
            regions_seen.add(region)

            if region:
                if region not in VALID_REGIONS:
                    errors.append(ValidationError(
                        field='region', value=region, rule='referential',
                        message=f'Unknown region "{region}"',
                        severity='warning', row_index=idx, substation_id=sid,
                    ))

                if region not in region_province_map:
                    region_province_map[region] = set()
                if province:
                    region_province_map[region].add(province)

        # Check we have at least 18 regions (expect 20)
        if len(regions_seen) < 18:
            errors.append(ValidationError(
                field='region', value=len(regions_seen), rule='completeness',
                message=f'Only {len(regions_seen)} regions found, expected 20',
                severity='warning',
            ))

        self._log(f"  Regions: {len(regions_seen)}, Provinces: {sum(len(v) for v in region_province_map.values())}")
        return errors

    def _compute_completeness(self, substations: list) -> Dict[str, float]:
        """Return % non-null for key fields."""
        if not substations:
            return {}

        fields = [
            'substation_id', 'name', 'lat', 'lon', 'voltage_kv',
            'region', 'province', 'R_median', 'R_P5', 'R_P95',
            'R_base_median', 'CI_width', 'classification',
            'fleet_percentile', 'confidence_tier',
        ]
        n = len(substations)
        completeness = {}
        for f in fields:
            present = sum(1 for s in substations if s.get(f) is not None)
            completeness[f] = present / n

        # Components
        for comp in ('C', 'V', 'I', 'E', 'S', 'T'):
            present = sum(
                1 for s in substations
                if isinstance(s.get('components'), dict) and s['components'].get(comp) is not None
            )
            completeness[f'components.{comp}'] = present / n

        # Modifiers
        for mod in ('R3', 'R4', 'R6', 'R7'):
            present = sum(
                1 for s in substations
                if isinstance(s.get('modifiers'), dict) and s['modifiers'].get(mod) is not None
            )
            completeness[f'modifiers.{mod}'] = present / n

        return completeness

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
