import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import numpy as np
import pandas as pd

from report_components.base_component import ReportComponent, AnalysisContext
from utils.consts import (
    NUM_EXAMPLES_LLM, RELATIONAL_FD_CONFIDENCE_THRESHOLD,
    RELATIONAL_MAX_FD_DETERMINANT_SIZE, RELATIONAL_MAX_VIOLATIONS
)


class ConstraintType(Enum):
    REFERENTIAL_INTEGRITY = "referential_integrity"
    FUNCTIONAL_DEPENDENCY = "functional_dependency"
    VALUE_DOMAIN = "value_domain"
    RANGE_CONSTRAINT = "range_constraint"
    PATTERN_CONSTRAINT = "pattern_constraint"
    CROSS_COLUMN = "cross_column"
    TEMPORAL = "temporal"
    UNIQUENESS = "uniqueness"
    NOT_NULL = "not_null"


class ViolationSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ConstraintViolation:
    constraint_type: ConstraintType
    description: str
    affected_columns: List[str]
    violation_count: int
    violation_ratio: float
    severity: ViolationSeverity
    example_violations: List[Dict[str, Any]]
    row_indices: List[int]
    recommendation: str


@dataclass
class FunctionalDependency:
    determinant: List[str]
    dependent: str
    confidence: float
    violation_count: int
    is_approximate: bool


@dataclass
class RelationalConsistencyResult:
    violations: List[ConstraintViolation]
    functional_dependencies: List[FunctionalDependency]
    detected_constraints: Dict[str, Any]
    overall_consistency_score: float
    column_consistency_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RelationalConsistencyComponent(ReportComponent):
    def __init__(
        self,
        context: AnalysisContext,
        check_uniqueness: bool = True,
        check_not_null: bool = True,
        check_value_domains: bool = True,
        check_patterns: bool = True,
        check_functional_dependencies: bool = True,
        check_temporal: bool = True,
        check_cross_column: bool = True,
        uniqueness_candidates: Optional[List[str]] = None,
        not_null_columns: Optional[List[str]] = None,
        value_domains: Optional[Dict[str, List[Any]]] = None,
        patterns: Optional[Dict[str, str]] = None,
        cross_column_rules: Optional[List[Dict[str, Any]]] = None,
        fd_confidence_threshold: float = RELATIONAL_FD_CONFIDENCE_THRESHOLD,
        max_fd_determinant_size: int = RELATIONAL_MAX_FD_DETERMINANT_SIZE,
        max_violations_to_report: int = RELATIONAL_MAX_VIOLATIONS,
        random_state: int = 42,
        use_llm_explanations: bool = True
    ):
        super().__init__(context, use_llm_explanations)
        self.check_uniqueness = check_uniqueness
        self.check_not_null = check_not_null
        self.check_value_domains = check_value_domains
        self.check_patterns = check_patterns
        self.check_functional_dependencies = check_functional_dependencies
        self.check_temporal = check_temporal
        self.check_cross_column = check_cross_column
        self.uniqueness_candidates = uniqueness_candidates or []
        self.not_null_columns = not_null_columns or []
        self.value_domains = value_domains or {}
        self.patterns = patterns or {}
        self.cross_column_rules = cross_column_rules or []
        self.fd_confidence_threshold = fd_confidence_threshold
        self.max_fd_determinant_size = max_fd_determinant_size
        self.max_violations_to_report = max_violations_to_report
        self.random_state = random_state
        self.llm_explanations = []

    def analyze(self):
        df = self.context.dataset.df
        if df is None or df.empty:
            self._set_empty_result("Dataset is empty or not provided")
            return

        if len(df) < 2:
            self._set_empty_result("Dataset too small for consistency analysis (need at least 2 rows)")
            return

        try:
            violations: List[ConstraintViolation] = []
            functional_dependencies: List[FunctionalDependency] = []
            detected_constraints: Dict[str, Any] = {}

            if self.check_uniqueness:
                uniqueness_violations, unique_cols = self._check_uniqueness(df)
                violations.extend(uniqueness_violations)
                detected_constraints["unique_columns"] = unique_cols

            if self.check_not_null:
                not_null_violations, null_stats = self._check_not_null(df)
                violations.extend(not_null_violations)
                detected_constraints["null_statistics"] = null_stats

            if self.check_value_domains:
                domain_violations, domain_info = self._check_value_domains(df)
                violations.extend(domain_violations)
                detected_constraints["value_domains"] = domain_info

            if self.check_patterns:
                pattern_violations, pattern_info = self._check_patterns(df)
                violations.extend(pattern_violations)
                detected_constraints["patterns"] = pattern_info

            if self.check_functional_dependencies:
                fd_violations, fds = self._analyze_functional_dependencies(df)
                violations.extend(fd_violations)
                functional_dependencies.extend(fds)
                detected_constraints["functional_dependencies"] = [
                    {
                        "determinant": fd.determinant,
                        "dependent": fd.dependent,
                        "confidence": fd.confidence,
                        "is_approximate": fd.is_approximate
                    }
                    for fd in fds
                ]

            if self.check_temporal:
                temporal_violations, temporal_info = self._check_temporal_consistency(df)
                violations.extend(temporal_violations)
                detected_constraints["temporal"] = temporal_info

            if self.check_cross_column:
                cross_violations = self._check_cross_column_rules(df)
                violations.extend(cross_violations)

            overall_score = self._compute_overall_consistency_score(violations, len(df))
            column_scores = self._compute_column_consistency_scores(violations, df.columns.tolist())

            violations_to_report = violations[:self.max_violations_to_report]
            violations_dicts = []

            for i, v in enumerate(violations_to_report):
                generate_llm = (i < NUM_EXAMPLES_LLM and self.llm)
                violations_dicts.append(self._violation_to_dict(v, generate_llm=generate_llm))

            self.result = {
                "summary": {
                    "overall_consistency_score": round(overall_score, 4),
                    "total_violations": len(violations),
                    "violations_by_severity": self._count_by_severity(violations),
                    "violations_by_type": self._count_by_type(violations),
                    "total_functional_dependencies": len(functional_dependencies)
                },
                "violations": violations_dicts,
                "functional_dependencies": [
                    {
                        "determinant": fd.determinant,
                        "dependent": fd.dependent,
                        "confidence": round(fd.confidence, 4),
                        "violation_count": fd.violation_count,
                        "is_approximate": fd.is_approximate
                    }
                    for fd in functional_dependencies
                ],
                "detected_constraints": detected_constraints,
                "column_consistency_scores": column_scores,
                "impact": self._assess_impact(violations, overall_score)
            }

            self.context.shared_artifacts["relational_violations"] = violations
            self.context.shared_artifacts["consistency_score"] = overall_score
            self.context.shared_artifacts["functional_dependencies"] = functional_dependencies

        except Exception as e:
            self._set_empty_result(f"Analysis failed: {str(e)}")

    def _set_empty_result(self, reason: str):
        self.result = {
            "summary": {
                "overall_consistency_score": 1.0,
                "total_violations": 0,
                "violations_by_severity": {},
                "violations_by_type": {},
                "total_functional_dependencies": 0,
                "skipped_reason": reason
            },
            "violations": [],
            "functional_dependencies": [],
            "detected_constraints": {},
            "column_consistency_scores": {},
            "impact": {"risk_level": "none", "ml_implications": [], "recommendations": [reason]}
        }

    def _check_uniqueness(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[ConstraintViolation], List[str]]:
        """
        Check uniqueness constraints on specified columns and auto-detect unique columns.
        """
        violations = []
        detected_unique = []

        # Check specified uniqueness candidates
        for col in self.uniqueness_candidates:
            if col not in df.columns:
                continue

            duplicates = df[df.duplicated(subset=[col], keep=False)]
            if len(duplicates) > 0:
                violation_indices = duplicates.index.tolist()
                examples = [
                    {col: val, "count": count}
                    for val, count in df[col].value_counts().items()
                    if count > 1
                ][:5]

                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.UNIQUENESS,
                    description=f"Column '{col}' expected to be unique but contains duplicates",
                    affected_columns=[col],
                    violation_count=len(duplicates),
                    violation_ratio=len(duplicates) / len(df),
                    severity=ViolationSeverity.HIGH,
                    example_violations=examples,
                    row_indices=violation_indices[:20],
                    recommendation=f"Investigate duplicate values in '{col}'. Consider deduplication or using a composite key."
                ))

        # Auto-detect columns that appear to be unique (potential keys)
        for col in df.columns:
            n_unique = df[col].nunique()
            n_rows = len(df)

            if n_unique == n_rows and n_rows > 1:
                detected_unique.append(col)

        return violations, detected_unique

    def _check_not_null(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[ConstraintViolation], Dict[str, Any]]:
        """
        Check not-null constraints and compute null statistics.
        """
        violations = []
        null_stats = {}

        for col in df.columns:
            null_count = df[col].isna().sum()
            null_ratio = null_count / len(df)
            null_stats[col] = {
                "null_count": int(null_count),
                "null_ratio": round(float(null_ratio), 4)
            }

            # Check explicit not-null constraints
            if col in self.not_null_columns and null_count > 0:
                null_indices = df[df[col].isna()].index.tolist()

                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.NOT_NULL,
                    description=f"Column '{col}' expected to be non-null but contains {null_count} null values",
                    affected_columns=[col],
                    violation_count=null_count,
                    violation_ratio=null_ratio,
                    severity=ViolationSeverity.HIGH if null_ratio > 0.1 else ViolationSeverity.MEDIUM,
                    example_violations=[{"row_index": idx} for idx in null_indices[:5]],
                    row_indices=null_indices[:20],
                    recommendation=f"Investigate missing values in '{col}'. Consider imputation or data collection improvement."
                ))

        return violations, null_stats

    def _check_value_domains(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[ConstraintViolation], Dict[str, Any]]:
        """
        Check value domain constraints and detect anomalous values.
        """
        violations = []
        domain_info = {}

        # Check explicit domain constraints
        for col, expected_values in self.value_domains.items():
            if col not in df.columns:
                continue

            actual_values = set(df[col].dropna().unique())
            expected_set = set(expected_values)
            unexpected = actual_values - expected_set

            if unexpected:
                invalid_mask = df[col].isin(unexpected)
                invalid_indices = df[invalid_mask].index.tolist()

                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.VALUE_DOMAIN,
                    description=f"Column '{col}' contains unexpected values: {list(unexpected)[:5]}",
                    affected_columns=[col],
                    violation_count=len(invalid_indices),
                    violation_ratio=len(invalid_indices) / len(df),
                    severity=ViolationSeverity.MEDIUM,
                    example_violations=[
                        {"value": val, "count": int(df[col].eq(val).sum())}
                        for val in list(unexpected)[:5]
                    ],
                    row_indices=invalid_indices[:20],
                    recommendation=f"Validate and correct unexpected values in '{col}'. Expected: {list(expected_set)[:10]}"
                ))

            domain_info[col] = {
                "expected": list(expected_set),
                "actual": list(actual_values),
                "unexpected": list(unexpected)
            }

        # Auto-detect categorical columns with potential domain issues
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col in self.value_domains:
                continue

            value_counts = df[col].value_counts(dropna=True)
            if len(value_counts) == 0:
                continue

            # Detect rare values that might be typos
            total_non_null = value_counts.sum()
            rare_threshold = max(1, total_non_null * 0.001)  # Less than 0.1% of data

            rare_values = value_counts[value_counts <= rare_threshold]
            if len(rare_values) > 0 and len(rare_values) < len(value_counts) * 0.1:
                # Only flag if rare values are a small fraction
                domain_info[col] = {
                    "detected_rare_values": [
                        {"value": str(val), "count": int(count)}
                        for val, count in rare_values.items()
                    ][:10],
                    "suggestion": "These rare values might be typos or data entry errors"
                }

        return violations, domain_info

    def _check_patterns(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[ConstraintViolation], Dict[str, Any]]:
        """
        Check pattern constraints using regex validation.
        """
        violations = []
        pattern_info = {}

        # Check explicit pattern constraints
        for col, pattern in self.patterns.items():
            if col not in df.columns:
                continue

            try:
                regex = re.compile(pattern)
            except re.error:
                continue

            non_null = df[col].dropna().astype(str)
            matches = non_null.apply(lambda x: bool(regex.match(x)))
            non_matching = non_null[~matches]

            if len(non_matching) > 0:
                invalid_indices = non_matching.index.tolist()

                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.PATTERN_CONSTRAINT,
                    description=f"Column '{col}' contains values not matching pattern '{pattern}'",
                    affected_columns=[col],
                    violation_count=len(non_matching),
                    violation_ratio=len(non_matching) / len(df),
                    severity=ViolationSeverity.MEDIUM,
                    example_violations=[
                        {"value": val}
                        for val in non_matching.head(5).tolist()
                    ],
                    row_indices=invalid_indices[:20],
                    recommendation=f"Fix values in '{col}' that don't match expected pattern: {pattern}"
                ))

            pattern_info[col] = {
                "pattern": pattern,
                "matching_ratio": round(matches.mean(), 4) if len(matches) > 0 else 1.0
            }

        # Auto-detect common patterns
        common_patterns = {
            "email": r'^[\w\.-]+@[\w\.-]+\.\w+$',
            "phone": r'^[\d\-\+\(\)\s]{7,20}$',
            "url": r'^https?://[\w\.-]+',
            "date_iso": r'^\d{4}-\d{2}-\d{2}',
            "uuid": r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
        }

        for col in df.select_dtypes(include=['object']).columns:
            if col in self.patterns:
                continue

            sample = df[col].dropna().astype(str).head(100)
            if len(sample) == 0:
                continue

            for pattern_name, pattern in common_patterns.items():
                regex = re.compile(pattern, re.IGNORECASE)
                match_ratio = sample.apply(lambda x: bool(regex.match(x))).mean()

                if match_ratio > 0.8:
                    pattern_info[col] = {
                        "detected_pattern": pattern_name,
                        "match_ratio": round(match_ratio, 4)
                    }
                    break

        return violations, pattern_info

    def _analyze_functional_dependencies(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[ConstraintViolation], List[FunctionalDependency]]:
        """
        Discover and validate functional dependencies using information theory.

        A functional dependency X -> Y holds if X uniquely determines Y.
        We detect approximate FDs where confidence >= threshold.
        """
        violations = []
        fds = []

        # Get candidate columns (non-high-cardinality)
        candidates = []
        for col in df.columns:
            n_unique = df[col].nunique()
            if n_unique < len(df) * 0.5 and n_unique > 1:  # Not unique, not constant
                candidates.append(col)

        if len(candidates) < 2:
            return violations, fds

        # Check single-column determinants
        for det_col in candidates:
            for dep_col in candidates:
                if det_col == dep_col:
                    continue

                fd = self._check_functional_dependency(df, [det_col], dep_col)
                if fd and fd.confidence >= self.fd_confidence_threshold:
                    fds.append(fd)

                    if fd.violation_count > 0:
                        violations.append(ConstraintViolation(
                            constraint_type=ConstraintType.FUNCTIONAL_DEPENDENCY,
                            description=f"Approximate FD: {det_col} -> {dep_col} has {fd.violation_count} violations",
                            affected_columns=[det_col, dep_col],
                            violation_count=fd.violation_count,
                            violation_ratio=1 - fd.confidence,
                            severity=ViolationSeverity.LOW if fd.confidence > 0.99 else ViolationSeverity.MEDIUM,
                            example_violations=self._get_fd_violation_examples(df, [det_col], dep_col),
                            row_indices=[],
                            recommendation=f"Investigate inconsistent mappings from '{det_col}' to '{dep_col}'"
                        ))

        # Check two-column determinants if enabled
        if self.max_fd_determinant_size >= 2 and len(candidates) >= 3:
            from itertools import combinations

            for det_cols in combinations(candidates, 2):
                det_list = list(det_cols)
                for dep_col in candidates:
                    if dep_col in det_list:
                        continue

                    fd = self._check_functional_dependency(df, det_list, dep_col)
                    if fd and fd.confidence >= self.fd_confidence_threshold:
                        # Only add if not subsumed by single-column FD
                        is_subsumed = any(
                            set(existing.determinant) < set(det_list) and existing.dependent == dep_col
                            for existing in fds
                        )
                        if not is_subsumed:
                            fds.append(fd)

        return violations, fds

    def _check_functional_dependency(
        self,
        df: pd.DataFrame,
        determinant: List[str],
        dependent: str
    ) -> Optional[FunctionalDependency]:
        """
        Check if determinant columns functionally determine the dependent column.
        """
        # Group by determinant and check if dependent is unique per group
        try:
            grouped = df.groupby(determinant, dropna=False)[dependent].nunique()
        except Exception:
            return None

        # FD holds perfectly if max unique values per group is 1
        violations = (grouped > 1).sum()
        total_groups = len(grouped)

        if total_groups == 0:
            return None

        confidence = 1 - (violations / total_groups)

        return FunctionalDependency(
            determinant=determinant,
            dependent=dependent,
            confidence=confidence,
            violation_count=int(violations),
            is_approximate=confidence < 1.0
        )

    def _get_fd_violation_examples(
        self,
        df: pd.DataFrame,
        determinant: List[str],
        dependent: str
    ) -> List[Dict[str, Any]]:
        """Get examples of FD violations."""
        examples = []

        try:
            grouped = df.groupby(determinant, dropna=False)[dependent]

            for name, group in grouped:
                if group.nunique() > 1:
                    examples.append({
                        "determinant_value": name if len(determinant) == 1 else list(name),
                        "dependent_values": group.unique().tolist()[:5]
                    })

                    if len(examples) >= 3:
                        break
        except Exception:
            pass

        return examples

    def _check_temporal_consistency(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[ConstraintViolation], Dict[str, Any]]:
        """
        Check temporal consistency of date/time columns.
        """
        violations = []
        temporal_info = {}

        # Identify datetime columns
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            elif df[col].dtype == 'object':
                # Try to detect date strings
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    try:
                        pd.to_datetime(sample, errors='raise')
                        datetime_cols.append(col)
                    except Exception:
                        pass

        if not datetime_cols:
            return violations, temporal_info

        # Check for future dates (might be errors)
        current_date = pd.Timestamp.now()
        for col in datetime_cols:
            try:
                dt_col = pd.to_datetime(df[col], errors='coerce')
                future_mask = dt_col > current_date
                future_count = future_mask.sum()

                if future_count > 0:
                    future_indices = df[future_mask].index.tolist()

                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.TEMPORAL,
                        description=f"Column '{col}' contains {future_count} future dates",
                        affected_columns=[col],
                        violation_count=future_count,
                        violation_ratio=future_count / len(df),
                        severity=ViolationSeverity.LOW,
                        example_violations=[
                            {"value": str(df.loc[idx, col])}
                            for idx in future_indices[:5]
                        ],
                        row_indices=future_indices[:20],
                        recommendation=f"Verify future dates in '{col}' are intentional"
                    ))

                temporal_info[col] = {
                    "min_date": str(dt_col.min()),
                    "max_date": str(dt_col.max()),
                    "null_count": int(dt_col.isna().sum()),
                    "future_count": int(future_count)
                }
            except Exception:
                pass

        # Check date pair ordering (e.g., start_date <= end_date)
        date_pairs = [
            ('start_date', 'end_date'),
            ('created_at', 'updated_at'),
            ('birth_date', 'death_date'),
            ('order_date', 'ship_date'),
            ('hire_date', 'termination_date')
        ]

        for start_col, end_col in date_pairs:
            # Try different naming conventions
            start_candidates = [c for c in datetime_cols if start_col.replace('_', '') in c.lower().replace('_', '')]
            end_candidates = [c for c in datetime_cols if end_col.replace('_', '') in c.lower().replace('_', '')]

            for sc in start_candidates:
                for ec in end_candidates:
                    if sc == ec:
                        continue

                    try:
                        start_dt = pd.to_datetime(df[sc], errors='coerce')
                        end_dt = pd.to_datetime(df[ec], errors='coerce')

                        invalid_mask = (start_dt > end_dt) & start_dt.notna() & end_dt.notna()
                        invalid_count = invalid_mask.sum()

                        if invalid_count > 0:
                            invalid_indices = df[invalid_mask].index.tolist()

                            violations.append(ConstraintViolation(
                                constraint_type=ConstraintType.TEMPORAL,
                                description=f"'{sc}' > '{ec}' in {invalid_count} rows (expected start <= end)",
                                affected_columns=[sc, ec],
                                violation_count=invalid_count,
                                violation_ratio=invalid_count / len(df),
                                severity=ViolationSeverity.HIGH,
                                example_violations=[
                                    {sc: str(df.loc[idx, sc]), ec: str(df.loc[idx, ec])}
                                    for idx in invalid_indices[:5]
                                ],
                                row_indices=invalid_indices[:20],
                                recommendation=f"Fix temporal ordering: '{sc}' should not be after '{ec}'"
                            ))
                    except Exception:
                        pass

        return violations, temporal_info

    def _check_cross_column_rules(
        self,
        df: pd.DataFrame
    ) -> List[ConstraintViolation]:
        """
        Check custom cross-column rules and auto-detect common patterns.
        """
        violations = []

        # Check explicit cross-column rules
        for rule in self.cross_column_rules:
            rule_type = rule.get('type')
            columns = rule.get('columns', [])

            if not all(c in df.columns for c in columns):
                continue

            if rule_type == 'comparison':
                # e.g., {"type": "comparison", "columns": ["a", "b"], "operator": "<="}
                violations.extend(self._check_comparison_rule(df, rule))

            elif rule_type == 'sum':
                # e.g., {"type": "sum", "columns": ["a", "b"], "total_column": "c"}
                violations.extend(self._check_sum_rule(df, rule))

        # Auto-detect numeric relationship violations
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Check for potential percentage columns that exceed bounds
        for col in numeric_cols:
            col_lower = col.lower()
            if any(term in col_lower for term in ['percent', 'pct', 'ratio', 'rate']):
                out_of_range = (df[col] < 0) | (df[col] > 100 if 'percent' in col_lower else df[col] > 1)
                out_of_range = out_of_range & df[col].notna()

                if out_of_range.sum() > 0:
                    invalid_indices = df[out_of_range].index.tolist()

                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.RANGE_CONSTRAINT,
                        description=f"Column '{col}' appears to be a percentage/ratio but has out-of-range values",
                        affected_columns=[col],
                        violation_count=int(out_of_range.sum()),
                        violation_ratio=out_of_range.sum() / len(df),
                        severity=ViolationSeverity.MEDIUM,
                        example_violations=[
                            {"value": float(df.loc[idx, col])}
                            for idx in invalid_indices[:5]
                        ],
                        row_indices=invalid_indices[:20],
                        recommendation=f"Verify values in '{col}' - expected to be in valid percentage/ratio range"
                    ))

        # Check for negative values in columns that should be non-negative
        non_negative_hints = ['count', 'quantity', 'qty', 'amount', 'price', 'cost', 'age', 'size']
        for col in numeric_cols:
            col_lower = col.lower()
            if any(hint in col_lower for hint in non_negative_hints):
                negative_mask = df[col] < 0
                if negative_mask.sum() > 0:
                    negative_indices = df[negative_mask].index.tolist()

                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.RANGE_CONSTRAINT,
                        description=f"Column '{col}' contains negative values but appears to require non-negative values",
                        affected_columns=[col],
                        violation_count=int(negative_mask.sum()),
                        violation_ratio=negative_mask.sum() / len(df),
                        severity=ViolationSeverity.MEDIUM,
                        example_violations=[
                            {"value": float(df.loc[idx, col])}
                            for idx in negative_indices[:5]
                        ],
                        row_indices=negative_indices[:20],
                        recommendation=f"Investigate negative values in '{col}'"
                    ))

        return violations

    def _check_comparison_rule(
        self,
        df: pd.DataFrame,
        rule: Dict[str, Any]
    ) -> List[ConstraintViolation]:
        """Check a comparison rule between two columns."""
        violations = []
        columns = rule.get('columns', [])
        operator = rule.get('operator', '<=')

        if len(columns) != 2:
            return violations

        col_a, col_b = columns

        try:
            if operator == '<=':
                invalid_mask = df[col_a] > df[col_b]
            elif operator == '<':
                invalid_mask = df[col_a] >= df[col_b]
            elif operator == '>=':
                invalid_mask = df[col_a] < df[col_b]
            elif operator == '>':
                invalid_mask = df[col_a] <= df[col_b]
            elif operator == '==':
                invalid_mask = df[col_a] != df[col_b]
            else:
                return violations

            invalid_mask = invalid_mask & df[col_a].notna() & df[col_b].notna()

            if invalid_mask.sum() > 0:
                invalid_indices = df[invalid_mask].index.tolist()

                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.CROSS_COLUMN,
                    description=f"Constraint '{col_a}' {operator} '{col_b}' violated in {invalid_mask.sum()} rows",
                    affected_columns=columns,
                    violation_count=int(invalid_mask.sum()),
                    violation_ratio=invalid_mask.sum() / len(df),
                    severity=ViolationSeverity.HIGH,
                    example_violations=[
                        {col_a: df.loc[idx, col_a], col_b: df.loc[idx, col_b]}
                        for idx in invalid_indices[:5]
                    ],
                    row_indices=invalid_indices[:20],
                    recommendation=f"Fix values violating constraint: {col_a} {operator} {col_b}"
                ))
        except Exception:
            pass

        return violations

    def _check_sum_rule(
        self,
        df: pd.DataFrame,
        rule: Dict[str, Any]
    ) -> List[ConstraintViolation]:
        """Check that columns sum to a total column."""
        violations = []
        columns = rule.get('columns', [])
        total_col = rule.get('total_column')
        tolerance = rule.get('tolerance', 0.01)

        if not total_col or total_col not in df.columns:
            return violations

        try:
            computed_sum = df[columns].sum(axis=1)
            actual_total = df[total_col]

            diff = abs(computed_sum - actual_total)
            invalid_mask = diff > tolerance * abs(actual_total).clip(lower=1)
            invalid_mask = invalid_mask & actual_total.notna()

            if invalid_mask.sum() > 0:
                invalid_indices = df[invalid_mask].index.tolist()

                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.CROSS_COLUMN,
                    description=f"Sum of {columns} doesn't match '{total_col}' in {invalid_mask.sum()} rows",
                    affected_columns=columns + [total_col],
                    violation_count=int(invalid_mask.sum()),
                    violation_ratio=invalid_mask.sum() / len(df),
                    severity=ViolationSeverity.HIGH,
                    example_violations=[
                        {
                            "computed_sum": float(computed_sum.loc[idx]),
                            "actual_total": float(actual_total.loc[idx])
                        }
                        for idx in invalid_indices[:5]
                    ],
                    row_indices=invalid_indices[:20],
                    recommendation=f"Verify sum computation: {' + '.join(columns)} should equal {total_col}"
                ))
        except Exception:
            pass

        return violations

    def _compute_overall_consistency_score(
        self,
        violations: List[ConstraintViolation],
        n_rows: int
    ) -> float:
        """
        Compute overall consistency score from 0 to 1.

        Higher score = better consistency (fewer violations).
        """
        if not violations or n_rows == 0:
            return 1.0

        # Weight violations by severity
        severity_weights = {
            ViolationSeverity.CRITICAL: 1.0,
            ViolationSeverity.HIGH: 0.8,
            ViolationSeverity.MEDIUM: 0.5,
            ViolationSeverity.LOW: 0.2,
            ViolationSeverity.INFO: 0.1
        }

        weighted_violation_ratio = 0.0
        for v in violations:
            weight = severity_weights.get(v.severity, 0.5)
            weighted_violation_ratio += v.violation_ratio * weight

        # Normalize to [0, 1] and invert (higher = better)
        # Cap at 1 for ratio sum
        consistency = max(0.0, 1.0 - min(weighted_violation_ratio, 1.0))
        return consistency

    def _compute_column_consistency_scores(
        self,
        violations: List[ConstraintViolation],
        columns: List[str]
    ) -> Dict[str, float]:
        """Compute per-column consistency scores."""
        scores = {col: 1.0 for col in columns}

        for v in violations:
            for col in v.affected_columns:
                if col in scores:
                    # Reduce score based on violation ratio
                    penalty = v.violation_ratio * 0.5
                    scores[col] = max(0.0, scores[col] - penalty)

        return {col: round(score, 4) for col, score in scores.items()}

    def _count_by_severity(
        self,
        violations: List[ConstraintViolation]
    ) -> Dict[str, int]:
        """Count violations by severity level."""
        counts = defaultdict(int)
        for v in violations:
            counts[v.severity.value] += 1
        return dict(counts)

    def _count_by_type(
        self,
        violations: List[ConstraintViolation]
    ) -> Dict[str, int]:
        """Count violations by constraint type."""
        counts = defaultdict(int)
        for v in violations:
            counts[v.constraint_type.value] += 1
        return dict(counts)

    def _violation_to_dict(self, v: ConstraintViolation, generate_llm: bool = False) -> Dict[str, Any]:
        result = {
            "constraint_type": v.constraint_type.value,
            "description": v.description,
            "affected_columns": v.affected_columns,
            "violation_count": v.violation_count,
            "violation_ratio": round(v.violation_ratio, 4),
            "severity": v.severity.value,
            "example_violations": v.example_violations,
            "row_indices": v.row_indices,
            "recommendation": v.recommendation
        }


        return result

    def _assess_impact(
        self,
        violations: List[ConstraintViolation],
        consistency_score: float
    ) -> Dict[str, Any]:
        """Assess the impact of consistency violations."""
        if not violations:
            return {
                "risk_level": "none",
                "ml_implications": [],
                "recommendations": ["Data passes all consistency checks"]
            }

        # Determine risk level
        critical_count = sum(1 for v in violations if v.severity == ViolationSeverity.CRITICAL)
        high_count = sum(1 for v in violations if v.severity == ViolationSeverity.HIGH)

        if critical_count > 0 or consistency_score < 0.5:
            risk_level = "critical"
        elif high_count > 2 or consistency_score < 0.7:
            risk_level = "high"
        elif consistency_score < 0.85:
            risk_level = "medium"
        else:
            risk_level = "low"

        implications = [
            "Inconsistent data can lead to unreliable model predictions",
            "Violated constraints may indicate data pipeline issues",
            "Cross-column inconsistencies can cause feature engineering errors"
        ]

        if risk_level in ["critical", "high"]:
            implications.append("Data cleaning is strongly recommended before model training")

        # Generate recommendations
        recommendations = []
        types_found = set(v.constraint_type for v in violations)

        if ConstraintType.UNIQUENESS in types_found:
            recommendations.append("Resolve uniqueness violations - consider composite keys or deduplication")

        if ConstraintType.NOT_NULL in types_found:
            recommendations.append("Address missing values through imputation or data collection")

        if ConstraintType.FUNCTIONAL_DEPENDENCY in types_found:
            recommendations.append("Investigate functional dependency violations for data entry errors")

        if ConstraintType.TEMPORAL in types_found:
            recommendations.append("Fix temporal inconsistencies - verify date logic and data sources")

        if not recommendations:
            recommendations.append("Review minor violations and determine if they impact your use case")

        return {
            "risk_level": risk_level,
            "consistency_score": round(consistency_score, 4),
            "ml_implications": implications,
            "recommendations": recommendations
        }

    def summarize(self) -> dict:
        if self.result is None:
            raise RuntimeError("analyze() must be called before summarize()")

        if "skipped_reason" in self.result["summary"]:
            return {
                "skipped": True,
                "reason": self.result["summary"]["skipped_reason"],
                "overall_consistency_score": 1.0
            }

        clean_violations = []
        for v in self.result["violations"][:5]:
            clean_v = {k: val for k, val in v.items() if k != 'llm_explanation'}
            clean_violations.append(clean_v)

        summary = {
            "overall_consistency_score": self.result["summary"]["overall_consistency_score"],
            "total_violations": self.result["summary"]["total_violations"],
            "violations_by_severity": self.result["summary"]["violations_by_severity"],
            "total_functional_dependencies": self.result["summary"]["total_functional_dependencies"],
            "risk_level": self.result["impact"]["risk_level"],
            "example_violations": clean_violations
        }

        return summary

    def get_full_summary(self) -> str:
        if self.result is None:
            raise RuntimeError("analyze() must be called before get_full_summary()")

        if "skipped_reason" in self.result["summary"]:
            return f"⚠️ Analysis skipped: {self.result['summary']['skipped_reason']}"

        lines = []

        if self.llm:
            try:
                summary_data = self.summarize()
                component_summary = self.llm.generate_component_summary(
                    component_name="Relational Consistency",
                    metrics={
                        "consistency_score": summary_data["overall_consistency_score"],
                        "violations": summary_data["total_violations"]
                    },
                    findings=f"Found {summary_data['total_violations']} consistency violations, score: {summary_data['overall_consistency_score']:.2f}"
                )
                if component_summary:
                    lines.append(f"{'='*80}")
                    lines.append("📋 COMPONENT SUMMARY")
                    lines.append(f"{'='*80}")
                    lines.append(component_summary)
                    lines.append(f"{'='*80}\n")
            except Exception:
                pass

        return "\n".join(lines)

    def justify(self) -> str:
        return (
            "Relational consistency analysis validates data integrity through symbolic "
            "constraint checking and statistical dependency discovery. Detects violations "
            "in uniqueness, not-null, domains, patterns, and functional dependencies."
        )
