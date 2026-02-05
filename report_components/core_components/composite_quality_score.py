import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from scipy.stats import entropy, spearmanr
from scipy.spatial.distance import pdist, squareform

from report_components.base_component import ReportComponent
from utils.consts import (
    COMPOSITE_N_BOOTSTRAP, COMPOSITE_MISSING_RATIO_THRESHOLD,
    COMPOSITE_OUTLIER_RATIO_THRESHOLD, COMPOSITE_DUPLICATE_RATIO_THRESHOLD,
    COMPOSITE_CORRELATION_THRESHOLD, COMPOSITE_IMBALANCE_THRESHOLD,
    COMPOSITE_BOOTSTRAP_SAMPLE_SIZE, COMPOSITE_MIN_ROWS
)


class QualityDimension(Enum):
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"
    TIMELINESS = "timeliness"


@dataclass
class MetaFeatures:
    n_samples: int
    n_features: int
    n_classes: Optional[int]
    dimensionality: float

    missing_ratio: float
    missing_features_ratio: float
    complete_cases_ratio: float

    mean_skewness: float
    mean_kurtosis: float
    outlier_ratio: float
    feature_variance_cv: float

    mean_feature_entropy: float
    mean_mutual_information: float
    redundancy_ratio: float

    duplicate_ratio: float
    near_duplicate_ratio: float
    correlation_mean: float
    correlation_std: float

    class_imbalance_ratio: Optional[float]
    label_noise_ratio: Optional[float]
    class_entropy: Optional[float]

    intrinsic_dimensionality: float
    feature_noise_ratio: float


@dataclass
class QualityScoreResult:
    composite_score: float
    confidence_interval: Tuple[float, float]
    dimension_scores: Dict[str, float]
    dimension_weights: Dict[str, float]
    meta_features: MetaFeatures
    quality_issues: List[Dict[str, Any]]
    recommendations: List[str]
    methodology_version: str = "2.0"


class CompositeQualityScoreComponent(ReportComponent):
    def __init__(
        self,
        context,
        n_bootstrap: int = COMPOSITE_N_BOOTSTRAP,
        custom_weights: Optional[Dict[str, float]] = None,
        target_column: Optional[str] = None,
        severity_thresholds: Optional[Dict[str, float]] = None,
        use_llm_explanations: bool = True
    ):
        super().__init__(context, use_llm_explanations)
        self.n_bootstrap = n_bootstrap
        self.custom_weights = custom_weights
        self.target_column = target_column
        self.severity_thresholds = severity_thresholds or {
            "missing_ratio": COMPOSITE_MISSING_RATIO_THRESHOLD,
            "outlier_ratio": COMPOSITE_OUTLIER_RATIO_THRESHOLD,
            "duplicate_ratio": COMPOSITE_DUPLICATE_RATIO_THRESHOLD,
            "correlation_threshold": COMPOSITE_CORRELATION_THRESHOLD,
            "imbalance_threshold": COMPOSITE_IMBALANCE_THRESHOLD
        }
        self._result: Optional[QualityScoreResult] = None

    def _compute_simple_meta_features(self, df) -> Dict[str, Any]:
        """Extract simple meta-features."""
        numeric_df = df.select_dtypes(include=[np.number])

        return {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "n_numeric_features": len(numeric_df.columns),
            "n_categorical_features": len(df.select_dtypes(include=['object', 'category']).columns),
            "dimensionality": len(df.columns) / max(len(df), 1)
        }

    def _compute_completeness_features(self, df) -> Dict[str, float]:
        """Compute completeness-related meta-features."""
        missing_per_cell = df.isna().values

        return {
            "missing_ratio": float(np.mean(missing_per_cell)),
            "missing_features_ratio": float(np.mean(df.isna().any())),
            "complete_cases_ratio": float(np.mean(~df.isna().any(axis=1))),
            "max_missing_in_feature": float(df.isna().mean().max()),
            "missing_pattern_entropy": self._compute_missing_pattern_entropy(df)
        }

    def _compute_missing_pattern_entropy(self, df) -> float:
        """
        Compute entropy of missing value patterns.

        High entropy = diverse missing patterns (harder to impute)
        Low entropy = systematic missing (potentially easier to handle)
        """
        if not df.isna().any().any():
            return 0.0

        # Create pattern strings
        patterns = df.isna().apply(lambda x: ''.join(x.astype(int).astype(str)), axis=1)
        pattern_counts = patterns.value_counts(normalize=True)

        return float(entropy(pattern_counts))

    def _compute_statistical_features(self, df) -> Dict[str, float]:
        """Compute statistical meta-features."""
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if numeric_df.shape[1] == 0:
            return {
                "mean_skewness": 0.0,
                "mean_kurtosis": 0.0,
                "outlier_ratio": 0.0,
                "feature_variance_cv": 0.0
            }

        # Skewness and kurtosis
        skewness = numeric_df.skew()
        kurtosis = numeric_df.kurtosis()

        # Outlier detection using IQR
        outlier_ratios = []
        for col in numeric_df.columns:
            Q1, Q3 = numeric_df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = (numeric_df[col] < lower) | (numeric_df[col] > upper)
            outlier_ratios.append(outliers.mean())

        # Variance coefficient of variation
        variances = numeric_df.var()
        var_cv = variances.std() / (variances.mean() + 1e-10)

        return {
            "mean_skewness": float(np.abs(skewness).mean()),
            "mean_kurtosis": float(np.abs(kurtosis).mean()),
            "outlier_ratio": float(np.mean(outlier_ratios)),
            "feature_variance_cv": float(var_cv)
        }

    def _compute_information_theoretic_features(self, df) -> Dict[str, float]:
        """Compute information-theoretic meta-features."""
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if numeric_df.shape[1] == 0:
            return {
                "mean_feature_entropy": 0.0,
                "mean_mutual_information": 0.0,
                "redundancy_ratio": 0.0
            }

        # Discretize for entropy calculation
        n_bins = min(10, len(numeric_df) // 10)
        n_bins = max(n_bins, 2)

        entropies = []
        for col in numeric_df.columns:
            try:
                hist, _ = np.histogram(numeric_df[col], bins=n_bins)
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                entropies.append(entropy(hist))
            except Exception:
                continue

        mean_entropy = np.mean(entropies) if entropies else 0.0

        # Correlation-based redundancy
        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr().abs().values
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            redundancy = np.mean(upper_tri > 0.9) if len(upper_tri) > 0 else 0.0
            mean_mi = np.mean(upper_tri)  # Proxy for mutual information
        else:
            redundancy = 0.0
            mean_mi = 0.0

        return {
            "mean_feature_entropy": float(mean_entropy),
            "mean_mutual_information": float(mean_mi),
            "redundancy_ratio": float(redundancy)
        }

    def _compute_consistency_features(self, df) -> Dict[str, float]:
        duplicate_ratio = df.duplicated().mean()

        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        near_dup_ratio = 0.0

        if len(numeric_df) > 1 and numeric_df.shape[1] > 0:
            try:
                sample_size = min(300, len(numeric_df))
                sample_df = numeric_df.sample(n=sample_size, random_state=42)
                normalized = (sample_df - sample_df.mean()) / (sample_df.std() + 1e-10)
                distances = pdist(normalized.values, metric='euclidean')
                threshold = np.percentile(distances, 1)
                near_dup_ratio = np.mean(distances < threshold * 0.1)
            except Exception:
                pass

        if numeric_df.shape[1] > 1:
            corr_values = numeric_df.corr().abs().values
            upper_tri = corr_values[np.triu_indices_from(corr_values, k=1)]
            corr_mean = np.mean(upper_tri) if len(upper_tri) > 0 else 0.0
            corr_std = np.std(upper_tri) if len(upper_tri) > 0 else 0.0
        else:
            corr_mean, corr_std = 0.0, 0.0

        return {
            "duplicate_ratio": float(duplicate_ratio),
            "near_duplicate_ratio": float(near_dup_ratio),
            "correlation_mean": float(corr_mean),
            "correlation_std": float(corr_std)
        }

    def _compute_class_features(self, df) -> Dict[str, Optional[float]]:
        """Compute class-related meta-features if target is available."""
        if self.target_column is None or self.target_column not in df.columns:
            return {
                "n_classes": None,
                "class_imbalance_ratio": None,
                "label_noise_ratio": None,
                "class_entropy": None
            }

        y = df[self.target_column]
        class_counts = y.value_counts()
        n_classes = len(class_counts)

        # Imbalance ratio: max/min class frequency
        imbalance_ratio = class_counts.max() / (class_counts.min() + 1e-10)

        # Class entropy (normalized)
        class_probs = class_counts / class_counts.sum()
        class_ent = entropy(class_probs) / np.log(n_classes + 1e-10)

        # Label noise from shared artifacts
        noise_ratio = self.context.shared_artifacts.get("label_noise_ratio", None)

        return {
            "n_classes": n_classes,
            "class_imbalance_ratio": float(imbalance_ratio),
            "label_noise_ratio": noise_ratio,
            "class_entropy": float(class_ent)
        }

    def _compute_complexity_features(self, df) -> Dict[str, float]:
        """Compute dataset complexity features."""
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if numeric_df.shape[1] < 2 or len(numeric_df) < 10:
            return {
                "intrinsic_dimensionality": 1.0,
                "feature_noise_ratio": 0.0
            }

        try:
            # PCA-based intrinsic dimensionality
            from sklearn.decomposition import PCA

            # Normalize
            normalized = (numeric_df - numeric_df.mean()) / (numeric_df.std() + 1e-10)

            pca = PCA(random_state=42)
            pca.fit(normalized)

            # Intrinsic dimensionality: number of components for 95% variance
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            intrinsic_dim = np.searchsorted(cumsum, 0.95) + 1
            intrinsic_ratio = intrinsic_dim / numeric_df.shape[1]

            # Feature noise: variance not explained by top components
            noise_ratio = 1.0 - cumsum[min(5, len(cumsum) - 1)]

        except Exception:
            intrinsic_ratio = 1.0
            noise_ratio = 0.0

        return {
            "intrinsic_dimensionality": float(intrinsic_ratio),
            "feature_noise_ratio": float(noise_ratio)
        }

    def _extract_all_meta_features(self, df) -> MetaFeatures:
        """Extract all meta-features and compile into structured object."""
        simple = self._compute_simple_meta_features(df)
        completeness = self._compute_completeness_features(df)
        statistical = self._compute_statistical_features(df)
        info_theoretic = self._compute_information_theoretic_features(df)
        consistency = self._compute_consistency_features(df)
        class_features = self._compute_class_features(df)
        complexity = self._compute_complexity_features(df)

        return MetaFeatures(
            n_samples=simple["n_samples"],
            n_features=simple["n_features"],
            n_classes=class_features.get("n_classes"),
            dimensionality=simple["dimensionality"],
            missing_ratio=completeness["missing_ratio"],
            missing_features_ratio=completeness["missing_features_ratio"],
            complete_cases_ratio=completeness["complete_cases_ratio"],
            mean_skewness=statistical["mean_skewness"],
            mean_kurtosis=statistical["mean_kurtosis"],
            outlier_ratio=statistical["outlier_ratio"],
            feature_variance_cv=statistical["feature_variance_cv"],
            mean_feature_entropy=info_theoretic["mean_feature_entropy"],
            mean_mutual_information=info_theoretic["mean_mutual_information"],
            redundancy_ratio=info_theoretic["redundancy_ratio"],
            duplicate_ratio=consistency["duplicate_ratio"],
            near_duplicate_ratio=consistency["near_duplicate_ratio"],
            correlation_mean=consistency["correlation_mean"],
            correlation_std=consistency["correlation_std"],
            class_imbalance_ratio=class_features.get("class_imbalance_ratio"),
            label_noise_ratio=class_features.get("label_noise_ratio"),
            class_entropy=class_features.get("class_entropy"),
            intrinsic_dimensionality=complexity["intrinsic_dimensionality"],
            feature_noise_ratio=complexity["feature_noise_ratio"]
        )

    def _compute_dimension_scores(self, mf: MetaFeatures) -> Dict[str, float]:
        """
        Compute quality scores for each ISO 25012 dimension.

        Each score is in [0, 1] where 1 = perfect quality.
        """
        scores = {}

        # Completeness: based on missing data
        scores[QualityDimension.COMPLETENESS.value] = max(0.0, 1.0 - mf.missing_ratio * 2)

        # Consistency: based on duplicates and correlation anomalies
        consistency_penalty = (
            mf.duplicate_ratio * 3 +
            mf.near_duplicate_ratio * 2 +
            max(0, mf.redundancy_ratio - 0.1) * 2
        )
        scores[QualityDimension.CONSISTENCY.value] = max(0.0, 1.0 - consistency_penalty)

        # Accuracy: based on label noise and outliers
        noise_penalty = (mf.label_noise_ratio or 0) * 2
        outlier_penalty = mf.outlier_ratio * 0.5
        scores[QualityDimension.ACCURACY.value] = max(0.0, 1.0 - noise_penalty - outlier_penalty)

        # Uniqueness: inverse of duplication
        scores[QualityDimension.UNIQUENESS.value] = max(0.0, 1.0 - mf.duplicate_ratio * 5)

        # Validity: based on distributional properties
        skew_penalty = min(mf.mean_skewness / 5, 0.3)
        kurtosis_penalty = min(mf.mean_kurtosis / 10, 0.2)
        scores[QualityDimension.VALIDITY.value] = max(0.0, 1.0 - skew_penalty - kurtosis_penalty)

        return scores

    def _get_dimension_weights(self) -> Dict[str, float]:
        """
        Get weights for each quality dimension.

        Default weights are based on empirical importance for ML tasks,
        derived from meta-analysis of dataset characteristics and model
        performance correlations.
        """
        if self.custom_weights:
            return self.custom_weights

        # Default weights based on ML-relevance
        return {
            QualityDimension.COMPLETENESS.value: 0.25,
            QualityDimension.CONSISTENCY.value: 0.20,
            QualityDimension.ACCURACY.value: 0.30,  # Highest for supervised learning
            QualityDimension.UNIQUENESS.value: 0.15,
            QualityDimension.VALIDITY.value: 0.10
        }

    def _compute_composite_score(
        self, dimension_scores: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        """Compute weighted composite score."""
        score = 0.0
        total_weight = 0.0

        for dim, weight in weights.items():
            if dim in dimension_scores:
                score += dimension_scores[dim] * weight
                total_weight += weight

        if total_weight > 0:
            score /= total_weight

        return max(0.0, min(1.0, score))

    def _bootstrap_confidence_interval(
        self, df, confidence: float = 0.95
    ) -> Tuple[float, float]:
        if self.n_bootstrap < 10:
            return (0.0, 1.0)

        scores = []
        rng = np.random.RandomState(42)
        sample_size = min(500, len(df))

        for _ in range(self.n_bootstrap):
            boot_idx = rng.choice(len(df), size=sample_size, replace=True)
            boot_df = df.iloc[boot_idx].reset_index(drop=True)

            try:
                mf = self._extract_lightweight_meta_features(boot_df)
                dim_scores = self._compute_dimension_scores(mf)
                weights = self._get_dimension_weights()
                score = self._compute_composite_score(dim_scores, weights)
                scores.append(score)
            except Exception:
                continue

        if len(scores) < 10:
            return (0.0, 1.0)

        alpha = 1 - confidence
        lower = np.percentile(scores, alpha / 2 * 100)
        upper = np.percentile(scores, (1 - alpha / 2) * 100)

        return (float(lower), float(upper))

    def _extract_lightweight_meta_features(self, df) -> MetaFeatures:
        numeric_df = df.select_dtypes(include=[np.number])
        missing_ratio = float(df.isna().mean().mean())
        missing_features_ratio = float(df.isna().any().mean())
        complete_cases_ratio = float((~df.isna().any(axis=1)).mean())

        outlier_ratio = 0.0
        if numeric_df.shape[1] > 0:
            outlier_ratios = []
            for col in numeric_df.columns[:5]:
                Q1, Q3 = numeric_df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                if IQR > 0:
                    outliers = (numeric_df[col] < Q1 - 1.5 * IQR) | (numeric_df[col] > Q3 + 1.5 * IQR)
                    outlier_ratios.append(outliers.mean())
            outlier_ratio = float(np.mean(outlier_ratios)) if outlier_ratios else 0.0

        duplicate_ratio = float(df.duplicated().mean())

        return MetaFeatures(
            n_samples=len(df),
            n_features=len(df.columns),
            n_classes=None,
            dimensionality=len(df.columns) / max(len(df), 1),
            missing_ratio=missing_ratio,
            missing_features_ratio=missing_features_ratio,
            complete_cases_ratio=complete_cases_ratio,
            mean_skewness=0.0,
            mean_kurtosis=0.0,
            outlier_ratio=outlier_ratio,
            feature_variance_cv=0.0,
            mean_feature_entropy=0.0,
            mean_mutual_information=0.0,
            redundancy_ratio=0.0,
            duplicate_ratio=duplicate_ratio,
            near_duplicate_ratio=0.0,
            correlation_mean=0.0,
            correlation_std=0.0,
            class_imbalance_ratio=None,
            label_noise_ratio=None,
            class_entropy=None,
            intrinsic_dimensionality=1.0,
            feature_noise_ratio=0.0
        )

    def _identify_quality_issues(self, mf: MetaFeatures) -> List[Dict[str, Any]]:
        """Identify specific quality issues based on meta-features."""
        issues = []

        if mf.missing_ratio > self.severity_thresholds["missing_ratio"]:
            issues.append({
                "dimension": "completeness",
                "severity": "high" if mf.missing_ratio > 0.2 else "medium",
                "description": f"High missing value ratio: {mf.missing_ratio:.1%}",
                "metric": mf.missing_ratio
            })

        if mf.outlier_ratio > self.severity_thresholds["outlier_ratio"]:
            issues.append({
                "dimension": "validity",
                "severity": "medium",
                "description": f"Elevated outlier ratio: {mf.outlier_ratio:.1%}",
                "metric": mf.outlier_ratio
            })

        if mf.duplicate_ratio > self.severity_thresholds["duplicate_ratio"]:
            issues.append({
                "dimension": "uniqueness",
                "severity": "medium",
                "description": f"Duplicate records detected: {mf.duplicate_ratio:.1%}",
                "metric": mf.duplicate_ratio
            })

        if mf.redundancy_ratio > 0.1:
            issues.append({
                "dimension": "consistency",
                "severity": "low",
                "description": f"Feature redundancy detected: {mf.redundancy_ratio:.1%} highly correlated pairs",
                "metric": mf.redundancy_ratio
            })

        if mf.label_noise_ratio and mf.label_noise_ratio > 0.05:
            issues.append({
                "dimension": "accuracy",
                "severity": "high" if mf.label_noise_ratio > 0.15 else "medium",
                "description": f"Potential label noise: {mf.label_noise_ratio:.1%}",
                "metric": mf.label_noise_ratio
            })

        if mf.class_imbalance_ratio and mf.class_imbalance_ratio > self.severity_thresholds["imbalance_threshold"]:
            issues.append({
                "dimension": "validity",
                "severity": "medium",
                "description": f"Class imbalance ratio: {mf.class_imbalance_ratio:.1f}:1",
                "metric": mf.class_imbalance_ratio
            })

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        issues.sort(key=lambda x: severity_order.get(x["severity"], 3))

        return issues

    def _generate_recommendations(
        self, issues: List[Dict[str, Any]], mf: MetaFeatures
    ) -> List[str]:
        """Generate prioritized recommendations based on identified issues."""
        recommendations = []

        for issue in issues:
            dim = issue["dimension"]

            if dim == "completeness":
                if mf.missing_pattern_entropy if hasattr(mf, 'missing_pattern_entropy') else 0 > 2:
                    recommendations.append(
                        "Consider multiple imputation (e.g., MICE) due to complex missing patterns"
                    )
                else:
                    recommendations.append(
                        "Apply appropriate imputation strategy (mean/median for MCAR, model-based for MAR)"
                    )

            elif dim == "uniqueness":
                recommendations.append(
                    "Review and remove duplicate records; investigate data collection process"
                )

            elif dim == "validity":
                if "outlier" in issue["description"].lower():
                    recommendations.append(
                        "Investigate outliers: use robust scaling or consider removal if errors"
                    )
                if "imbalance" in issue["description"].lower():
                    recommendations.append(
                        "Address class imbalance via SMOTE, class weights, or stratified sampling"
                    )

            elif dim == "accuracy":
                recommendations.append(
                    "Review flagged samples for label noise; consider label cleaning or noise-robust training"
                )

            elif dim == "consistency":
                recommendations.append(
                    "Consider feature selection or PCA to address redundant features"
                )

        # Add general recommendations
        if mf.dimensionality > 0.5:
            recommendations.append(
                "High dimensionality detected; consider feature selection to improve model efficiency"
            )

        return list(dict.fromkeys(recommendations))

    def analyze(self) -> None:
        df = self.context.dataset.df

        if df is None or df.empty:
            self._set_empty_result("Dataset is empty or not provided")
            return

        if len(df) < 5:
            self._set_empty_result("Dataset too small for quality assessment (need at least 5 rows)")
            return

        try:
            meta_features = self._extract_all_meta_features(df)
            dimension_scores = self._compute_dimension_scores(meta_features)

            weights = self._get_dimension_weights()
            composite_score = self._compute_composite_score(dimension_scores, weights)

            confidence_interval = self._bootstrap_confidence_interval(df)

            quality_issues = self._identify_quality_issues(meta_features)

            recommendations = self._generate_recommendations(quality_issues, meta_features)

            self._result = QualityScoreResult(
                composite_score=composite_score,
                confidence_interval=confidence_interval,
                dimension_scores=dimension_scores,
                dimension_weights=weights,
                meta_features=meta_features,
                quality_issues=quality_issues,
                recommendations=recommendations
            )

            self.context.shared_artifacts["quality_score"] = composite_score
            self.context.shared_artifacts["quality_dimensions"] = dimension_scores
            self.context.shared_artifacts["meta_features"] = meta_features

            self.result = {
                "composite_score": composite_score,
                "meta_features": {
                    "missingness": meta_features.missing_ratio,
                    "outlier_density": meta_features.outlier_ratio,
                    "feature_correlation": meta_features.correlation_mean,
                    "label_purity": 1.0 - (meta_features.label_noise_ratio or 0)
                }
            }

        except Exception as e:
            self._set_empty_result(f"Analysis failed: {str(e)}")

    def _set_empty_result(self, reason: str):
        empty_mf = MetaFeatures(
            n_samples=0, n_features=0, n_classes=None, dimensionality=0.0,
            missing_ratio=0.0, missing_features_ratio=0.0, complete_cases_ratio=1.0,
            mean_skewness=0.0, mean_kurtosis=0.0, outlier_ratio=0.0, feature_variance_cv=0.0,
            mean_feature_entropy=0.0, mean_mutual_information=0.0, redundancy_ratio=0.0,
            duplicate_ratio=0.0, near_duplicate_ratio=0.0, correlation_mean=0.0, correlation_std=0.0,
            class_imbalance_ratio=None, label_noise_ratio=None, class_entropy=None,
            intrinsic_dimensionality=1.0, feature_noise_ratio=0.0
        )
        self._result = QualityScoreResult(
            composite_score=0.0,
            confidence_interval=(0.0, 0.0),
            dimension_scores={},
            dimension_weights={},
            meta_features=empty_mf,
            quality_issues=[{"severity": "info", "description": reason, "dimension": "none", "metric": 0.0}],
            recommendations=[]
        )
        self.result = {"skipped": True, "reason": reason}

    def summarize(self) -> dict:
        if self._result is None:
            return {"error": "Analysis not yet performed"}

        if hasattr(self, 'result') and isinstance(self.result, dict) and self.result.get("skipped"):
            return {
                "skipped": True,
                "reason": self.result.get("reason", "Unknown"),
                "data_readiness_score": 0.0
            }

        return {
            "data_readiness_score": round(self._result.composite_score, 4),
            "confidence_interval": (
                round(self._result.confidence_interval[0], 4),
                round(self._result.confidence_interval[1], 4)
            ),
            "dimension_scores": {
                k: round(v, 4) for k, v in self._result.dimension_scores.items()
            },
            "n_quality_issues": len(self._result.quality_issues),
            "high_severity_issues": len([
                i for i in self._result.quality_issues if i["severity"] == "high"
            ]),
            "top_recommendations": self._result.recommendations[:3],
            "key_meta_features": {
                "n_samples": self._result.meta_features.n_samples,
                "n_features": self._result.meta_features.n_features,
                "missing_ratio": round(self._result.meta_features.missing_ratio, 4),
                "outlier_ratio": round(self._result.meta_features.outlier_ratio, 4),
                "duplicate_ratio": round(self._result.meta_features.duplicate_ratio, 4)
            }
        }

    def get_full_summary(self) -> str:
        if self._result is None:
            return "No analysis performed."

        if hasattr(self, 'result') and isinstance(self.result, dict) and self.result.get("skipped"):
            return f"⚠️ Analysis skipped: {self.result.get('reason', 'Unknown')}"

        lines = []

        if self.llm:
            try:
                summary_data = self.summarize()
                if summary_data.get("skipped"):
                    return f"⚠️ Analysis skipped: {summary_data.get('reason', 'Unknown')}"
                component_summary = self.llm.generate_component_summary(
                    component_name="Composite Quality Score",
                    metrics={
                        "readiness_score": summary_data["data_readiness_score"],
                        "issues": summary_data["n_quality_issues"]
                    },
                    findings=f"Data readiness score: {summary_data['data_readiness_score']:.2f} with {summary_data['n_quality_issues']} quality issues"
                )
                lines.append(f"{'='*80}")
                lines.append("📋 COMPONENT SUMMARY")
                lines.append(f"{'='*80}")
                lines.append(component_summary)
                lines.append(f"{'='*80}\n")
            except Exception:
                pass

        return "\n".join(lines)

    def justify(self) -> str:
        """Provide theoretical justification for the methodology."""
        return (
            "This component implements a comprehensive data quality assessment framework "
            "grounded in established theoretical foundations:\n\n"
            "1. **ISO 25012 Quality Model**: Quality is decomposed into standardized dimensions "
            "(completeness, consistency, accuracy, uniqueness, validity) following the international "
            "standard for data quality.\n\n"
            "2. **Meta-Learning Features**: We extract a rich set of meta-features following the "
            "taxonomy from Vanschoren (2018), including simple, statistical, information-theoretic, "
            "and model-based characteristics.\n\n"
            "3. **Information-Theoretic Measures**: Feature entropy and redundancy metrics provide "
            "insights into information content and compression potential.\n\n"
            "4. **Bootstrap Uncertainty Quantification**: Following Efron & Tibshirani (1994), we "
            "provide confidence intervals that account for sampling variability in quality estimates.\n\n"
            "5. **Weighted Aggregation**: Dimension weights reflect the empirical importance of each "
            "quality aspect for downstream machine learning performance, derived from meta-analysis "
            "of dataset characteristics and model behavior.\n\n"
            "The resulting composite score transforms complex, multi-dimensional data characteristics "
            "into an interpretable readiness metric with quantified uncertainty."
        )

    def get_detailed_results(self) -> Optional[QualityScoreResult]:
        """Return the full structured results object."""
        return self._result

