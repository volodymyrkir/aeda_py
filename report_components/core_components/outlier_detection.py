from typing import Dict, Any, List, Optional

import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from report_components.base_component import ReportComponent, AnalysisContext


class OutlierDetectionComponent(ReportComponent):
    """
    Learning-based multivariate outlier detection with configurable parameters,
    local explanations, and optional AI-enhanced narratives.
    Supports missing value handling and advanced interpretability via SHAP.
    Intelligently skips high-cardinality numeric columns to avoid false positives.
    """

    def __init__(
            self,
            context: AnalysisContext,
            n_estimators: int = 200,
            contamination: str = "auto",
            threshold_percentile: Optional[float] = None,
            max_explain_features: int = 5,
            impute_missing: bool = True,
            use_shap: bool = True,
            random_state: int = 42,
            skip_high_cardinality: bool = True,
            cardinality_threshold: float = 0.85,  # Skip if unique_ratio > this (e.g., IDs)
            min_unique_for_id_heuristic: int = 50  # Avoid skipping low-unique columns
    ):
        super().__init__(context)
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.threshold_percentile = threshold_percentile
        self.max_explain_features = max_explain_features
        self.impute_missing = impute_missing
        self.use_shap = use_shap
        self.random_state = random_state
        self.skip_high_cardinality = skip_high_cardinality
        self.cardinality_threshold = cardinality_threshold
        self.min_unique_for_id_heuristic = min_unique_for_id_heuristic

    def analyze(self):
        df = self.context.dataset.df
        if df is None or df.empty:
            raise ValueError("Dataset is empty or not provided")

        candidate_numeric_cols = self.context.shared_artifacts.get(
            "numeric_columns",
            list(df.select_dtypes(include=[np.number]).columns)
        )

        if not candidate_numeric_cols:
            raise ValueError("No numeric columns available for outlier detection")

        # Filter out high-cardinality columns (e.g., IDs)
        n_rows = len(df)
        numeric_cols = []
        skipped_cols = []

        for col in candidate_numeric_cols:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            unique_count = non_null.nunique()
            unique_ratio = unique_count / len(non_null)

            if (self.skip_high_cardinality and
                unique_ratio > self.cardinality_threshold and
                unique_count >= self.min_unique_for_id_heuristic):
                skipped_cols.append({
                    "column": col,
                    "unique_ratio": round(unique_ratio, 4),
                    "unique_count": int(unique_count)
                })
                continue

            numeric_cols.append(col)

        if not numeric_cols:
            raise ValueError("No suitable numeric columns remain after filtering high-cardinality columns")

        X = df[numeric_cols].copy()

        if self.impute_missing:
            imputer = SimpleImputer(strategy="mean")
            X = pd.DataFrame(imputer.fit_transform(X), columns=numeric_cols)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )
        model.fit(X_scaled)

        scores = -model.score_samples(X_scaled)

        if self.threshold_percentile is not None:
            threshold = np.percentile(scores, self.threshold_percentile)
            outlier_mask = scores >= threshold
        else:
            predictions = model.predict(X_scaled)
            outlier_mask = predictions == -1
            threshold = model.offset_

        explanations = self._explain_outliers(
            model, X_scaled, scores, outlier_mask, numeric_cols
        )
        self.result = {
            "summary": {
                "outlier_ratio": round(float(outlier_mask.mean()), 5),
                "threshold": float(threshold),
                "model_params": {
                    "n_estimators": self.n_estimators,
                    "contamination": self.contamination
                },
                "skipped_high_cardinality_columns": skipped_cols,
                "used_numeric_columns": numeric_cols
            },
            "outliers": explanations
        }
        self.context.shared_artifacts["outlier_scores"] = scores
        self.context.shared_artifacts["outlier_mask"] = outlier_mask
        self.context.shared_artifacts["skipped_numeric_columns"] = skipped_cols  # For downstream use

    def _explain_outliers(
            self,
            model: IsolationForest,
            X_scaled: np.ndarray,
            scores: np.ndarray,
            mask: np.ndarray,
            feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        outlier_indices = np.where(mask)[0]
        explanations = []

        if self.use_shap:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled[outlier_indices])
            if shap_values.ndim == 3:
                shap_values = np.mean(shap_values, axis=0)

        mean_vector = np.mean(X_scaled, axis=0)

        for i, idx in enumerate(outlier_indices):
            if self.use_shap:
                deviation = np.abs(shap_values[i])
            else:
                deviation = np.abs(X_scaled[idx] - mean_vector)

            top_indices = np.argsort(deviation)[::-1][:self.max_explain_features]
            feature_contributions = {
                feature_names[j]: round(float(deviation[j]), 4) for j in top_indices
            }

            narrative = self._generate_narrative(
                scores[idx], feature_contributions, feature_names
            )

            explanations.append({
                "row_index": int(idx),
                "outlier_score": round(float(scores[idx]), 5),
                "top_contributing_features": feature_contributions,
                "explanation_narrative": narrative
            })

        return explanations

    def _generate_narrative(
            self,
            score: float,
            contributions: Dict[str, float],
            feature_names: List[str]
    ) -> str:
        top_features = list(contributions.keys())
        narrative = f"This record has an anomaly score of {score:.5f}, indicating deviation from the data distribution. "
        narrative += "Key contributors include: "
        narrative += ", ".join([f"{feat} (deviation {contributions[feat]:.4f})" for feat in top_features])
        narrative += ". This suggests potential issues like measurement errors or rare events in these dimensions."
        return narrative

    def summarize(self) -> dict:
        if self.result is None:
            raise RuntimeError("analyze() must be called before summarize()")
        return {
            "outlier_ratio": self.result["summary"]["outlier_ratio"],
            "example_outliers": self.result["outliers"][:3],
            "model_params": self.result["summary"]["model_params"],
            "skipped_columns": self.result["summary"]["skipped_high_cardinality_columns"]
        }

    def justify(self) -> str:
        method = "SHAP values" if self.use_shap else "feature deviations from the mean"
        return (
                "This component employs Isolation Forest, an unsupervised ensemble method that isolates anomalies via random partitioning, "
                "capturing complex multivariate interactions and non-linear density structures. It enhances autonomy by handling missing values, "
                "skipping high-cardinality columns to prevent false positives, and providing configurable parameters for flexibility. "
                "Local explanations use " + method + " to quantify feature contributions, "
                "with AI-generated narratives for interpretable insights. This approach outperforms heuristic methods in detecting subtle outliers, "
                "making it suitable for professional dataset analysis in research contexts."
        )