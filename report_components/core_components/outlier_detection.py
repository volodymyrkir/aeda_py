from typing import Dict, Any, List, Optional
import html

import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from report_components.base_component import ReportComponent, AnalysisContext
from utils.consts import (
    NUM_EXAMPLES_LLM, OUTLIER_N_ESTIMATORS, OUTLIER_CONTAMINATION,
    OUTLIER_CARDINALITY_THRESHOLD, OUTLIER_MIN_UNIQUE_FOR_ID, OUTLIER_MAX_EXPLAIN_FEATURES
)


class OutlierDetectionComponent(ReportComponent):
    def __init__(
            self,
            context: AnalysisContext,
            n_estimators: int = OUTLIER_N_ESTIMATORS,
            contamination: str = OUTLIER_CONTAMINATION,
            threshold_percentile: Optional[float] = None,
            max_explain_features: int = OUTLIER_MAX_EXPLAIN_FEATURES,
            impute_missing: bool = True,
            use_shap: bool = True,
            random_state: int = 42,
            skip_high_cardinality: bool = True,
            cardinality_threshold: float = OUTLIER_CARDINALITY_THRESHOLD,
            min_unique_for_id_heuristic: int = OUTLIER_MIN_UNIQUE_FOR_ID,
            use_llm_explanations: bool = True
    ):
        super().__init__(context, use_llm_explanations)
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
        self.llm_explanations = []

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
        df = self.context.dataset.df
        outlier_indices = np.where(mask)[0]
        explanations = []

        shap_values = None
        use_shap_for_explanation = self.use_shap

        if self.use_shap and len(outlier_indices) > 0:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled[outlier_indices])
                if shap_values.ndim == 3:
                    shap_values = np.mean(shap_values, axis=0)
            except (IndexError, ValueError, Exception):
                use_shap_for_explanation = False
                shap_values = None

        mean_vector = np.mean(X_scaled, axis=0)

        for i, idx in enumerate(outlier_indices):
            if use_shap_for_explanation and shap_values is not None:
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

            explanation_entry = {
                "row_index": int(idx),
                "outlier_score": round(float(scores[idx]), 5),
                "top_contributing_features": feature_contributions,
                "explanation_narrative": narrative
            }

            if i < NUM_EXAMPLES_LLM and self.llm:
                try:
                    row_data = df.iloc[idx].to_dict()
                    llm_explanation = self.llm.explain_outlier(
                        row_data=row_data,
                        outlier_score=scores[idx],
                        contributing_features=feature_contributions,
                        dataset_context=f"Dataset has {len(df)} rows"
                    )
                    explanation_entry["llm_explanation"] = llm_explanation
                    self.llm_explanations.append({
                        "row_index": int(idx),
                        "explanation": llm_explanation
                    })
                except Exception:
                    pass

            explanations.append(explanation_entry)

        return explanations

    def _generate_narrative(
            self,
            score: float,
            contributions: Dict[str, float],
            feature_names: List[str]
    ) -> str:
        top_features = list(contributions.keys())
        narrative = f"Anomaly score: {score:.5f}. Key contributors: "
        narrative += ", ".join([f"{feat} ({contributions[feat]:.4f})" for feat in top_features])
        return narrative

    def summarize(self) -> dict:
        if self.result is None:
            raise RuntimeError("analyze() must be called before summarize()")

        example_outliers = []
        for outlier in self.result["outliers"][:3]:
            clean_outlier = {k: v for k, v in outlier.items() if k != 'llm_explanation'}
            example_outliers.append(clean_outlier)

        summary = {
            "outlier_ratio": self.result["summary"]["outlier_ratio"],
            "example_outliers": example_outliers,
            "model_params": self.result["summary"]["model_params"],
            "skipped_columns": self.result["summary"]["skipped_high_cardinality_columns"]
        }

        return summary

    def get_full_summary(self) -> str:
        if self.result is None:
            raise RuntimeError("analyze() must be called before get_full_summary()")

        lines = []
        df = self.context.dataset.df

        if self.llm_explanations:
            cards_html = []
            for expl in self.llm_explanations:
                row_idx = expl['row_index']
                row_data = df.iloc[row_idx].to_dict() if row_idx < len(df) else {}

                outlier_entry = next((o for o in self.result["outliers"] if o["row_index"] == row_idx), {})
                top_features = outlier_entry.get("top_contributing_features", {})
                outlier_score = outlier_entry.get("outlier_score", 0)

                feature_rows = []
                for col, val in list(row_data.items())[:10]:
                    col_lower = col.lower()
                    is_id = 'id' in col_lower or 'name' in col_lower or df[col].nunique() / len(df) > 0.9
                    if is_id:
                        continue
                    highlight = col in top_features
                    style = " style='background:#fef2f2;'" if highlight else ""
                    escaped_col = html.escape(str(col))
                    escaped_val = html.escape(str(val)) if val is not None else "N/A"
                    feature_rows.append(f"<tr{style}><td class='ln-key'>{escaped_col}</td><td class='ln-val'>{escaped_val}</td></tr>")

                features_table = ''.join(feature_rows)
                top_feats_str = ", ".join([f"{k}: {v:.2f}" for k, v in list(top_features.items())[:3]])
                escaped_explanation = html.escape(str(expl['explanation']))

                card = f"""<div class='ln-card'>
                    <div class='ln-title'>Row {row_idx} · Score: {outlier_score:.4f}</div>
                    <div class='ln-meta'>Top Features: {html.escape(top_feats_str)}</div>
                    <table class='ln-table'>{features_table}</table>
                    <div class='ln-exp'>{escaped_explanation}</div>
                </div>"""
                cards_html.append(card)

            grid_html = "<div class='ln-grid'>" + ''.join(cards_html) + "</div>"
            lines.append("🤖 LLM EXAMPLE EXPLANATIONS")
            lines.append(grid_html)

        if self.llm:
            try:
                summary_data = self.summarize()
                component_summary = self.llm.generate_component_summary(
                    component_name="Outlier Detection",
                    metrics={"outlier_ratio": summary_data["outlier_ratio"], "num_outliers": len(self.result["outliers"])},
                    findings=f"Found {len(self.result['outliers'])} outliers ({summary_data['outlier_ratio']:.1%} of data)"
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
        return (
            "Isolation Forest-based multivariate outlier detection with SHAP explanations. "
            "Handles missing values and filters high-cardinality columns."
        )