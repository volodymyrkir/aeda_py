from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from report_components.base_component import ReportComponent, AnalysisContext
from utils.consts import NUM_EXAMPLES_LLM

class CategoricalOutlierDetectionComponent(ReportComponent):
    def __init__(
        self,
        context: AnalysisContext,
        min_frequency_threshold: float = 0.01,
        adaptive_threshold: bool = True,
        skip_high_cardinality: bool = True,
        cardinality_threshold: float = 0.8,
        max_explanations_per_col: int = 5,
        use_chi_square: bool = False,
        min_null_ratio_to_warn: float = 0.5,
        use_llm_explanations: bool = True
    ):
        super().__init__(context, use_llm_explanations)
        self.min_frequency_threshold = min_frequency_threshold
        self.adaptive_threshold = adaptive_threshold
        self.skip_high_cardinality = skip_high_cardinality
        self.cardinality_threshold = cardinality_threshold
        self.max_explanations_per_col = max_explanations_per_col
        self.use_chi_square = use_chi_square
        self.min_null_ratio_to_warn = min_null_ratio_to_warn
        self.llm_explanations = []

    def analyze(self):
        df = self.context.dataset.df
        if df is None or df.empty:
            raise ValueError("Dataset is empty or not provided")

        cat_cols = self.context.shared_artifacts.get(
            "categorical_columns",
            list(df.select_dtypes(include=['object', 'category']).columns)
        )
        if not cat_cols:
            return

        n_rows = len(df)
        outlier_masks = {}
        explanations = []
        skipped_cols = []
        high_null_cols = []
        llm_count = 0

        for col in cat_cols:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            null_ratio = 1 - (len(non_null) / n_rows)
            if null_ratio > self.min_null_ratio_to_warn:
                high_null_cols.append({"column": col, "null_ratio": round(null_ratio, 4)})

            freq = non_null.value_counts(normalize=True)
            unique_ratio = len(freq) / len(non_null)

            if self.skip_high_cardinality and unique_ratio > self.cardinality_threshold:
                skipped_cols.append({"column": col, "unique_ratio": round(unique_ratio, 4)})
                continue

            threshold = self.min_frequency_threshold
            if self.adaptive_threshold:
                adaptive_min = max(0.001, 5 / len(non_null))
                threshold = min(threshold, adaptive_min)

            rare_mask = df[col].isin(freq[freq < threshold].index)
            outlier_masks[col] = rare_mask

            if self.use_chi_square and len(cat_cols) > 1:
                for other_col in cat_cols:
                    if other_col != col:
                        contingency = pd.crosstab(df[col], df[other_col])
                        if not contingency.empty:
                            chi2, p, _, _ = chi2_contingency(contingency)
                            if p < 0.05:
                                explanations.append({
                                    "columns": [col, other_col],
                                    "chi2_p_value": round(p, 5),
                                    "note": "Unexpected distribution interaction"
                                })

            rare_df = pd.DataFrame({
                "index": non_null[non_null.isin(freq[freq < threshold].index)].index,
                "value": non_null[non_null.isin(freq[freq < threshold].index)],
                "frequency": non_null[non_null.isin(freq[freq < threshold].index)].map(freq)
            }).sort_values("frequency").head(self.max_explanations_per_col)

            for _, row in rare_df.iterrows():
                narrative = self._generate_narrative(row["frequency"], col, row["value"])
                explanation_entry = {
                    "row_index": int(row["index"]),
                    "column": col,
                    "value": row["value"],
                    "frequency": row["frequency"],
                    "narrative": narrative
                }

                if llm_count < NUM_EXAMPLES_LLM and self.llm:
                    try:
                        row_data = df.iloc[int(row["index"])].to_dict()
                        llm_explanation = self.llm.explain_outlier(
                            row_data=row_data,
                            outlier_score=1 - row["frequency"],
                            contributing_features={col: 1 - row["frequency"]},
                            dataset_context=f"Rare value '{row['value']}' in '{col}'"
                        )
                        explanation_entry["llm_explanation"] = llm_explanation
                        self.llm_explanations.append({
                            "column": col,
                            "value": row["value"],
                            "explanation": llm_explanation
                        })
                        llm_count += 1
                    except Exception:
                        pass

                explanations.append(explanation_entry)

        overall_outlier_ratio = np.mean([mask.mean() for mask in outlier_masks.values()]) if outlier_masks else 0
        self.result = {
            "summary": {
                "outlier_ratio": round(overall_outlier_ratio, 5),
                "skipped_columns": skipped_cols,
                "high_null_columns": high_null_cols
            },
            "outliers": explanations
        }
        self.context.shared_artifacts["categorical_outlier_masks"] = outlier_masks

    def _generate_narrative(self, freq: float, col: str, value: Any) -> str:
        return f"Value '{value}' in '{col}' appears with frequency {freq:.5f}"

    def summarize(self) -> dict:
        if self.result is None:
            return {"summary": {"outlier_ratio": 0}, "outliers": []}

        example_outliers = []
        for outlier in self.result["outliers"][:5]:
            clean_outlier = {k: v for k, v in outlier.items() if k != 'llm_explanation'}
            example_outliers.append(clean_outlier)

        summary = {
            "outlier_ratio": self.result["summary"]["outlier_ratio"],
            "skipped_columns": self.result["summary"]["skipped_columns"],
            "example_outliers": example_outliers
        }

        return summary

    def get_full_summary(self) -> str:
        if self.result is None:
            return "No analysis performed."

        lines = []

        if self.llm_explanations:
            lines.append(f"\n{'='*80}")
            lines.append("🤖 LLM EXAMPLE EXPLANATIONS")
            lines.append(f"{'='*80}")
            for i, expl in enumerate(self.llm_explanations, 1):
                lines.append(f"\n{i}. Column: {expl['column']}, Value: {expl['value']}")
                lines.append(f"   {expl['explanation']}")
            lines.append("")

        if self.llm:
            try:
                summary_data = self.summarize()
                component_summary = self.llm.generate_component_summary(
                    component_name="Categorical Outlier Detection",
                    metrics={"outlier_ratio": summary_data["outlier_ratio"], "num_outliers": len(self.result["outliers"])},
                    findings=f"Found {len(self.result['outliers'])} rare categorical values ({summary_data['outlier_ratio']:.1%} of data)"
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
            "Frequency-based categorical outlier detection with adaptive thresholds. "
            "Filters high-cardinality columns and flags rare values."
        )
