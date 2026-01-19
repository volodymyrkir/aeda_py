import hashlib
from typing import Dict, Any, List

import pandas as pd

from report_components.base_component import ReportComponent

LOW_DUPLICATES_RATIO = 0.001
MEDIUM_DUPLICATES_RATIO = 0.01

class ExactDuplicateDetectionComponent(ReportComponent):
    """
    Detects and analyzes exact duplicate rows and evaluates their impact.
    """

    def analyze(self):
        df = self.context.dataset.df

        if df is None or df.empty:
            raise ValueError("Dataset is empty or not provided")

        duplicate_mask = df.duplicated(keep=False)
        duplicate_df = df[duplicate_mask]

        self.result = {
            "summary": self._compute_summary(df, duplicate_mask),
            "groups": self._group_duplicates(duplicate_df),
            "impact": self._assess_impact(df, duplicate_df)
        }

        self.context.shared_artifacts["has_exact_duplicates"] = bool(
            self.result["summary"]["duplicate_rows"] > 0
        )


    def _compute_summary(
        self, df: pd.DataFrame, duplicate_mask: pd.Series
    ) -> Dict[str, Any]:
        return {
            "total_rows": int(len(df)),
            "duplicate_rows": int(duplicate_mask.sum()),
            "duplicate_ratio": round(float(duplicate_mask.mean()), 5),
            "unique_rows": int(len(df) - duplicate_mask.sum())
        }

    def _group_duplicates(self, duplicate_df: pd.DataFrame) -> List[Dict[str, Any]]:
        if duplicate_df.empty:
            return []

        grouped = duplicate_df.groupby(
            list(duplicate_df.columns),
            dropna=False
        )

        groups = []
        for values, group in grouped:
            groups.append({
                "row_hash": self._hash_row(values),
                "count": int(len(group)),
                "example": group.iloc[0].to_dict()
            })

        return sorted(groups, key=lambda x: x["count"], reverse=True)

    def _assess_impact(
        self, full_df: pd.DataFrame, duplicate_df: pd.DataFrame
    ) -> Dict[str, Any]:
        if duplicate_df.empty:
            return {"risk_level": "none"}

        ratio = len(duplicate_df) / len(full_df)

        return {
            "duplication_pressure": round(float(ratio), 5),
            "risk_level": self._risk_level(ratio),
            "ml_implications": [
                "Inflated confidence in frequent patterns",
                "Train/test leakage if duplicates cross splits",
                "Bias in frequency-based or probabilistic models"
            ]
        }


    def _hash_row(self, row_values) -> str:
        joined = "|".join(map(str, row_values))
        return hashlib.md5(joined.encode("utf-8")).hexdigest()

    def _risk_level(self, ratio: float) -> str:
        if ratio < LOW_DUPLICATES_RATIO:
            return "low"
        if ratio < MEDIUM_DUPLICATES_RATIO:
            return "medium"
        return "high"

    def summarize(self) -> dict:
        if self.result is None:
            raise RuntimeError("analyze() must be called before summarize()")

        return {
            "duplicate_ratio": self.result["summary"]["duplicate_ratio"],
            "duplicate_groups": len(self.result["groups"]),
            "risk_level": self.result["impact"].get("risk_level", "none")
        }

    def justify(self) -> str:
        return (
            "Exact duplicate detection is critical for ensuring the statistical validity "
            "of downstream machine learning experiments. Duplicate observations artificially "
            "inflate empirical support for certain patterns, bias loss minimization, and can "
            "lead to data leakage across training and evaluation splits. This component not "
            "only identifies duplicates but quantifies their systemic impact on model behavior, "
            "making it an essential part of a learning-based data quality assessment pipeline."
        )
