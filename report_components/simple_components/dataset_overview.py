from typing import Dict, Any, Hashable

import pandas as pd
import numpy as np

from report_components.base_component import ReportComponent, AnalysisContext
from utils.consts import IDENTIFIER_UNIQUENESS_THRESHOLD, IDENTIFIER_MIN_UNIQUE, LOW_CARDINALITY_THRESHOLD


class DatasetOverviewComponent(ReportComponent):
    def __init__(self, context: AnalysisContext, use_llm_explanations: bool = True):
        super().__init__(context, use_llm_explanations)

    def analyze(self):
        df = self.context.dataset.df

        if df is None or df.empty:
            raise ValueError("Dataset is empty or not provided")

        overview = {
            "shape": self._analyze_shape(df),
            "dtypes": self._analyze_dtypes(df),
            "memory": self._analyze_memory(df),
            "cardinality": self._analyze_cardinality(df),
            "basic_statistics": self._analyze_statistics(df),
        }

        self.result = overview

        self.context.shared_artifacts["numeric_columns"] = list(
            df.select_dtypes(include=[np.number]).columns
        )
        self.context.shared_artifacts["categorical_columns"] = list(
            df.select_dtypes(exclude=[np.number]).columns
        )

    @staticmethod
    def _analyze_shape(df: pd.DataFrame) -> Dict[str, int]:
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        }

    def _analyze_dtypes(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "per_column": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "distribution": df.dtypes.astype(str).value_counts().to_dict()
        }

    @staticmethod
    def _analyze_memory(df: pd.DataFrame) -> Dict[str, Any]:
        mem = df.memory_usage(deep=True)
        return {
            "total_mb": round(mem.sum() / 1024**2, 3),
            "per_column_mb": {
                col: round(mem[col] / 1024**2, 4) for col in df.columns
            }
        }

    @staticmethod
    def _analyze_cardinality(df: pd.DataFrame) -> Dict[str, Any]:
        cardinality = {}

        for col in df.columns:
            unique = df[col].nunique(dropna=True)
            ratio = unique / len(df)

            cardinality[col] = {
                "unique_values": int(unique),
                "uniqueness_ratio": round(ratio, 5),
                "potential_identifier": bool(ratio > IDENTIFIER_UNIQUENESS_THRESHOLD and unique > IDENTIFIER_MIN_UNIQUE),
                "low_cardinality": bool(unique < LOW_CARDINALITY_THRESHOLD)
            }

        return cardinality

    @staticmethod
    def _analyze_statistics(df: pd.DataFrame) -> dict[Hashable, dict[Any, float]] | dict[Any, Any]:
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return {}

        desc = numeric_df.describe(
            percentiles=[0.01, 0.05, 0.95, 0.99]
        ).to_dict()

        return {
            col: {k: float(v) for k, v in stats.items()}
            for col, stats in desc.items()
        }

    def summarize(self) -> dict:
        if self.result is None:
            raise RuntimeError("analyze() must be called before summarize()")

        return {
            "dataset_shape": self.result["shape"],
            "type_distribution": self.result["dtypes"]["distribution"],
            "memory_mb": self.result["memory"]["total_mb"],
            "potential_identifiers": [
                col for col, info in self.result["cardinality"].items()
                if info["potential_identifier"]
            ]
        }

    def justify(self) -> str:
        return (
            "The Dataset Overview component establishes a structural and statistical "
            "baseline for the entire analysis pipeline. Understanding data types, "
            "cardinality, memory footprint, and numerical distributions is a prerequisite "
            "for any downstream data quality assessment or machine learning task. "
            "This component provides context-aware signals (e.g., potential identifiers, "
            "low-cardinality features) that directly influence model selection, "
            "feature engineering, leakage detection, and anomaly interpretation."
        )

    def get_full_summary(self) -> str:
        if self.result is None:
            return ""

        lines = []
        if self.llm:
            try:
                summary_data = self.summarize()
                identifiers = summary_data.get("potential_identifiers", [])
                type_dist = summary_data.get("type_distribution", {})

                findings_parts = [
                    f"Dataset has {summary_data['dataset_shape']['rows']} rows and {summary_data['dataset_shape']['columns']} columns"
                ]

                if type_dist:
                    type_info = ", ".join([f"{count} {dtype}" for dtype, count in type_dist.items()])
                    findings_parts.append(f"Column types: {type_info}")

                if identifiers:
                    findings_parts.append(f"Potential identifier columns detected: {', '.join(identifiers)}")
                else:
                    findings_parts.append("No high-cardinality identifier columns detected")

                findings = ". ".join(findings_parts)

                component_summary = self.llm.generate_component_summary(
                    component_name="Dataset Overview",
                    metrics={"rows": summary_data['dataset_shape']['rows'], "columns": summary_data['dataset_shape']['columns']},
                    findings=findings
                )
                lines.append(f"\n{'='*80}")
                lines.append("📋 COMPONENT SUMMARY")
                lines.append(f"{'='*80}")
                lines.append(component_summary)
                lines.append(f"{'='*80}\n")
            except Exception:
                pass

        return "\n".join(lines)
