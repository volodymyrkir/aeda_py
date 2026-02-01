from typing import Dict, Any, Optional, List
import json

from report_components.base_component import ReportComponent, AnalysisContext


class LLMDatasetSummaryComponent(ReportComponent):
    def __init__(
        self,
        context: AnalysisContext,
        include_raw_summaries: bool = True,
        custom_context: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        use_llm_explanations: bool = True
    ):
        super().__init__(context, use_llm_explanations)
        self.include_raw_summaries = include_raw_summaries
        self.custom_context = custom_context
        self.focus_areas = focus_areas or []

    def _get_dataset_info(self) -> Dict[str, Any]:
        df = self.context.dataset.df

        info = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "numeric_columns": list(df.select_dtypes(include=['number']).columns),
            "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
        }

        return info

    def _collect_component_summaries(self) -> Dict[str, Dict[str, Any]]:
        return self.context.component_results.copy()

    def _generate_fallback_summary(
        self,
        dataset_info: Dict[str, Any],
        component_summaries: Dict[str, Dict[str, Any]]
    ) -> str:
        lines = [
            "# Dataset Quality Summary",
            "",
            "## Dataset Overview",
            f"- **Rows**: {dataset_info['num_rows']:,}",
            f"- **Columns**: {dataset_info['num_columns']}",
            "",
            "## Analysis Results",
            ""
        ]

        for component_name, summary in component_summaries.items():
            lines.append(f"### {component_name}")
            for key, value in summary.items():
                if isinstance(value, dict):
                    lines.append(f"- **{key}**: {json.dumps(value, default=str)}")
                elif isinstance(value, list):
                    lines.append(f"- **{key}**: {len(value)} items")
                else:
                    lines.append(f"- **{key}**: {value}")
            lines.append("")

        lines.extend([
            "## Recommendations",
            "",
            "*LLM service not available for detailed recommendations.*",
            "*Please review individual component results above.*",
            "",
            "---",
            "*Note: Enable LLM service for comprehensive AI-powered analysis.*"
        ])

        return "\n".join(lines)

    def _generate_llm_summary(
        self,
        dataset_info: Dict[str, Any],
        component_summaries: Dict[str, Dict[str, Any]]
    ) -> str:
        if not self.llm or not self.llm.is_available:
            return self._generate_fallback_summary(dataset_info, component_summaries)

        try:
            condensed_summaries = {}
            skip_keys = {'llm_explanation', 'llm_explanations', 'example_violations',
                         'example_outliers', 'example_explanations', 'memory_mb',
                         'potential_identifiers', 'worst_columns'}

            for comp_name, comp_data in component_summaries.items():
                condensed_summaries[comp_name] = {}
                for key, value in comp_data.items():
                    if key in skip_keys:
                        continue
                    if isinstance(value, str) and len(value) > 100:
                        continue
                    if isinstance(value, (list, dict)):
                        continue
                    condensed_summaries[comp_name][key] = value

            summary = self.llm.generate_dataset_summary(
                component_results=condensed_summaries,
                dataset_info={"num_rows": dataset_info["num_rows"], "num_columns": dataset_info["num_columns"]}
            )

            return summary

        except Exception as e:
            fallback = self._generate_fallback_summary(dataset_info, component_summaries)
            return f"{fallback}\n\n*LLM generation failed: {str(e)}*"

    def analyze(self):
        dataset_info = self._get_dataset_info()
        component_summaries = self._collect_component_summaries()

        llm_summary = self._generate_llm_summary(dataset_info, component_summaries)

        total_issues = self._count_total_issues(component_summaries)
        risk_level = self._assess_overall_risk(component_summaries)

        self.result = {
            "summary": {
                "dataset_info": dataset_info,
                "total_components_analyzed": len(component_summaries),
                "total_issues_detected": total_issues,
                "overall_risk_level": risk_level
            },
            "llm_summary": llm_summary,
            "component_summaries": component_summaries if self.include_raw_summaries else {}
        }

    def _count_total_issues(self, summaries: Dict[str, Dict[str, Any]]) -> int:
        total = 0

        issue_keys = [
            'outlier_ratio', 'near_duplicate_ratio', 'duplicate_ratio',
            'missing_ratio', 'total_violations', 'noisy_label_ratio',
            'affected_rows', 'violation_count'
        ]

        for summary in summaries.values():
            for key in issue_keys:
                if key in summary:
                    value = summary[key]
                    if isinstance(value, (int, float)) and value > 0:
                        if 'ratio' in key:
                            total += 1
                        else:
                            total += int(value)

        return total

    def _assess_overall_risk(self, summaries: Dict[str, Dict[str, Any]]) -> str:
        risk_scores = []

        for component_name, summary in summaries.items():
            if 'risk_level' in summary:
                risk_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
                risk_scores.append(risk_map.get(summary['risk_level'], 0))

            for key, value in summary.items():
                if 'ratio' in key and isinstance(value, (int, float)):
                    if value > 0.2:
                        risk_scores.append(3)
                    elif value > 0.1:
                        risk_scores.append(2)
                    elif value > 0.05:
                        risk_scores.append(1)

        if not risk_scores:
            return "unknown"

        max_risk = max(risk_scores)
        avg_risk = sum(risk_scores) / len(risk_scores)

        final_risk = (max_risk + avg_risk) / 2

        if final_risk >= 3.5:
            return "critical"
        elif final_risk >= 2.5:
            return "high"
        elif final_risk >= 1.5:
            return "medium"
        elif final_risk >= 0.5:
            return "low"
        else:
            return "none"

    def summarize(self) -> dict:
        if self.result is None:
            raise RuntimeError("analyze() must be called before summarize()")

        return {
            "total_components_analyzed": self.result["summary"]["total_components_analyzed"],
            "total_issues_detected": self.result["summary"]["total_issues_detected"],
            "overall_risk_level": self.result["summary"]["overall_risk_level"],
            "llm_summary_available": bool(self.result.get("llm_analysis"))
        }

    def justify(self) -> str:
        return (
            "The Dataset Summary component synthesizes results from all analysis "
            "components into a coherent overview that highlights key findings, risks, "
            "and actionable recommendations. It combines factual metrics from each "
            "component with AI-powered insights for both technical and non-technical "
            "stakeholders."
        )

    def _format_component_metrics(self) -> str:
        lines = []
        component_summaries = self.result.get("component_summaries", {})

        metric_mappings = {
            "MissingValuesReport": [("num_columns_with_missing", "Columns with missing values")],
            "DatasetOverviewComponent": [("dataset_shape", "Dataset shape")],
            "ExactDuplicateDetectionComponent": [("duplicate_ratio", "Duplicate ratio"), ("risk_level", "Risk level")],
            "OutlierDetectionComponent": [("outlier_ratio", "Outlier ratio")],
            "CategoricalOutlierDetectionComponent": [("outlier_ratio", "Categorical outlier ratio")],
            "DistributionModelingComponent": [("mean_reconstruction_error", "Mean reconstruction error"), ("high_error_ratio", "High error ratio")],
            "CompositeQualityScoreComponent": [("data_readiness_score", "Data readiness score"), ("overall_quality", "Overall quality")],
            "LabelNoiseDetectionComponent": [("noise_ratio", "Label noise ratio"), ("suspicious_sample_count", "Suspicious samples")],
            "RelationalConsistencyComponent": [("total_violations", "Total violations"), ("violation_ratio", "Violation ratio")],
            "NearDuplicateDetectionComponent": [("near_duplicate_ratio", "Near-duplicate ratio"), ("affected_rows", "Affected rows")]
        }

        for comp_name, metrics in metric_mappings.items():
            if comp_name in component_summaries:
                summary = component_summaries[comp_name]
                comp_lines = []
                for key, label in metrics:
                    if key in summary:
                        value = summary[key]
                        if isinstance(value, float):
                            if 'ratio' in key or 'score' in key:
                                comp_lines.append(f"  • {label}: {value:.1%}")
                            else:
                                comp_lines.append(f"  • {label}: {value:.4f}")
                        elif isinstance(value, dict):
                            comp_lines.append(f"  • {label}: {value}")
                        else:
                            comp_lines.append(f"  • {label}: {value}")

                if comp_lines:
                    display_name = comp_name.replace("Component", "").replace("Report", "")
                    lines.append(f"📌 {display_name}")
                    lines.extend(comp_lines)

        return "\n".join(lines)

    def _generate_pros_cons_prompt(self) -> str:
        component_summaries = self.result.get("component_summaries", {})
        dataset_info = self.result["summary"]["dataset_info"]

        metrics = []
        metrics.append(f"Dataset: {dataset_info['num_rows']} rows, {dataset_info['num_columns']} columns")

        if "ExactDuplicateDetectionComponent" in component_summaries:
            s = component_summaries["ExactDuplicateDetectionComponent"]
            metrics.append(f"Duplicates: {s.get('duplicate_ratio', 0):.1%}")

        if "MissingValuesReport" in component_summaries:
            s = component_summaries["MissingValuesReport"]
            metrics.append(f"Missing: {s.get('num_columns_with_missing', 0)} columns affected")

        if "OutlierDetectionComponent" in component_summaries:
            s = component_summaries["OutlierDetectionComponent"]
            metrics.append(f"Outliers: {s.get('outlier_ratio', 0):.1%}")

        if "LabelNoiseDetectionComponent" in component_summaries:
            s = component_summaries["LabelNoiseDetectionComponent"]
            metrics.append(f"Label noise: {s.get('noise_ratio', 0):.1%}")

        if "CompositeQualityScoreComponent" in component_summaries:
            s = component_summaries["CompositeQualityScoreComponent"]
            metrics.append(f"Readiness score: {s.get('data_readiness_score', 0):.1%}")

        return ". ".join(metrics)

    def _generate_strengths_weaknesses(self) -> tuple:
        """Generate strengths and weaknesses based on component results."""
        component_summaries = self.result.get("component_summaries", {})
        strengths = []
        weaknesses = []
        priority_action = ""

        # Check duplicates
        if "ExactDuplicateDetectionComponent" in component_summaries:
            s = component_summaries["ExactDuplicateDetectionComponent"]
            dup_ratio = s.get('duplicate_ratio', 0)
            if dup_ratio == 0:
                strengths.append("No duplicate records - data integrity maintained")
            elif dup_ratio > 0.05:
                weaknesses.append(f"High duplicate ratio ({dup_ratio:.1%}) - needs deduplication")

        # Check missing values
        if "MissingValuesReport" in component_summaries:
            s = component_summaries["MissingValuesReport"]
            missing_cols = s.get('num_columns_with_missing', 0)
            if missing_cols == 0:
                strengths.append("No missing values - complete dataset")
            elif missing_cols <= 2:
                strengths.append(f"Low missingness - only {missing_cols} columns affected")
            else:
                weaknesses.append(f"{missing_cols} columns have missing values - imputation needed")
                if not priority_action:
                    priority_action = "Handle missing values through imputation or removal"

        # Check outliers
        if "OutlierDetectionComponent" in component_summaries:
            s = component_summaries["OutlierDetectionComponent"]
            outlier_ratio = s.get('outlier_ratio', 0)
            if outlier_ratio < 0.05:
                strengths.append(f"Low outlier ratio ({outlier_ratio:.1%}) - clean data distribution")
            elif outlier_ratio > 0.2:
                weaknesses.append(f"High outlier ratio ({outlier_ratio:.1%}) - review flagged records")

        # Check label noise
        if "LabelNoiseDetectionComponent" in component_summaries:
            s = component_summaries["LabelNoiseDetectionComponent"]
            noise_ratio = s.get('noise_ratio', 0)
            if noise_ratio < 0.05:
                strengths.append(f"Low label noise ({noise_ratio:.1%}) - reliable labels")
            elif noise_ratio > 0.1:
                weaknesses.append(f"Significant label noise ({noise_ratio:.1%}) - verify suspicious labels")
                if not priority_action:
                    priority_action = "Review and correct potentially mislabeled records"

        # Check quality score
        if "CompositeQualityScoreComponent" in component_summaries:
            s = component_summaries["CompositeQualityScoreComponent"]
            score = s.get('data_readiness_score', 0)
            if score >= 0.8:
                strengths.append(f"High readiness score ({score:.0%}) - suitable for ML")
            elif score < 0.6:
                weaknesses.append(f"Low readiness score ({score:.0%}) - significant preprocessing needed")

        # Check near duplicates
        if "NearDuplicateDetectionComponent" in component_summaries:
            s = component_summaries["NearDuplicateDetectionComponent"]
            nd_ratio = s.get('near_duplicate_ratio', 0)
            if nd_ratio > 0.1:
                weaknesses.append(f"Near-duplicates detected ({nd_ratio:.1%}) - may affect model training")

        # Check relational consistency
        if "RelationalConsistencyComponent" in component_summaries:
            s = component_summaries["RelationalConsistencyComponent"]
            violations = s.get('total_violations', 0)
            if violations == 0:
                strengths.append("No consistency violations - logically coherent data")
            elif violations > 5:
                weaknesses.append(f"{violations} consistency violations found - data integrity issues")

        if not priority_action and weaknesses:
            priority_action = "Address the highest-impact issue first based on your ML task requirements"

        return strengths[:3], weaknesses[:3], priority_action

    def get_full_summary(self) -> str:
        if self.result is None:
            raise RuntimeError("analyze() must be called before get_full_summary()")

        lines = []

        lines.append("=" * 80)
        lines.append("📊 COMPONENT METRICS OVERVIEW")
        lines.append("=" * 80)
        lines.append(self._format_component_metrics())
        lines.append("")

        # Generate strengths and weaknesses programmatically
        strengths, weaknesses, priority_action = self._generate_strengths_weaknesses()

        lines.append("=" * 80)
        lines.append("🤖 AI QUALITY ASSESSMENT")
        lines.append("=" * 80)

        lines.append("✅ STRENGTHS")
        for s in strengths:
            lines.append(f"• {s}")

        lines.append("")
        lines.append("❌ ISSUES")
        for w in weaknesses:
            lines.append(f"• {w}")

        lines.append("")
        lines.append("🎯 PRIORITY ACTION")
        lines.append(f"• {priority_action}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def print_summary(self):
        print(self.get_full_summary())