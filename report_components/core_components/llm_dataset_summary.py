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
            for comp_name, comp_data in component_summaries.items():
                condensed_summaries[comp_name] = {}
                for key, value in comp_data.items():
                    if key in ['llm_explanation', 'llm_explanations', 'example_violations',
                               'example_outliers', 'example_explanations']:
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
            "llm_summary_available": bool(self.result["llm_summary"])
        }

    def justify(self) -> str:
        return (
            "The LLM Dataset Summary component synthesizes results from all analysis "
            "components into a coherent narrative that highlights key findings, risks, "
            "and actionable recommendations. Using large language models enables natural "
            "language explanations that are accessible to both technical and non-technical "
            "stakeholders. This component bridges the gap between detailed statistical "
            "analysis and practical decision-making by providing prioritized recommendations "
            "and ML readiness assessments in human-readable format."
        )

    def get_full_summary(self) -> str:
        if self.result is None:
            raise RuntimeError("analyze() must be called before get_full_summary()")

        lines = [
            "",
            "=" * 80,
            "📊 LLM DATASET QUALITY ASSESSMENT",
            "=" * 80,
            self.result["llm_summary"],
            "=" * 80
        ]
        return "\n".join(lines)

    def print_summary(self):
        print(self.get_full_summary())