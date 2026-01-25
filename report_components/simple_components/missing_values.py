from report_components.base_component import ReportComponent, AnalysisContext


class MissingValuesReport(ReportComponent):
    def __init__(self, context: AnalysisContext, use_llm_explanations: bool = True):
        super().__init__(context, use_llm_explanations)

    def analyze(self):
        df = self.context.dataset.df

        missing_counts = df.isna().sum()
        missing_ratio = missing_counts / len(df)

        self.result = {
            "missing_counts": missing_counts.to_dict(),
            "missing_ratio": missing_ratio.to_dict(),
            "columns_with_missing": [
                col for col, ratio in missing_ratio.items() if ratio > 0
            ]
        }

    def summarize(self) -> dict:
        return {
            "num_columns_with_missing": len(self.result["columns_with_missing"]),
            "worst_columns": sorted(
                self.result["missing_ratio"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }

    def justify(self) -> str:
        return (
            "Missing values analysis identifies incomplete features that may "
            "introduce bias, reduce statistical power, or invalidate assumptions "
            "of machine learning models. Patterned or high missingness often "
            "requires explicit handling before model training."
        )

    def get_full_summary(self) -> str:
        if self.result is None:
            return ""

        lines = []
        if self.llm:
            try:
                summary_data = self.summarize()
                worst_cols = summary_data["worst_columns"][:3]
                findings = f"{summary_data['num_columns_with_missing']} columns with missing values"
                if worst_cols:
                    findings += f". Worst: {worst_cols[0][0]} ({worst_cols[0][1]:.1%} missing)"

                component_summary = self.llm.generate_component_summary(
                    component_name="Missing Values Analysis",
                    metrics={"num_columns_with_missing": summary_data["num_columns_with_missing"]},
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

