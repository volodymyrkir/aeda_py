from report_components.base_component import ReportComponent

class MissingValuesReport(ReportComponent):
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
