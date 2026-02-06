from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AnalysisConfig:
    dataset_overview: bool = True
    missing_values: bool = True
    exact_duplicates: bool = True
    near_duplicates: bool = True
    outlier_detection: bool = True
    categorical_outliers: bool = True
    label_noise: bool = True
    relational_consistency: bool = True
    distribution_modeling: bool = True
    composite_quality: bool = True
    dataset_summary: bool = True
    target_column: Optional[str] = None
    use_llm: bool = True
    engine: str = "auto"
    output_path: str = "data_quality_report.html"
    report_title: str = "AEDA Data Quality Report"

    def get_enabled_components(self) -> List[str]:
        components = []
        if self.dataset_overview:
            components.append("dataset_overview")
        if self.missing_values:
            components.append("missing_values")
        if self.exact_duplicates:
            components.append("exact_duplicates")
        if self.near_duplicates:
            components.append("near_duplicates")
        if self.outlier_detection:
            components.append("outlier_detection")
        if self.categorical_outliers:
            components.append("categorical_outliers")
        if self.label_noise and self.target_column:
            components.append("label_noise")
        if self.relational_consistency:
            components.append("relational_consistency")
        if self.distribution_modeling:
            components.append("distribution_modeling")
        if self.composite_quality:
            components.append("composite_quality")
        if self.dataset_summary:
            components.append("dataset_summary")
        return components
