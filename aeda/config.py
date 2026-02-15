from dataclasses import dataclass, field
from typing import Optional, List, Type


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
    exclude_components: List[Type] = field(default_factory=list)

    def get_enabled_components(self) -> List[str]:
        excluded_names = {cls.__name__ for cls in self.exclude_components}

        component_map = {
            "dataset_overview": ("DatasetOverviewComponent", self.dataset_overview),
            "missing_values": ("MissingValuesReport", self.missing_values),
            "exact_duplicates": ("ExactDuplicateDetectionComponent", self.exact_duplicates),
            "near_duplicates": ("NearDuplicateDetectionComponent", self.near_duplicates),
            "outlier_detection": ("OutlierDetectionComponent", self.outlier_detection),
            "categorical_outliers": ("CategoricalOutlierDetectionComponent", self.categorical_outliers),
            "label_noise": ("LabelNoiseDetectionComponent", self.label_noise and bool(self.target_column)),
            "relational_consistency": ("RelationalConsistencyComponent", self.relational_consistency),
            "distribution_modeling": ("DistributionModelingComponent", self.distribution_modeling),
            "composite_quality": ("CompositeQualityScoreComponent", self.composite_quality),
            "dataset_summary": ("LLMDatasetSummaryComponent", self.dataset_summary),
        }

        components = []
        for key, (class_name, enabled) in component_map.items():
            if enabled and class_name not in excluded_names:
                components.append(key)

        return components
