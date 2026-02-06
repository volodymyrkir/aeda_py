import os
import webbrowser
from pathlib import Path
from typing import Optional, Union, Dict, Any

import pandas as pd

from aeda.config import AnalysisConfig
from preprocessing.dataset import Dataset
from report_components.base_component import AnalysisContext
from core.report import Report
from core.html_report_generator import HTMLReportGenerator

from report_components.simple_components.dataset_overview import DatasetOverviewComponent
from report_components.simple_components.missing_values import MissingValuesReport
from report_components.simple_components.exact_duplicates import ExactDuplicateDetectionComponent
from report_components.core_components.near_duplicate_detection import NearDuplicateDetectionComponent
from report_components.core_components.outlier_detection import OutlierDetectionComponent
from report_components.core_components.categoircal_outlier_detection import CategoricalOutlierDetectionComponent
from report_components.core_components.label_noise_detection import LabelNoiseDetectionComponent
from report_components.core_components.relational_consistency import RelationalConsistencyComponent
from report_components.core_components.distribution_modelling import DistributionModelingComponent
from report_components.core_components.composite_quality_score import CompositeQualityScoreComponent
from report_components.core_components.llm_dataset_summary import LLMDatasetSummaryComponent


class AEDAAnalyzer:
    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        config: Optional[AnalysisConfig] = None
    ):
        self.config = config or AnalysisConfig()
        self._dataset = self._load_dataset(data)
        self._context = AnalysisContext(self._dataset)
        self._report = Report()
        self._results: Dict[str, Any] = {}
        self._report_path: Optional[str] = None

    def _load_dataset(self, data: Union[str, Path, pd.DataFrame]) -> Dataset:
        if isinstance(data, pd.DataFrame):
            return Dataset(dataframe=data, engine="pandas")

        path = str(data)
        engine = self.config.engine

        if engine == "auto":
            engine = None

        return Dataset(path, engine=engine)

    def analyze(self) -> "AEDAAnalyzer":
        enabled = self.config.get_enabled_components()
        use_llm = self.config.use_llm

        if "dataset_overview" in enabled:
            component = DatasetOverviewComponent(self._context)
            self._run_component(component)

        if "missing_values" in enabled:
            component = MissingValuesReport(self._context)
            self._run_component(component)

        if "exact_duplicates" in enabled:
            component = ExactDuplicateDetectionComponent(self._context)
            self._run_component(component)

        if "near_duplicates" in enabled:
            component = NearDuplicateDetectionComponent(
                self._context, use_llm_explanations=use_llm
            )
            self._run_component(component)

        if "outlier_detection" in enabled:
            component = OutlierDetectionComponent(
                self._context, use_llm_explanations=use_llm
            )
            self._run_component(component)

        if "categorical_outliers" in enabled:
            component = CategoricalOutlierDetectionComponent(
                self._context, use_llm_explanations=use_llm
            )
            self._run_component(component)

        if "label_noise" in enabled and self.config.target_column:
            component = LabelNoiseDetectionComponent(
                self._context,
                self.config.target_column,
                use_llm_explanations=use_llm
            )
            self._run_component(component)

        if "relational_consistency" in enabled:
            component = RelationalConsistencyComponent(
                self._context, use_llm_explanations=use_llm
            )
            self._run_component(component)

        if "distribution_modeling" in enabled:
            component = DistributionModelingComponent(
                self._context, use_llm_explanations=use_llm
            )
            self._run_component(component)

        if "composite_quality" in enabled:
            component = CompositeQualityScoreComponent(self._context)
            self._run_component(component)

        if "dataset_summary" in enabled:
            component = LLMDatasetSummaryComponent(self._context)
            self._run_component(component)

        return self

    def _run_component(self, component) -> None:
        component.analyze()
        self._report.add_component(component)
        name = component.__class__.__name__
        try:
            self._results[name] = component.summarize()
            self._context.component_results[name] = self._results[name]
        except Exception:
            pass

    def generate_report(self, output_path: Optional[str] = None) -> str:
        path = output_path or self.config.output_path
        generator = HTMLReportGenerator(self.config.report_title)
        self._report_path = generator.generate(self._report.components, path)
        return os.path.abspath(self._report_path)

    def open_report(self) -> None:
        if self._report_path:
            webbrowser.open("file://" + os.path.abspath(self._report_path))

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self, component_name: str) -> Optional[Dict[str, Any]]:
        return self._results.get(component_name)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataset.df

    @property
    def components(self):
        return self._report.components
