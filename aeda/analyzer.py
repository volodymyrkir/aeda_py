import os
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any, Callable

import pandas as pd

from aeda.config import AnalysisConfig
from preprocessing.dataset import Dataset
from utils.consts import POLARS_SIZE_THRESHOLD_MB
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
        config: Optional[AnalysisConfig] = None,
        verbose: bool = True,
        log_callback: Optional[Callable[[str], None]] = None
    ):
        self.config = config or AnalysisConfig()
        self._verbose = verbose
        self._log_callback = log_callback
        self._dataset = self._load_dataset(data)
        self._context = AnalysisContext(self._dataset)
        self._report = Report()
        self._results: Dict[str, Any] = {}
        self._report_path: Optional[str] = None
        self._component_count = 0
        self._total_components = 0

    def _log(self, message: str):
        if self._verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted = f"[{timestamp}] {message}"
            if self._log_callback:
                self._log_callback(formatted)
            else:
                print(formatted)

    def _load_dataset(self, data: Union[str, Path, pd.DataFrame]) -> Dataset:
        if isinstance(data, pd.DataFrame):
            self._log("Loading DataFrame directly with pandas engine")
            return Dataset(dataframe=data, engine="pandas")

        path = str(data)
        engine = self.config.engine

        if engine == "auto":
            engine = None
            self._log("Engine set to auto - will use recommended based on file size")
        else:
            self._log(f"Engine specified: {engine}")

        dataset = Dataset(path, engine=engine)
        info = dataset.get_info()

        self._log(f"Dataset: {path}")
        self._log(f"File size: {info['file_size_mb']:.2f} MB | Threshold: {POLARS_SIZE_THRESHOLD_MB} MB")
        self._log(f"Recommended engine: {info['recommended_engine'].upper()}")
        self._log(f"Using engine: {info['engine'].upper()}")

        return dataset

    def analyze(self) -> "AEDAAnalyzer":
        enabled = self.config.get_enabled_components()
        use_llm = self.config.use_llm
        self._total_components = len(enabled)
        self._component_count = 0

        self._log(f"Starting analysis with {self._total_components} components...")

        if "dataset_overview" in enabled:
            component = DatasetOverviewComponent(self._context)
            self._run_component(component, "Dataset Overview")

        if "missing_values" in enabled:
            component = MissingValuesReport(self._context)
            self._run_component(component, "Missing Values")

        if "exact_duplicates" in enabled:
            component = ExactDuplicateDetectionComponent(self._context)
            self._run_component(component, "Exact Duplicates")

        if "near_duplicates" in enabled:
            component = NearDuplicateDetectionComponent(
                self._context, use_llm_explanations=use_llm
            )
            self._run_component(component, "Near Duplicates")

        if "outlier_detection" in enabled:
            component = OutlierDetectionComponent(
                self._context, use_llm_explanations=use_llm
            )
            self._run_component(component, "Outlier Detection")

        if "categorical_outliers" in enabled:
            component = CategoricalOutlierDetectionComponent(
                self._context, use_llm_explanations=use_llm
            )
            self._run_component(component, "Categorical Outliers")

        if "label_noise" in enabled and self.config.target_column:
            component = LabelNoiseDetectionComponent(
                self._context,
                self.config.target_column,
                use_llm_explanations=use_llm
            )
            self._run_component(component, "Label Noise Detection")

        if "relational_consistency" in enabled:
            component = RelationalConsistencyComponent(
                self._context, use_llm_explanations=use_llm
            )
            self._run_component(component, "Relational Consistency")

        if "distribution_modeling" in enabled:
            component = DistributionModelingComponent(
                self._context, use_llm_explanations=use_llm
            )
            self._run_component(component, "Distribution Modeling")

        if "composite_quality" in enabled:
            component = CompositeQualityScoreComponent(self._context)
            self._run_component(component, "Composite Quality Score")

        if "dataset_summary" in enabled:
            component = LLMDatasetSummaryComponent(self._context)
            self._run_component(component, "Dataset Summary")

        self._log("Analysis complete!")
        return self

    def _run_component(self, component, name: str) -> None:
        self._component_count += 1
        self._log(f"[{self._component_count}/{self._total_components}] Running: {name}")
        component.analyze()
        self._report.add_component(component)
        class_name = component.__class__.__name__
        try:
            self._results[class_name] = component.summarize()
            self._context.component_results[class_name] = self._results[class_name]
        except Exception:
            pass
        self._log(f"[{self._component_count}/{self._total_components}] ✓ Completed: {name}")

    def generate_report(self, output_path: Optional[str] = None) -> str:
        self._log("Generating HTML report...")
        path = output_path or self.config.output_path
        generator = HTMLReportGenerator(self.config.report_title)
        self._report_path = generator.generate(self._report.components, path)
        full_path = os.path.abspath(self._report_path)
        self._log(f"✓ Report saved: {full_path}")
        return full_path

    def open_report(self) -> None:
        if self._report_path:
            self._log("Opening report in browser...")
            webbrowser.open("file://" + os.path.abspath(self._report_path))

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self, component_name: str) -> Optional[Dict[str, Any]]:
        return self._results.get(component_name)

    def get_engine_info(self) -> Dict[str, Any]:
        info = self._dataset.get_info()
        return {
            "engine": info["engine"],
            "recommended_engine": info["recommended_engine"],
            "file_size_mb": info["file_size_mb"],
            "threshold_mb": POLARS_SIZE_THRESHOLD_MB
        }

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataset.df

    @property
    def components(self):
        return self._report.components
