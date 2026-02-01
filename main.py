import datetime
import os
import webbrowser

from core.html_report_generator import HTMLReportGenerator
from core.report import Report
from preprocessing.dataset import Dataset
from report_components.base_component import AnalysisContext
from report_components.simple_components.missing_values import MissingValuesReport
from report_components.simple_components.dataset_overview import DatasetOverviewComponent
from report_components.simple_components.exact_duplicates import ExactDuplicateDetectionComponent
from report_components.core_components.outlier_detection import OutlierDetectionComponent
from report_components.core_components.categoircal_outlier_detection import CategoricalOutlierDetectionComponent
from report_components.core_components.distribution_modelling import DistributionModelingComponent
from report_components.core_components.label_noise_detection import LabelNoiseDetectionComponent
from report_components.core_components.composite_quality_score import CompositeQualityScoreComponent
from report_components.core_components.relational_consistency import RelationalConsistencyComponent
from report_components.core_components.near_duplicate_detection import NearDuplicateDetectionComponent
from report_components.core_components.llm_dataset_summary import LLMDatasetSummaryComponent

def main():
    dataset = Dataset.from_parquet("titanic.parquet")
    #dataset = Dataset.from_csv("sample_dataset.csv")
    context = AnalysisContext(dataset)

    report = Report()
    report.add_component(DatasetOverviewComponent(context))
    report.add_component(MissingValuesReport(context))
    report.add_component(ExactDuplicateDetectionComponent(context))
    report.add_component(NearDuplicateDetectionComponent(context))
    report.add_component(OutlierDetectionComponent(context))
    report.add_component(CategoricalOutlierDetectionComponent(context))
    report.add_component(LabelNoiseDetectionComponent(context, 'Survived'))
    report.add_component(RelationalConsistencyComponent(context))
    report.add_component(DistributionModelingComponent(context))
    report.add_component(CompositeQualityScoreComponent(context))
    report.add_component(LLMDatasetSummaryComponent(context))

    report.run()

    generator = HTMLReportGenerator("AEDA Data Quality Report")
    report_path = generator.generate(report.components, "data_quality_report.html")

    webbrowser.open('file://' + os.path.realpath(report_path))

    print(f'\n✅ Report generated: {report_path} - {datetime.datetime.now().strftime("H%M")}')

if __name__ == "__main__":
    main()