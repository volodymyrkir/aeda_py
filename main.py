from preprocessing.dataset import Dataset
from report_components.base_component import AnalysisContext
from core.report import Report
from report_components.simple_components.missing_values import MissingValuesReport
from report_components.simple_components.dataset_overview import DatasetOverviewComponent
from report_components.simple_components.exact_duplicates import ExactDuplicateDetectionComponent

def main():
    dataset = Dataset.from_parquet("titanic.parquet")
    # dataset = Dataset.from_csv("sample_dataset.csv")
    context = AnalysisContext(dataset)

    report = Report()
    report.add_component(MissingValuesReport(context))
    report.add_component(DatasetOverviewComponent(context))
    report.add_component(ExactDuplicateDetectionComponent(context))

    report.run()

    for component in report.components:
        print("=" * 80)
        print(component.__class__.__name__)
        print("- Justification:")
        print(component.justify())
        print("- Summary:")
        for k, v in component.summarize().items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
