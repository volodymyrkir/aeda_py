"""
Mock main file for quick HTML report testing.
Uses mocked component outputs instead of running actual analysis.
"""
import os
import webbrowser

from core.html_report_generator import HTMLReportGenerator
from report_components.base_component import ReportComponent, AnalysisContext


class MockComponent(ReportComponent):
    """Base mock component that doesn't require a real context."""

    def __init__(self, name: str, justification: str, summary: dict, full_summary: str = ""):
        self._name = name
        self._justification = justification
        self._summary = summary
        self._full_summary = full_summary
        self.result = summary

    def analyze(self):
        pass

    def summarize(self) -> dict:
        return self._summary

    def justify(self) -> str:
        return self._justification

    def get_full_summary(self) -> str:
        return self._full_summary


class MockMissingValuesReport(MockComponent):
    def __init__(self):
        super().__init__(
            name="MissingValuesReport",
            justification="Missing values analysis identifies incomplete features that may introduce bias, reduce statistical power, or invalidate assumptions of machine learning models.",
            summary={
                "num_columns_with_missing": 3,
                "worst_columns": [
                    ("Cabin", 0.771),
                    ("Age", 0.199),
                    ("Embarked", 0.002)
                ]
            },
            full_summary="""
================================================================================
🤖 LLM EXAMPLE EXPLANATIONS
================================================================================

1. Column: Cabin (77.1% missing)
   High missingness suggests inconsistent recording, likely only available for certain passenger classes.

2. Column: Age (19.9% missing)
   Moderate missingness could benefit from model-based imputation using Pclass and Fare.

================================================================================
📋 COMPONENT SUMMARY
================================================================================
Three columns have missing values. Cabin has critical missingness (77%) - consider dropping or creating has_cabin feature. Age needs imputation before modeling. Embarked has minimal impact.
================================================================================
"""
        )


class MockDatasetOverviewComponent(MockComponent):
    def __init__(self):
        super().__init__(
            name="DatasetOverviewComponent",
            justification="The Dataset Overview component establishes a structural and statistical baseline for the entire analysis pipeline.",
            summary={
                "dataset_shape": {"rows": 891, "columns": 12},
                "type_distribution": {"int64": 5, "str": 5, "float64": 2},
                "potential_identifiers": ["PassengerId", "Name"]
            },
            full_summary="""
================================================================================
📋 COMPONENT SUMMARY
================================================================================
Dataset has 891 rows and 12 columns with mixed types. PassengerId and Name detected as potential identifiers - exclude from ML features to prevent leakage.
================================================================================
"""
        )


class MockExactDuplicateComponent(MockComponent):
    def __init__(self):
        super().__init__(
            name="ExactDuplicateDetectionComponent",
            justification="Exact duplicate detection is critical for ensuring the statistical validity of downstream machine learning experiments.",
            summary={
                "duplicate_ratio": 0.0,
                "duplicate_groups": 0,
                "risk_level": "none"
            },
            full_summary="""
================================================================================
📋 COMPONENT SUMMARY
================================================================================
No exact duplicates found. Data integrity is good with unique observations. No risk of data leakage from duplicated records.
================================================================================
"""
        )


class MockOutlierDetectionComponent(MockComponent):
    def __init__(self):
        super().__init__(
            name="OutlierDetectionComponent",
            justification="Isolation Forest-based multivariate outlier detection with SHAP explanations.",
            summary={
                "outlier_ratio": 0.286,
                "example_outliers": [
                    {
                        "row_index": 7,
                        "outlier_score": 0.561,
                        "top_contributing_features": {
                            "SibSp": 2.005,
                            "Age": 1.464,
                            "Parch": 0.557
                        }
                    },
                    {
                        "row_index": 8,
                        "outlier_score": 0.525,
                        "top_contributing_features": {
                            "Parch": 2.636,
                            "Survived": 0.909,
                            "Age": 0.406
                        }
                    }
                ],
                "model_params": {"n_estimators": 200, "contamination": "auto"}
            },
            full_summary="""
================================================================================
🤖 LLM EXAMPLE EXPLANATIONS
================================================================================

1. Row 7 (Score: 0.561)
   Unusually high SibSp (4 siblings/spouses) combined with young age creates multivariate anomaly. Likely genuine rare case.

2. Row 8 (Score: 0.525)
   High Parch value with unusual survival pattern. Worth investigating but likely represents real family group.

================================================================================
📋 COMPONENT SUMMARY
================================================================================
28.6% of rows flagged as outliers. Most anomalies driven by family size features (SibSp, Parch). Review flagged records but most appear to be genuine rare cases rather than data errors.
================================================================================
"""
        )


class MockCategoricalOutlierComponent(MockComponent):
    def __init__(self):
        super().__init__(
            name="CategoricalOutlierDetectionComponent",
            justification="Frequency-based categorical outlier detection with adaptive thresholds.",
            summary={
                "outlier_ratio": 0.286,
                "skipped_columns": [{"column": "Name", "unique_ratio": 1.0}],
                "example_outliers": [
                    {
                        "row_index": 529,
                        "column": "Ticket",
                        "value": "29104",
                        "frequency": 0.001
                    },
                    {
                        "row_index": 516,
                        "column": "Ticket",
                        "value": "C.A. 34260",
                        "frequency": 0.001
                    }
                ]
            },
            full_summary="""
================================================================================
🤖 LLM EXAMPLE EXPLANATIONS
================================================================================

1. Column: Ticket, Value: '29104'
   Unique ticket number appearing only once. Expected for ticket IDs, not a data quality issue.

2. Column: Ticket, Value: 'C.A. 34260'
   Rare ticket format. May indicate special booking or different ticket class.

================================================================================
📋 COMPONENT SUMMARY
================================================================================
28.6% rare categorical values found, mostly in Ticket column. This is expected since ticket numbers are semi-unique identifiers. Consider excluding Ticket from categorical analysis.
================================================================================
"""
        )


class MockDistributionModelingComponent(MockComponent):
    def __init__(self):
        super().__init__(
            name="DistributionModelingComponent",
            justification="Autoencoder-based distribution modeling detects complex multivariate anomalies.",
            summary={
                "anomaly_ratio": 0.112,
                "reconstruction_threshold": 0.85,
                "top_anomalies": [
                    {"row_index": 258, "reconstruction_error": 1.24},
                    {"row_index": 679, "reconstruction_error": 1.18}
                ]
            },
            full_summary="""
================================================================================
🤖 LLM EXAMPLE EXPLANATIONS
================================================================================

1. Row 258 (Error: 1.24)
   High reconstruction error indicates unusual feature combination. Possibly VIP or crew member.

2. Row 679 (Error: 1.18)
   Record doesn't fit learned patterns. May be data entry error or genuine rare case.

================================================================================
📋 COMPONENT SUMMARY
================================================================================
11.2% of records show distribution anomalies. Autoencoder detected unusual feature combinations that deviate from typical passenger profiles. Investigate high-error records for potential data issues.
================================================================================
"""
        )


class MockCompositeQualityScoreComponent(MockComponent):
    def __init__(self):
        super().__init__(
            name="CompositeQualityScoreComponent",
            justification="Aggregates multiple quality signals into a single interpretable score.",
            summary={
                "overall_quality_score": 0.72,
                "component_scores": {
                    "completeness": 0.85,
                    "uniqueness": 0.95,
                    "consistency": 0.68,
                    "validity": 0.78
                },
                "risk_level": "medium"
            },
            full_summary="""
================================================================================
📋 COMPONENT SUMMARY
================================================================================
Overall quality score: 72/100 (Medium). Strengths: no duplicates (95%), good completeness (85%). Areas for improvement: consistency (68%) due to Cabin missingness, validity (78%) due to outliers. Prioritize handling missing Age values and Cabin column strategy.
================================================================================
"""
        )


class MockLabelNoiseDetectionComponent(MockComponent):
    def __init__(self):
        super().__init__(
            name="LabelNoiseDetectionComponent",
            justification="Identifies potentially mislabeled records that could harm model training.",
            summary={
                "label_column": "Survived",
                "noise_ratio": 0.034,
                "suspicious_records": 30,
                "confidence_threshold": 0.8
            },
            full_summary="""
================================================================================
🤖 LLM EXAMPLE EXPLANATIONS
================================================================================

1. Row 45 - Label: Survived=0
   Female, 1st class passenger labeled as deceased. High survival probability for this profile suggests potential mislabel.

2. Row 156 - Label: Survived=1
   Male, 3rd class passenger labeled as survived. Low survival probability for this profile warrants verification.

================================================================================
📋 COMPONENT SUMMARY
================================================================================
3.4% of labels (30 records) appear potentially mislabeled. Most suspicious cases involve labels inconsistent with demographic survival patterns. Review flagged records before model training.
================================================================================
"""
        )


class MockRelationalConsistencyComponent(MockComponent):
    def __init__(self):
        super().__init__(
            name="RelationalConsistencyComponent",
            justification="Detects logical inconsistencies between related columns.",
            summary={
                "inconsistencies_found": 5,
                "rules_checked": 8,
                "examples": [
                    {"rule": "Age > 0", "violations": 0},
                    {"rule": "Fare >= 0", "violations": 0},
                    {"rule": "SibSp + Parch < 15", "violations": 0}
                ]
            },
            full_summary="""
================================================================================
🤖 LLM EXAMPLE EXPLANATIONS
================================================================================

1. Embarked-Fare inconsistency (3 violations)
   Passengers with Cherbourg embarkation have unusually low fares. May be discounted tickets or data entry errors.

2. Cabin-Pclass mismatch (2 violations)
   Cabin prefixes don't match expected passenger class. Could be upgrades or recording errors.

================================================================================
📋 COMPONENT SUMMARY
================================================================================
5 consistency violations found across 8 rules checked. Most rules pass validation. Minor issues in Embarked-Fare and Cabin-Pclass relationships warrant investigation.
================================================================================
"""
        )


class MockNearDuplicateDetectionComponent(MockComponent):
    def __init__(self):
        super().__init__(
            name="NearDuplicateDetectionComponent",
            justification="Identifies records that are very similar but not exactly identical.",
            summary={
                "near_duplicate_pairs": 12,
                "similarity_threshold": 0.85,
                "affected_rows": 24
            },
            full_summary="""
================================================================================
🤖 LLM EXAMPLE EXPLANATIONS
================================================================================

1. Pair 123-124 (Similarity: 94%)
   Same ticket number, similar names. Likely family members traveling together. Keep both records.

2. Pair 456-457 (Similarity: 91%)
   Same cabin and demographics. Family group booking expected for Titanic data. No action needed.

================================================================================
📋 COMPONENT SUMMARY
================================================================================
12 near-duplicate pairs found affecting 24 rows (2.7%). Most appear to be family members with shared tickets/cabins rather than data errors. No deduplication recommended.
================================================================================
"""
        )


class MockLLMDatasetSummaryComponent(MockComponent):
    def __init__(self):
        super().__init__(
            name="LLMDatasetSummaryComponent",
            justification="Synthesizes all analysis results into a coherent overview with factual metrics and AI-powered insights.",
            summary={
                "total_components_analyzed": 10,
                "total_issues_detected": 8,
                "overall_risk_level": "medium",
                "llm_summary_available": True
            },
            full_summary="""
================================================================================
📊 COMPONENT METRICS OVERVIEW
================================================================================
📌 MissingValues
  • Columns with missing values: 3

📌 DatasetOverview
  • Dataset shape: {'rows': 891, 'columns': 12}

📌 ExactDuplicateDetection
  • Duplicate ratio: 0.0%
  • Risk level: none

📌 OutlierDetection
  • Outlier ratio: 28.6%

📌 CategoricalOutlierDetection
  • Categorical outlier ratio: 28.6%

📌 DistributionModeling
  • Mean reconstruction error: 0.2725
  • High error ratio: 11.2%

📌 CompositeQualityScore
  • Data readiness score: 90.5%

📌 LabelNoiseDetection
  • Label noise ratio: 3.4%
  • Suspicious samples: 30

================================================================================
🤖 AI QUALITY ASSESSMENT
================================================================================
✅ STRENGTHS
• No duplicate records - data integrity maintained
• High readiness score (90.5%) - suitable for ML
• Low label noise (3.4%) - labels mostly reliable

❌ ISSUES  
• High missing rate in Cabin column (77%)
• Significant outlier ratio (28.6%)
• Age column needs imputation (20% missing)

🎯 PRIORITY ACTION
Handle Cabin column first (drop or create has_cabin feature), then impute Age values.
================================================================================
"""
        )


def main():
    # Create mock components (no real analysis needed)
    components = [
        MockMissingValuesReport(),
        MockDatasetOverviewComponent(),
        MockExactDuplicateComponent(),
        MockOutlierDetectionComponent(),
        MockCategoricalOutlierComponent(),
        MockDistributionModelingComponent(),
        MockCompositeQualityScoreComponent(),
        MockLabelNoiseDetectionComponent(),
        MockRelationalConsistencyComponent(),
        MockNearDuplicateDetectionComponent(),
        MockLLMDatasetSummaryComponent(),
    ]

    # Generate HTML report
    generator = HTMLReportGenerator("AEDA Data Quality Report (Mock Data)")
    report_path = generator.generate(components, "data_quality_report_mock.html")

    # Open in browser
    webbrowser.open('file://' + os.path.realpath(report_path))

    print(f"\n✅ Mock report generated: {report_path}")
    print("Click 'Download as PDF' button in the browser to save as PDF")


if __name__ == "__main__":
    main()
