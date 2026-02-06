# AEDA - Automated Exploratory Data Analysis

A comprehensive data quality analysis library with LLM-powered insights.

## Installation

```bash
pip install aeda
```

Or install from source:

```bash
git clone https://github.com/aeda/aeda.git
cd aeda
pip install -e .
```

## Quick Start

```python
from aeda import AEDAAnalyzer, AnalysisConfig

# Basic usage with default settings
analyzer = AEDAAnalyzer("your_dataset.csv")
analyzer.analyze()
report_path = analyzer.generate_report()
analyzer.open_report()

# With custom configuration
config = AnalysisConfig(
    target_column="label",
    use_llm=True,
    output_path="my_report.html",
    report_title="My Data Quality Report"
)
analyzer = AEDAAnalyzer("your_dataset.parquet", config=config)
analyzer.analyze().generate_report()

# Access results programmatically
results = analyzer.get_results()
outlier_summary = analyzer.get_summary("OutlierDetectionComponent")
```

## Configuration Options

```python
from aeda import AnalysisConfig

config = AnalysisConfig(
    # Component toggles
    dataset_overview=True,
    missing_values=True,
    exact_duplicates=True,
    near_duplicates=True,
    outlier_detection=True,
    categorical_outliers=True,
    label_noise=True,
    relational_consistency=True,
    distribution_modeling=True,
    composite_quality=True,
    dataset_summary=True,
    
    # ML settings
    target_column="Survived",  # Required for label noise detection
    
    # LLM settings
    use_llm=True,
    
    # Engine settings
    engine="auto",  # "auto", "pandas", or "polars"
    
    # Output settings
    output_path="data_quality_report.html",
    report_title="AEDA Data Quality Report"
)
```

## Using with DataFrames

```python
import pandas as pd
from aeda import AEDAAnalyzer

df = pd.read_csv("data.csv")
analyzer = AEDAAnalyzer(df)
analyzer.analyze().generate_report()
```

## Components

- **Dataset Overview**: Basic statistics and structure
- **Missing Values**: Missing data analysis
- **Exact Duplicates**: Duplicate row detection
- **Near Duplicates**: Similar record detection using MinHash LSH
- **Outlier Detection**: Isolation Forest with SHAP explanations
- **Categorical Outliers**: Rare category detection
- **Label Noise**: Confident Learning framework for mislabeled data
- **Relational Consistency**: Constraint validation
- **Distribution Modeling**: Autoencoder-based anomaly detection
- **Composite Quality Score**: Overall data readiness assessment
- **Dataset Summary**: LLM-powered analysis summary

## License

MIT
