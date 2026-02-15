import pandas as pd
from aeda import AEDAAnalyzer, AnalysisConfig

print("Testing DataFrame initialization...")

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': ['x', 'y', 'z', 'x', 'y']
})

config = AnalysisConfig(
    use_llm=False,
    output_path="test_dataframe_init.html",
    distribution_modeling=False,
    label_noise=False
)

print("Creating analyzer from DataFrame...")
analyzer = AEDAAnalyzer(df, config=config, verbose=True)

print("\nRunning analysis...")
analyzer.analyze()

print("\nGenerating report...")
report_path = analyzer.generate_report()

print(f"\n✓ Test completed successfully!")
print(f"Report saved to: {report_path}")

engine_info = analyzer.get_engine_info()
print(f"\nEngine info:")
print(f"  Engine: {engine_info['engine']}")
print(f"  Recommended: {engine_info['recommended_engine']}")
print(f"  Size: {engine_info['file_size_mb']:.2f} MB")
