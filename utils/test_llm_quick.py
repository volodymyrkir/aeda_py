"""Quick test to verify LLM optimizations"""

from utils.llm_service import LLMService

llm = LLMService.get_instance()

print("Testing concise prompt generation...")
print("\n" + "="*60)
print("Test 1: Outlier Explanation")
print("="*60)

result = llm.explain_outlier(
    row_data={"Age": 65, "Fare": 512.3, "SibSp": 3, "Parch": 2, "Pclass": 1},
    outlier_score=0.85,
    contributing_features={"Age": 2.1, "Fare": 3.5, "SibSp": 1.2},
    dataset_context="Dataset has 891 rows"
)

print(f"Result length: {len(result)} chars")
print(f"Result: {result}")

print("\n" + "="*60)
print("Test 2: Consistency Violation")
print("="*60)

result2 = llm.explain_consistency_violation(
    violation_type="uniqueness",
    affected_columns=["PassengerId"],
    example_violations=[],
    violation_ratio=0.05
)

print(f"Result length: {len(result2)} chars")
print(f"Result: {result2}")

print("\n" + "="*60)
print("Test 3: Dataset Summary")
print("="*60)

result3 = llm.generate_dataset_summary(
    component_results={
        "outlier_detection": {"outlier_ratio": 0.28},
        "missing_values": {"num_columns_with_missing": 3}
    },
    dataset_info={"num_rows": 891, "num_columns": 12, "memory_mb": 0.1}
)

print(f"Result length: {len(result3)} chars")
print(f"Result: {result3}")

print("\nAll tests completed!")
