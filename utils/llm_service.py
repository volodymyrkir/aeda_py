import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

_llm_service_instance: Optional["LLMService"] = None


class LLMProvider(Enum):
    LOCAL_HF = "local_hf"
    NONE = "none"


@dataclass
class LLMConfig:
    provider: LLMProvider
    model_name: Optional[str] = None
    max_tokens: int = 13
    temperature: float = 0.3
    timeout: int = 30


class BaseLLMProvider(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    def _format_data_analysis_prompt(
        self,
        component_name: str,
        context_data: Dict[str, Any],
        question: str,
    ) -> str:
        return f"""You are a data quality analyst assistant. Analyze the following information and provide a clear, concise explanation.
Component: {component_name}
Context Data:
{json.dumps(context_data, indent=2, default=str)}
Question: {question}
Provide a brief, actionable explanation (2-4 sentences). Focus on:

What the issue is
Why it might have occurred
Potential impact on data quality or ML models"""


class LocalHuggingFaceProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._pipeline = None

    def _load_model(self):
        if self._pipeline is not None:
            return

        try:
            import torch
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers and torch not installed. Run: pip install transformers torch accelerate"
            )

        model_name = self.config.model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32

        self._pipeline = pipeline(
            "text-generation",
            model=model_name,
            dtype=torch_dtype,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True,
        )

        if device == "cpu":
            self._pipeline.model = self._pipeline.model.to("cpu")

    def is_available(self) -> bool:
        try:
            import torch
            from transformers import pipeline

            return True
        except ImportError:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        self._load_model()

        formatted = (
            f"<|system|>\n{system_prompt or ''}<|end|>\n"
            f"<|user|>\n{prompt}<|end|>\n"
            f"<|assistant|>"
        )

        outputs = self._pipeline(
            formatted,
            max_new_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,
        )

        generated = outputs[0]["generated_text"]
        response = generated[len(formatted) :].strip()
        return response


class LLMService:
    SYSTEM_PROMPT = (
        "You are a data quality analyst. Be extremely concise (2 sentences max). "
        "No code, no examples. Focus on: what the issue means and how to fix it semantically."
    )

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        self._enabled = True

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None, **kwargs) -> "LLMService":
        global _llm_service_instance

        if _llm_service_instance is None:
            config = LLMConfig(
                provider=LLMProvider.LOCAL_HF,
                model_name=model_name,
                **kwargs,
            )

            llm_provider = LocalHuggingFaceProvider(config)

            if not llm_provider.is_available():
                raise ImportError("Failed to load the module")

            _llm_service_instance = cls(llm_provider)

        return _llm_service_instance

    @classmethod
    def create(cls, model_name: Optional[str] = None, **kwargs) -> "LLMService":
        return cls.get_instance(model_name=model_name, **kwargs)

    @classmethod
    def reset_instance(cls):
        global _llm_service_instance
        _llm_service_instance = None

    @property
    def is_available(self) -> bool:
        return self._enabled and self.provider.is_available()

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        if not self._enabled:
            return "[LLM explanations disabled]"

        try:
            return self.provider.generate(
                prompt,
                system_prompt or self.SYSTEM_PROMPT,
                max_tokens,
            )
        except Exception as e:
            return f"[LLM explanation failed: {str(e)}]"

    def explain_outlier(
        self,
        row_data: Dict[str, Any],
        outlier_score: float,
        contributing_features: Dict[str, float],
        dataset_context: Optional[str] = None,
    ) -> str:
        relevant_data = {
            k: row_data[k]
            for k in list(contributing_features.keys())[:3]
            if k in row_data
        }

        prompt = (
            f"Outlier detected. Score: {outlier_score:.3f}. "
            f"Key features: {', '.join(contributing_features.keys())}. "
            f"Values: {json.dumps(relevant_data, default=str)}. "
            f"Explain in 2 sentences why this is anomalous."
        )

        return self.generate(prompt)

    def explain_near_duplicate(
        self,
        row_a: Dict[str, Any],
        row_b: Dict[str, Any],
        similarity_score: float,
        matching_columns: List[str],
        differing_columns: List[str],
    ) -> str:
        prompt = (
            f"Records {similarity_score:.0%} similar. "
            f"Same: {', '.join(matching_columns[:3])}. "
            f"Different: {', '.join(differing_columns[:2])}. "
            f"One sentence: why similar and should merge/keep/investigate?"
        )

        return self.generate(prompt)

    def explain_consistency_violation(
        self,
        violation_type: str,
        affected_columns: List[str],
        example_violations: List[Dict[str, Any]],
        violation_ratio: float,
    ) -> str:
        prompt = (
            f"{violation_type} issue in {', '.join(affected_columns)} ({violation_ratio:.1%} of rows). "
            f"One sentence: what it means. One sentence: how to fix semantically."
        )

        return self.generate(prompt)

    def explain_label_noise(
        self,
        row_data: Dict[str, Any],
        current_label: Any,
        suggested_label: Any,
        confidence: float,
    ) -> str:
        key_features = {
            k: v for k, v in list(row_data.items())[:4] if k != "Name"
        }

        prompt = (
            f"Potential mislabel. Current: {current_label}, "
            f"Suggested: {suggested_label}, Confidence: {confidence:.1%}. "
            f"Features: {json.dumps(key_features, default=str)}. "
            f"Explain in 2 sentences why label may be wrong."
        )

        return self.generate(prompt)

    def explain_distribution_anomaly(
        self,
        column_name: str,
        detected_distribution: str,
        anomaly_details: Dict[str, Any],
    ) -> str:
        error = anomaly_details.get("reconstruction_error", 0)
        threshold = anomaly_details.get("threshold", 0)
        features = anomaly_details.get("contributing_features", {})
        top_features = dict(list(features.items())[:3])

        prompt = (
            f"Distribution anomaly in {column_name} ({detected_distribution}). "
            f"Reconstruction error: {error:.3f} (threshold: {threshold:.3f}). "
            f"Top deviating features: {json.dumps(top_features, default=str)}. "
            f"In 2 sentences: why does this record deviate and is it a data error or rare case?"
        )

        return self.generate(prompt)

    def generate_dataset_summary(
        self,
        component_results: Dict[str, Dict[str, Any]],
        dataset_info: Dict[str, Any],
    ) -> str:
        condensed_results = {}

        for component, data in component_results.items():
            condensed_results[component] = {
                k: v
                for k, v in data.items()
                if k not in {"llm_explanation", "llm_explanations"}
                and (not isinstance(v, str) or len(v) < 100)
            }

        prompt = f"""Dataset: {dataset_info['num_rows']} rows, {dataset_info['num_columns']} cols.
Analysis findings: {json.dumps(condensed_results, indent=1, default=str)[:800]}
Provide concise assessment:

Overall Data Quality (1-2 sentences - is data ready for ML?)
Top 3 Critical Issues (bullet points)
Key Recommendation (1 sentence - what to fix first)

Focus on actionable insights about data quality, not basic statistics.
"""

        return self.generate(
            prompt,
            system_prompt=(
                "You are a data quality expert. Be concise and actionable. "
                "Focus on issues and recommendations, not basic statistics."
            ),
            max_tokens=300,
        )

    def generate_component_summary(
        self,
        component_name: str,
        metrics: Dict[str, Any],
        findings: str,
    ) -> str:
        prompt = f"""Component: {component_name}
Metrics: {json.dumps(metrics, default=str)}
Findings: {findings}
In 2-3 sentences: What does this analysis reveal about data quality? Any concerns or recommendations?
"""

        return self.generate(
            prompt,
            system_prompt="You are a data quality analyst. Be brief and actionable.",
        )
