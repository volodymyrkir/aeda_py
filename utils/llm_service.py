"""
LLM Service Module

This module provides a unified interface for LLM-based explanations in dataset analysis.
It supports local models via Hugging Face transformers (downloaded on first use).

The service can work offline with local models.

Usage:
    from utils.llm_service import LLMService

    # Will use local Hugging Face model (singleton - same instance returned)
    llm = LLMService.get_instance()

    # Or specify a different model on first call
    llm = LLMService.get_instance(model_name="microsoft/phi-2")

    # Legacy create() still works but returns singleton
    llm = LLMService.create()
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Global singleton instance
_llm_service_instance: Optional["LLMService"] = None


class LLMProvider(Enum):
    """Supported LLM providers."""
    LOCAL_HF = "local_hf"  # Local via Hugging Face transformers
    NONE = "none"  # No LLM available


@dataclass
class LLMConfig:
    provider: LLMProvider
    model_name: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.3
    timeout: int = 30


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text completion."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available."""
        pass

    def _format_data_analysis_prompt(
        self,
        component_name: str,
        context_data: Dict[str, Any],
        question: str
    ) -> str:
        """Format a prompt for data analysis explanation."""
        return f"""You are a data quality analyst assistant. Analyze the following information and provide a clear, concise explanation.

                Component: {component_name}
                Context Data:
                {json.dumps(context_data, indent=2, default=str)}
                
                Question: {question}
                
                Provide a brief, actionable explanation (2-4 sentences). Focus on:
                1. What the issue is
                2. Why it might have occurred
                3. Potential impact on data quality or ML models"""


class LocalHuggingFaceProvider(BaseLLMProvider):
    """
    Local Hugging Face transformers provider.

    Uses a small model like TinyLlama-1.1B-Chat-v1.0 by default (~2GB download, ~1.1B params).
    First run will download the model automatically (may take several minutes).
    Requires: pip install transformers torch accelerate
    """

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

        logger.info(f"Loading local model: {model_name}")
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
            print(f"   Device: CUDA (GPU)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float16
            print(f"   Device: MPS (Apple Silicon)")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print(f"   Device: CPU")

        logger.info(f"Using device: {device}")

        self._pipeline = pipeline(
            "text-generation",
            model=model_name,
            dtype=torch_dtype,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True
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

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self._load_model()

        formatted = f"<|system|>\n{system_prompt or ''}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"

        outputs = self._pipeline(
            formatted,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            pad_token_id=self._pipeline.tokenizer.eos_token_id
        )

        generated = outputs[0]["generated_text"]
        response = generated[len(formatted):].strip()
        return response


class LLMService:
    """
    Main LLM service class providing unified access to local LLM provider.

    Uses local Hugging Face model by default.
    Uses singleton pattern - model is loaded once and reused across all calls.
    """

    SYSTEM_PROMPT = """You are a data quality analyst. Explain data issues concisely (2-3 sentences max). Focus on what the issue is and why it matters."""

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        self._enabled = True

    @classmethod
    def get_instance(
        cls,
        model_name: Optional[str] = None,
        **kwargs
    ) -> "LLMService":
        """
        Get the singleton LLM service instance (creates one if it doesn't exist).

        The model is loaded once and reused across all calls, avoiding repeated
        loading which can be slow.

        Args:
            model_name: Specific model to use (only used on first call)
            **kwargs: Additional config options (only used on first call)

        Returns:
            LLMService singleton instance
        """
        global _llm_service_instance

        if _llm_service_instance is None:
            config = LLMConfig(
                provider=LLMProvider.LOCAL_HF,
                model_name=model_name,
                **kwargs
            )

            llm_provider = LocalHuggingFaceProvider(config)

            if not llm_provider.is_available():
                raise ImportError("Failed to load the module")

            _llm_service_instance = cls(llm_provider)

        return _llm_service_instance

    @classmethod
    def create(
        cls,
        model_name: Optional[str] = None,
        **kwargs
    ) -> "LLMService":
        """
        Create/get an LLM service instance (uses singleton pattern).

        Note: This is now an alias for get_instance() to maintain backward compatibility.

        Args:
            model_name: Specific model to use (default: "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            **kwargs: Additional config options

        Returns:
            LLMService singleton instance
        """
        return cls.get_instance(model_name=model_name, **kwargs)

    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance (useful for testing or changing models).
        Next call to get_instance() will create a new instance.
        """
        global _llm_service_instance
        _llm_service_instance = None
        print("🔧 LLM Service singleton reset")

    @property
    def is_available(self) -> bool:
        """Check if LLM service is available and enabled."""
        return self._enabled and self.provider.is_available()

    def disable(self):
        """Disable LLM explanations."""
        self._enabled = False

    def enable(self):
        """Enable LLM explanations."""
        self._enabled = True

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using the configured provider."""
        if not self._enabled:
            return "[LLM explanations disabled]"

        try:
            return self.provider.generate(
                prompt,
                system_prompt or self.SYSTEM_PROMPT
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"[LLM explanation failed: {str(e)}]"

    def explain_outlier(
        self,
        row_data: Dict[str, Any],
        outlier_score: float,
        contributing_features: Dict[str, float],
        dataset_context: Optional[str] = None
    ) -> str:
        relevant_data = {k: row_data[k] for k in list(contributing_features.keys())[:3] if k in row_data}

        prompt = f"""Outlier detected. Score: {outlier_score:.3f}. Key features: {', '.join(contributing_features.keys())}. Values: {json.dumps(relevant_data, default=str)}. Explain in 2 sentences why this is anomalous."""

        return self.generate(prompt)

    def explain_near_duplicate(
        self,
        row_a: Dict[str, Any],
        row_b: Dict[str, Any],
        similarity_score: float,
        matching_columns: List[str],
        differing_columns: List[str]
    ) -> str:
        """Generate an explanation for why two rows are near-duplicates."""
        prompt = f"""Explain why these two records are flagged as near-duplicates.

        Record A: {json.dumps(row_a, indent=2, default=str)}
        
        Record B: {json.dumps(row_b, indent=2, default=str)}
        
        Similarity Score: {similarity_score:.2%}
        Matching Columns: {', '.join(matching_columns)}
        Differing Columns: {', '.join(differing_columns)}
        
        Provide a brief explanation (2-3 sentences) of:
        1. What makes these records nearly identical
        2. Possible causes (data entry variation, ETL issues, temporal versions)
        3. Recommendation for handling (merge, keep both, investigate)"""

        return self.generate(prompt)

    def explain_consistency_violation(
        self,
        violation_type: str,
        affected_columns: List[str],
        example_violations: List[Dict[str, Any]],
        violation_ratio: float
    ) -> str:
        prompt = f"""Consistency issue: {violation_type}. Columns: {', '.join(affected_columns)}. Rate: {violation_ratio:.1%}. Explain in 2 sentences."""

        return self.generate(prompt)

    def explain_label_noise(
        self,
        row_data: Dict[str, Any],
        current_label: Any,
        suggested_label: Any,
        confidence: float
    ) -> str:
        """Generate an explanation for why a label might be noisy/incorrect."""
        prompt = f"""Explain why this record's label might be incorrect (label noise).

        Row Data: {json.dumps(row_data, indent=2, default=str)}
        
        Current Label: {current_label}
        Suggested Label: {suggested_label}
        Confidence: {confidence:.2%}
        
        Provide a brief explanation (2-3 sentences) of:
        1. Why the current label seems inconsistent with the features
        2. Possible labeling errors (human error, ambiguous cases)
        3. Recommendation (correct label, investigate, keep as is)"""

        return self.generate(prompt)

    def explain_distribution_anomaly(
        self,
        column_name: str,
        detected_distribution: str,
        anomaly_details: Dict[str, Any]
    ) -> str:
        """Generate an explanation for a distribution anomaly."""
        prompt = f"""Explain this column's distribution characteristics.

        Column: {column_name}
        Detected Distribution: {detected_distribution}
        Analysis Details: {json.dumps(anomaly_details, indent=2, default=str)}
        
        Provide a brief explanation (2-3 sentences) of:
        1. What the distribution tells us about this data
        2. Any concerns for statistical analysis or ML modeling
        3. Recommendations for handling (transformation, outlier treatment)"""

        return self.generate(prompt)

    def generate_dataset_summary(
        self,
        component_results: Dict[str, Dict[str, Any]],
        dataset_info: Dict[str, Any]
    ) -> str:
        condensed_results = {}
        for component, data in component_results.items():
            condensed_results[component] = {
                k: v for k, v in data.items()
                if k not in ['llm_explanation', 'llm_explanations'] and not isinstance(v, str) or len(str(v)) < 100
            }

        prompt = f"""Dataset: {dataset_info['num_rows']} rows, {dataset_info['num_columns']} cols. 

Analysis findings: {json.dumps(condensed_results, indent=1, default=str)[:800]}

Provide concise assessment:
1. Overall Data Quality (1-2 sentences - is data ready for ML?)
2. Top 3 Critical Issues (bullet points)
3. Key Recommendation (1 sentence - what to fix first)

Focus on actionable insights about data quality, not basic statistics."""

        return self.generate(prompt, system_prompt="You are a data quality expert. Be concise and actionable. Focus on issues and recommendations, not basic statistics.")

    def generate_component_summary(
        self,
        component_name: str,
        metrics: Dict[str, Any],
        findings: str
    ) -> str:
        prompt = f"""Component: {component_name}
Metrics: {json.dumps(metrics, default=str)}
Findings: {findings}

In 2-3 sentences: What does this analysis reveal about data quality? Any concerns or recommendations?"""

        return self.generate(prompt, system_prompt="You are a data quality analyst. Be brief and actionable.")

