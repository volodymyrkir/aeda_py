import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from utils.consts import (
    DEFAULT_LLM_MODEL, DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_TEMPERATURE, DEFAULT_LLM_TIMEOUT,
    DEFAULT_GGUF_REPO, DEFAULT_GGUF_FILENAME, DEFAULT_GGUF_CONTEXT_LENGTH, DEFAULT_GGUF_N_THREADS,
)

logger = logging.getLogger(__name__)

_llm_service_instance: Optional["LLMService"] = None


class LLMProvider(Enum):
    LOCAL_GGUF = "local_gguf"
    LOCAL_HF = "local_hf"
    NONE = "none"


@dataclass
class LLMConfig:
    provider: LLMProvider
    model_name: Optional[str] = None
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    temperature: float = DEFAULT_LLM_TEMPERATURE
    timeout: int = DEFAULT_LLM_TIMEOUT
    gguf_repo: str = DEFAULT_GGUF_REPO
    gguf_filename: str = DEFAULT_GGUF_FILENAME
    gguf_context_length: int = DEFAULT_GGUF_CONTEXT_LENGTH
    gguf_n_threads: int = DEFAULT_GGUF_N_THREADS


class BaseLLMProvider(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class LocalGGUFProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._llm = None

    def is_available(self) -> bool:
        try:
            from llama_cpp import Llama
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._llm is not None:
            return

        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "aeda", "gguf")
        os.makedirs(cache_dir, exist_ok=True)

        model_path = os.path.join(cache_dir, self.config.gguf_filename)
        if not os.path.exists(model_path):
            print(f"\n{'=' * 60}")
            print("DOWNLOADING GGUF MODEL")
            print(f"{'=' * 60}")
            print(f"   Repo:  {self.config.gguf_repo}")
            print(f"   File:  {self.config.gguf_filename}")
            print(f"   Cache: {cache_dir}")
            print("   This is a one-time download...")
            model_path = hf_hub_download(
                repo_id=self.config.gguf_repo,
                filename=self.config.gguf_filename,
                local_dir=cache_dir,
            )
            print("   ✅ Download complete!")
            print(f"{'=' * 60}\n")
        else:
            print(f"\n{'=' * 60}")
            print("🔧 LOADING GGUF MODEL")
            print(f"{'=' * 60}")
            print(f"   Model: {self.config.gguf_filename}")
            print(f"   Path:  {model_path}")

        import multiprocessing
        n_threads = self.config.gguf_n_threads or multiprocessing.cpu_count()

        self._llm = Llama(
            model_path=model_path,
            n_ctx=self.config.gguf_context_length,
            n_threads=n_threads,
            n_gpu_layers=0,
            verbose=False,
        )
        print("   ✅ Model loaded successfully! (GGUF / llama-cpp-python)")
        print(f"{'=' * 60}\n")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        self._load_model()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
            stop=["</s>", "<|im_end|>", "<|end|>"],
        )
        return response["choices"][0]["message"]["content"].strip()


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

        model_name = self.config.model_name or DEFAULT_LLM_MODEL

        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32

        print(f"\n{'=' * 60}")
        print("🔧 LOADING LLM MODEL (HuggingFace)")
        print(f"{'=' * 60}")
        print(f"   Model:  {model_name}")
        print(f"   Device: {device.upper()}")
        print("   Status: Loading model... (this may take a minute)")

        self._pipeline = pipeline(
            "text-generation",
            model=model_name,
            dtype=torch_dtype,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True,
        )

        if device == "cpu":
            self._pipeline.model = self._pipeline.model.to("cpu")

        print("   Status: ✅ Model loaded successfully!")
        print(f"{'=' * 60}\n")

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
        response = generated[len(formatted):].strip()
        return response


def _detect_best_provider() -> LLMProvider:
    try:
        from llama_cpp import Llama
        return LLMProvider.LOCAL_GGUF
    except ImportError:
        pass
    try:
        import torch
        from transformers import pipeline
        return LLMProvider.LOCAL_HF
    except ImportError:
        pass
    return LLMProvider.NONE


class LLMService:
    SYSTEM_PROMPT = "You are a data quality specialist. Answer in exactly 2 sentences, max 25 words total. State only the cause. No lists. No code. No examples."

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        self._enabled = True

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None, **kwargs) -> "LLMService":
        global _llm_service_instance

        if _llm_service_instance is None:
            best = _detect_best_provider()

            config = LLMConfig(
                provider=best,
                model_name=model_name,
            )

            if best == LLMProvider.LOCAL_GGUF:
                provider = LocalGGUFProvider(config)
            elif best == LLMProvider.LOCAL_HF:
                provider = LocalHuggingFaceProvider(config)
            else:
                raise ImportError(
                    "No LLM backend found. Install llama-cpp-python (recommended) or transformers+torch.\n"
                    "  Fast (recommended): pip install llama-cpp-python\n"
                    "  Fallback:           pip install transformers torch accelerate"
                )

            _llm_service_instance = cls(provider)

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
        top_features = list(contributing_features.keys())[:3]
        values_parts = []
        for k in top_features:
            if k in row_data:
                val = row_data[k]
                score = contributing_features[k]
                values_parts.append(f"{k}={val} (deviation={score:.2f})")
        values_str = ", ".join(values_parts) if values_parts else "unknown features"
        prompt = (
            f"Record has anomalous values: {values_str}. "
            f"Outlier score: {outlier_score:.2f}. "
            f"In exactly 2 sentences (max 20 words total): state which values are extreme and why they stand out."
        )
        return self.generate(prompt, max_tokens=80)

    def explain_near_duplicate(
        self,
        row_a: Dict[str, Any],
        row_b: Dict[str, Any],
        similarity_score: float,
        matching_columns: List[str],
        differing_columns: List[str],
    ) -> str:
        diffs = []
        for col in differing_columns[:2]:
            val_a = row_a.get(col, "?")
            val_b = row_b.get(col, "?")
            diffs.append(f"{col}: {val_a} vs {val_b}")
        diff_str = "; ".join(diffs) if diffs else "minor differences"
        prompt = f"{similarity_score:.0%} similar records. Differences: {diff_str}. Why duplicates? 2 sentences, max 25 words."
        return self.generate(prompt, max_tokens=100)

    def explain_label_noise(
        self,
        row_data: Dict[str, Any],
        current_label: Any,
        suggested_label: Any,
        confidence: float,
        model_prediction: Optional[Any] = None,
        model_confidence: Optional[float] = None,
        current_label_prob: Optional[float] = None,
        class_noise_rate: Optional[float] = None,
        confused_with: Optional[List[str]] = None,
    ) -> str:
        facts = []

        if model_prediction is not None and str(model_prediction) != str(current_label):
            if model_confidence is not None and current_label_prob is not None:
                facts.append(f"Model assigns {model_confidence:.0%} probability to class {model_prediction}, but only {current_label_prob:.0%} to current label {current_label}")
            else:
                facts.append(f"Model predicts class {model_prediction}, not {current_label}")

        if class_noise_rate is not None and class_noise_rate > 0.1:
            facts.append(f"Class {current_label} has {class_noise_rate:.0%} overall noise rate in dataset")

        if confused_with and len(confused_with) > 0:
            facts.append(f"Class {current_label} is frequently mislabeled as {confused_with[0]}")

        if not facts:
            return f"Label {current_label} flagged as potentially noisy based on ensemble classifier disagreement."

        return " ".join(facts)

    def explain_distribution_anomaly(
        self,
        column_name: str,
        detected_distribution: str,
        anomaly_details: Dict[str, Any],
    ) -> str:
        error = anomaly_details.get("reconstruction_error", 0)
        threshold = anomaly_details.get("threshold", 0)
        features = anomaly_details.get("contributing_features", {})
        row_data = anomaly_details.get("row_data", {})
        feature_means = anomaly_details.get("feature_means", {})

        top_features = list(features.keys())[:3]
        deviations = []
        for f in top_features:
            val = row_data.get(f)
            mean = feature_means.get(f)
            if val is not None and mean is not None:
                deviations.append(f"{f}={val:.2f} (avg={mean:.2f})")
            elif val is not None:
                deviations.append(f"{f}={val}")

        dev_str = ", ".join(deviations) if deviations else "multiple features"
        severity = "severe" if error > threshold * 1.5 else "moderate"
        prompt = f"{severity.title()} anomaly (error {error:.3f}, threshold {threshold:.3f}). Deviations: {dev_str}. Why unusual? 2 sentences, max 25 words."
        return self.generate(prompt, max_tokens=150)

    def generate_dataset_summary(
        self,
        component_results: Dict[str, Dict[str, Any]],
        dataset_info: Dict[str, Any],
    ) -> str:
        issues = []
        for comp, data in component_results.items():
            if "outlier_ratio" in data and data["outlier_ratio"] > 0.1:
                issues.append(f"{data['outlier_ratio']:.0%} outliers")
            if "noise_ratio" in data and data["noise_ratio"] > 0.05:
                issues.append(f"{data['noise_ratio']:.0%} noisy labels")
            if "duplicate_ratio" in data and data["duplicate_ratio"] > 0:
                issues.append(f"{data['duplicate_ratio']:.0%} duplicates")
            if "num_columns_with_missing" in data and data["num_columns_with_missing"] > 0:
                issues.append(f"{data['num_columns_with_missing']} cols missing")

        issues_str = ", ".join(issues[:3]) if issues else "no major issues"
        prompt = f"Dataset has: {issues_str}. Is it ML-ready? What to fix first? 3 sentences max."
        return self.generate(prompt, max_tokens=80)

    def generate_component_summary(
            self,
            component_name: str,
            metrics: Dict[str, Any],
            findings: str,
            total_rows: Optional[int] = None,
    ) -> str:
        row_line = f"Dataset size: {total_rows} rows.\n" if total_rows else ""
        prompt = f"""{row_line}Component: {component_name}
Metrics: {json.dumps(metrics, default=str)}
Findings: {findings}
In 2-3 sentences: Summarize what this component found. Do not make your answers too short. Do NOT make claims about ML readiness based on dataset size alone, but briefly describe how it affects the quality of the data. Focus on describing the actual findings. Use only the numbers provided above.
"""

        return self.generate(
            prompt,
            system_prompt="You are a data quality analyst. Describe only what the metrics show. Do not speculate about ML suitability based on row/column counts. Never invent row counts or percentages not stated in the prompt.",
        )
