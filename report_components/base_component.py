from abc import ABC, abstractmethod

_llm_service = None


def _get_llm_service():
    global _llm_service
    if _llm_service is None:
        try:
            from utils.llm_service import LLMService
            _llm_service = LLMService.get_instance()
        except Exception:
            _llm_service = None
    return _llm_service


class AnalysisContext:
    def __init__(self, dataset, llm_service=None):
        self.dataset = dataset
        self.shared_artifacts = {}
        self.component_results = {}
        self._llm_service = llm_service

    def store_component_result(self, component_name: str, summary: dict):
        self.component_results[component_name] = summary

    @property
    def llm_service(self):
        if self._llm_service is None:
            self._llm_service = _get_llm_service()
        return self._llm_service

    @llm_service.setter
    def llm_service(self, service):
        self._llm_service = service


class ReportComponent(ABC):
    def __init__(self, context: AnalysisContext, use_llm_explanations: bool = True):
        self.context = context
        self.result = None
        self.use_llm_explanations = use_llm_explanations
        self._llm = None

    @property
    def llm(self):
        if not self.use_llm_explanations:
            return None
        if self._llm is None:
            self._llm = self.context.llm_service
        return self._llm

    @abstractmethod
    def analyze(self):
        pass

    @abstractmethod
    def summarize(self) -> dict:
        pass

    @abstractmethod
    def justify(self) -> str:
        pass

    def get_full_summary(self) -> str:
        return ""
