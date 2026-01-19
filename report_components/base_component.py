from abc import ABC, abstractmethod

class AnalysisContext:
    def __init__(self, dataset):
        self.dataset = dataset
        self.shared_artifacts = {}


class ReportComponent(ABC):
    def __init__(self, context: AnalysisContext):
        self.context = context
        self.result = None

    @abstractmethod
    def analyze(self):
        pass

    @abstractmethod
    def summarize(self) -> dict:
        pass

    @abstractmethod
    def justify(self) -> str:
        pass
