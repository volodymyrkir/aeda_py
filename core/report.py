import datetime

from report_components.base_component import ReportComponent, AnalysisContext
from typing import Dict, Any


class Report:
    """
    Report orchestrator that runs analysis components and collects results.

    After running, component results are stored in the context for use by
    summary generation components.
    """

    def __init__(self):
        self.components = []

    def add_component(self, component: ReportComponent):
        self.components.append(component)

    def run(self):
        """Run all components and store their results in context."""
        for component in self.components:
            print(f"Running component: {component.__class__.__name__}, {datetime.datetime.now().strftime('%H:%M:%S')}")
            component.analyze()
            summary = component.summarize()
            component.context.store_component_result(
                component.__class__.__name__,
                summary
            )

    def get_all_summaries(self) -> Dict[str, Any]:
        """Get summaries from all components."""
        summaries = {}
        for component in self.components:
            try:
                summaries[component.__class__.__name__] = component.summarize()
            except Exception:
                pass
        return summaries
