from report_components.base_component import ReportComponent

class Report:
    def __init__(self):
        self.components = []

    def add_component(self, component: ReportComponent):
        self.components.append(component)

    def run(self):
        for component in self.components:
            component.analyze()
