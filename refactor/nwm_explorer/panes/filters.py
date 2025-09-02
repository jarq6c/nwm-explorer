"""
Filters used by evaluation dashboard to specify data sources.
"""
import panel as pn
from panel.viewable import Viewer

from nwm_explorer.application.api import EvaluationRegistry

class Filters(Viewer):
    """
    Houses main filtering widgets and tracks their state.

    Attributes
    ----------
    evaluation: str
        Currently selected evaluation.
    """
    def __init__(self, registry: EvaluationRegistry, **params) -> None:
        super().__init__(**params)

        # Build widgets
        self.evaluation_selector = pn.widgets.Select(
            name="Evaluation",
            options=list(registry.evaluations.keys())
        )
        self.domain_selector = pn.widgets.Select(
            name="Model domain",
            options=list(registry.evaluations[self.evaluation_selector.value].keys())
        )
        self.configuration_selector = pn.widgets.Select(
            name="Model configuration",
            options=list(registry.evaluations[self.evaluation_selector.value][self.domain_selector.value].keys())
        )

        # Callbacks
        def update_domain_options(evaluation: str) -> None:
            self.domain_selector.options = list(registry.evaluations[evaluation].keys())
            self.configuration_selector.options = list(registry.evaluations[evaluation][self.domain_selector.value].keys())
        pn.bind(update_domain_options, self.evaluation_selector.param.value, watch=True)

    def __panel__(self) -> pn.Card:
        return pn.Card(
            pn.Column(
                self.evaluation_selector,
                self.domain_selector,
                self.configuration_selector
            ),
            title="Filters",
            collapsible=False
        )
