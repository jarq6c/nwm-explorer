"""Generate and serve exploratory applications."""
from pathlib import Path
from dataclasses import dataclass
import panel as pn
from panel.template import BootstrapTemplate

from nwm_explorer.mappings import (EVALUATIONS, DOMAIN_STRINGS,
    DOMAIN_CONFIGURATION_MAPPING, Domain, Configuration, LEAD_TIME_VALUES)

@dataclass
class DashboardState:
    """Dashboard state variables."""
    evaluation: str
    domain: Domain
    configuration: str
    threshold: str
    metric: str
    confidence: str
    lead_time: int

class FilteringWidgets:
    def __init__(self):
        # Filtering options
        self.evaluation_filter = pn.widgets.Select(
            name="Evaluation",
            options=list(EVALUATIONS.keys())
        )
        self.domain_filter = pn.widgets.Select(
            name="Model Domain",
            options=list(DOMAIN_STRINGS.keys())
        )
        self.configuration_filter = pn.widgets.Select(
            name="Model Configuration",
            options=list(
            DOMAIN_CONFIGURATION_MAPPING[self.current_domain].keys()
            ))
        self.threshold_filter = pn.widgets.Select(
            name="Streamflow Threshold (≥)",
            options=[
                "0th Percentile (All data)",
                "85th Percentile",
                "95th Percentile",
                "99th Percentile"
            ]
        )
        self.metric_filter = pn.widgets.Select(
            name="Evaluation Metric",
            options=[
                "Mean relative bias",
                "Nash-Sutcliffe efficiency",
                "Pearson correlation coefficient",
                "Kling-Gupta efficiency"
            ]
        )
        self.confidence_filter = pn.widgets.Select(
            name="Confidence Estimate (95%)",
            options=[
                "Point",
                "Lower",
                "Upper"
            ]
        )
        if self.current_configuration in LEAD_TIME_VALUES:
            options = LEAD_TIME_VALUES[self.current_configuration]
        else:
            options = [0]
        self.lead_time_filter = pn.Row(pn.widgets.DiscretePlayer(
            name="Minimum lead time (hours)",
            options=options,
            show_loop_controls=False,
            visible_buttons=["previous", "next"],
            width=300
            ))

        def handle_domain_change(domain):
            if domain is None:
                return
            self.update_configurations()
        pn.bind(handle_domain_change, self.domain_filter, watch=True)

        def handle_configuration_change(domain):
            if domain is None:
                return
            self.update_lead_times()
        pn.bind(handle_configuration_change, self.configuration_filter,
            watch=True)

    @property
    def current_domain(self) -> Domain:
        return DOMAIN_STRINGS[self.domain_filter.value]

    @property
    def current_configuration(self) -> Configuration:
        return DOMAIN_CONFIGURATION_MAPPING[self.current_domain][self.configuration_filter.value]
    
    @property
    def current_lead_time(self) -> int:
        return self.lead_time_filter[0].value

    def update_configurations(self) -> None:
        """Set configuration options"""
        self.configuration_filter.options = list(
            DOMAIN_CONFIGURATION_MAPPING[self.current_domain].keys())

    def update_lead_times(self) -> None:
        """Set lead time options"""
        c = self.current_configuration
        if c in LEAD_TIME_VALUES:
            options = LEAD_TIME_VALUES[c]
        else:
            options = [0]
        
        self.lead_time_filter.objects = [
            pn.widgets.DiscretePlayer(
                name="Minimum lead time (hours)",
                options=options,
                show_loop_controls=False,
                visible_buttons=["previous", "next"],
                width=300
                )
        ]

    @property
    def state(self) -> DashboardState:
        """Current widget states."""
        return DashboardState(
            evaluation=self.evaluation_filter.value,
            domain=self.current_domain,
            configuration=self.current_configuration,
            threshold=self.threshold_filter.value,
            metric=self.metric_filter.value,
            confidence=self.confidence_filter.value,
            lead_time=self.current_lead_time
        )
    
    @property
    def layout(self) -> pn.Column:
        return pn.Column(
            self.evaluation_filter,
            self.domain_filter,
            self.configuration_filter,
            self.threshold_filter,
            self.metric_filter,
            self.confidence_filter,
            self.lead_time_filter
        )

class Dashboard:
    """Build a dashboard for exploring National Water Model output."""
    def __init__(self, root: Path, title: str):
        # Get widgets
        self.filter_widgets = FilteringWidgets()

        # Layout filtering options
        self.filter_card = pn.Card(
            self.filter_widgets.layout,
            title="Filters",
            collapsible=False
            )
        
        def callback(event):
            print(self.state)
        pn.bind(callback, self.filter_widgets.evaluation_filter, watch=True)

        # Layout cards
        layout = pn.Row(self.filter_card)
        self.template = BootstrapTemplate(title=title)
        self.template.main.append(layout)

    @property
    def state(self) -> DashboardState:
        """Current dashboard state."""
        return self.filter_widgets.state

def generate_dashboard(
        root: Path,
        title: str
        ) -> BootstrapTemplate:
    return Dashboard(root, title).template

def generate_dashboard_closure(
        root: Path,
        title: str
        ) -> BootstrapTemplate:
    def closure():
        return generate_dashboard(root, title)
    return closure

def serve_dashboard(
        root: Path,
        title: str
        ) -> None:
    # Slugify title
    slug = title.lower().replace(" ", "-")

    # Serve
    endpoints = {
        slug: generate_dashboard_closure(root, title)
    }
    pn.serve(endpoints)
