"""Generate and serve exploratory applications."""
from pathlib import Path
from typing import Callable
import pandas as pd
import panel as pn
from panel.template import BootstrapTemplate

from nwm_explorer.mappings import (EVALUATIONS, DOMAIN_STRINGS,
    DOMAIN_CONFIGURATION_MAPPING, Domain, Configuration, LEAD_TIME_VALUES,
    CONFIDENCE_STRINGS, Confidence, METRIC_STRINGS, Metric)
from nwm_explorer.readers import MetricReader, DashboardState

pn.extension("plotly")

class FilteringWidgets:
    def __init__(self):
        # Filtering options
        self.callbacks: list[Callable] = []
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
                "100% AEP-USGS (All data)"
            ]
        )
        self.metric_filter = pn.widgets.Select(
            name="Evaluation Metric",
            options=list(METRIC_STRINGS.keys())
        )
        self.confidence_filter = pn.widgets.Select(
            name="Confidence Estimate (95%)",
            options=list(CONFIDENCE_STRINGS.keys())
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
    def current_start_date(self) -> pd.Timestamp:
        return EVALUATIONS[self.evaluation_filter.value][0]
        
    @property
    def current_end_date(self) -> pd.Timestamp:
        return EVALUATIONS[self.evaluation_filter.value][1]

    @property
    def current_domain(self) -> Domain:
        return DOMAIN_STRINGS[self.domain_filter.value]

    @property
    def current_configuration(self) -> Configuration:
        return DOMAIN_CONFIGURATION_MAPPING[self.current_domain][self.configuration_filter.value]
    
    @property
    def current_lead_time(self) -> int:
        return self.lead_time_filter[0].value

    @property
    def current_confidence(self) -> Confidence:
        return CONFIDENCE_STRINGS[self.confidence_filter.value]

    @property
    def current_metric(self) -> Metric:
        return METRIC_STRINGS[self.metric_filter.value]

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
        
        v = self.lead_time_filter[0].value
        self.lead_time_filter.objects = [
            pn.widgets.DiscretePlayer(
                name="Minimum lead time (hours)",
                options=options,
                show_loop_controls=False,
                visible_buttons=["previous", "next"],
                width=300,
                value=v if v in options else 0
                )
        ]
        for func in self.callbacks:
            pn.bind(func, self.lead_time_filter[0], watch=True)

    @property
    def state(self) -> DashboardState:
        """Current widget states."""
        return DashboardState(
            start_date=self.current_start_date,
            end_date=self.current_end_date,
            domain=self.current_domain,
            configuration=self.current_configuration,
            threshold=self.threshold_filter.value,
            metric=self.current_metric,
            confidence=self.current_confidence,
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
    
    def register_callback(self, func: Callable) -> None:
        """Register callback function."""
        pn.bind(func, self.evaluation_filter, watch=True)
        pn.bind(func, self.domain_filter, watch=True)
        pn.bind(func, self.configuration_filter, watch=True)
        pn.bind(func, self.threshold_filter, watch=True)
        pn.bind(func, self.metric_filter, watch=True)
        pn.bind(func, self.confidence_filter, watch=True)
        pn.bind(func, self.lead_time_filter[0], watch=True)
        self.callbacks.append(func)

class Dashboard:
    """Build a dashboard for exploring National Water Model output."""
    def __init__(self, root: Path, title: str):
        # Data reader
        self.reader = MetricReader(root)
    
        # Get widgets
        self.filter_widgets = FilteringWidgets()

        # Layout filtering options
        self.filter_card = pn.Card(
            self.filter_widgets.layout,
            title="Filters",
            collapsible=False
            )
        
        # Setup map
        self.site_map = pn.pane.Plotly(
            self.reader.get_plotly_patch(self.state))
        self.map_card = pn.Card(
            self.site_map,
            collapsible=False,
            hide_header=True
            )
        
        def update_map(event):
            if event is None:
                return
            self.site_map.object = self.reader.get_plotly_patch(self.state)
        self.filter_widgets.register_callback(update_map)

        # Layout cards
        layout = pn.Row(self.filter_card, self.map_card)
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
