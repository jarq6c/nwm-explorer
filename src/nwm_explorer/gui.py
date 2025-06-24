"""Generate and serve exploratory applications."""
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import polars as pl
import panel as pn
from panel.template import BootstrapTemplate

from nwm_explorer.mappings import (EVALUATIONS, DOMAIN_STRINGS,
    DOMAIN_CONFIGURATION_MAPPING, Domain, Configuration, LEAD_TIME_VALUES,
    CONFIDENCE_STRINGS, Confidence, METRIC_STRINGS, Metric, DEFAULT_ZOOM)
from nwm_explorer.readers import MetricReader, DashboardState
from nwm_explorer.plotters import SiteMapPlotter, HistogramPlotter

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
            evaluation=self.evaluation_filter.value,
            start_date=self.current_start_date,
            end_date=self.current_end_date,
            domain=self.current_domain,
            configuration=self.current_configuration,
            threshold=self.threshold_filter.value,
            metric=self.current_metric,
            metric_label=self.metric_filter.value,
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
        
        # Status
        self.status_feed = pn.Feed(
            pn.pane.Alert("Initialized", alert_type="secondary"),
            width=300,
            height=200,
            view_latest=False
            )
        status_card = pn.Card(
            self.status_feed,
            title="Status",
            collapsible=False
            )
        
        # Setup map
        self.last_domain = self.state.domain
        self.site_plotter = SiteMapPlotter()
        self.site_map = pn.pane.Plotly(
            self.site_plotter.figure)
        self.map_card = pn.Card(
            self.site_map,
            collapsible=False,
            hide_header=True
            )
        
        # Histogram
        self.hist_plotters = [
            HistogramPlotter(),
            HistogramPlotter(),
            HistogramPlotter(),
            HistogramPlotter()
        ]
        self.histograms = [pn.pane.Plotly(hp.figure, config={"displayModeBar": False}) for hp in self.hist_plotters]
        self.histogram_cards = [pn.Card(
            hp,
            collapsible=False,
            hide_header=True
        ) for hp in self.histograms]
        self.hist_box = pn.GridBox(*self.histogram_cards, ncols=2)
        
        # Update data
        self.data = self.reader.query(self.state)

        def update_map(event):
            if event is None:
                return
            current_state = self.state
            self.data = self.reader.query(current_state)
            if self.data is None:
                self.status_feed.insert(0,
                    pn.pane.Alert("No data found", alert_type="warning"))
                return
            relayout_data = self.site_map.relayout_data
            
            if current_state.domain == self.last_domain and relayout_data is not None:
                self.site_plotter.update_colors(
                    values=self.data["value"].to_numpy(),
                    label=self.state.metric_label,
                    relayout_data=relayout_data
                )
            else:
                self.site_plotter.update_points(
                    values=self.data["value"].to_numpy(),
                    latitude=self.data["latitude"].to_numpy(),
                    longitude=self.data["longitude"].to_numpy(),
                    metric_label=current_state.metric_label,
                    zoom=DEFAULT_ZOOM[current_state.domain],
                    custom_data=self.data.select(["usgs_site_code", "nwm_feature_id"])
                )
                self.last_domain = current_state.domain
            self.site_map.object = self.site_plotter.figure
        self.filter_widgets.register_callback(update_map)

        def update_histograms(event, event_type: str = "normal"):
            # TODO Start splitting out this functionality into smaller dashboards
            # NOTE The app is starting to look like a dashboard of dashboards
            if event is None:
                return
            current_state = self.state
            kge = self.reader.query(current_state, column_override="kge")
            kge_lower = self.reader.query(current_state, column_override="kge_lower")
            kge_upper = self.reader.query(current_state, column_override="kge_upper")
            if event_type == "relayout":
                if "map._derived" in event:
                    bbox = event["map._derived"]["coordinates"]
                    lon_min = bbox[0][0]
                    lon_max = bbox[1][0]
                    lat_min = bbox[2][1]
                    lat_max = bbox[0][1]
                    expression = (
                        pl.col("latitude") >= lat_min,
                        pl.col("latitude") <= lat_max,
                        pl.col("longitude") >= lon_min,
                        pl.col("longitude") <= lon_max
                        )
                    kge = kge.filter(*expression)
                    kge_lower = kge_lower.filter(*expression)
                    kge_upper = kge_upper.filter(*expression)
            self.hist_plotters[0].update_bars(
                values=kge["value"].to_numpy(),
                values_lower=kge_lower["value"].to_numpy(),
                values_upper=kge_upper["value"].to_numpy(),
                vmin=-1.0, vmax=1.0, bin_width=0.2, xtitle="KGE"
            )
            self.histograms[0].object = self.hist_plotters[0].figure
        self.filter_widgets.register_callback(update_histograms)
        pn.bind(update_histograms, self.site_map.param.relayout_data,
            watch=True, event_type="relayout")

        # Layout cards
        layout = pn.Row(pn.Column(
            self.filter_card,
            status_card
            ),
        self.map_card,
        self.hist_box
        )
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
