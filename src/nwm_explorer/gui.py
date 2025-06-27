"""Generate and serve exploratory applications."""
from pathlib import Path
from typing import Callable
import pandas as pd
import polars as pl
import panel as pn
from panel.template import BootstrapTemplate

from nwm_explorer.mappings import (EVALUATIONS, DOMAIN_STRINGS,
    DOMAIN_CONFIGURATION_MAPPING, Domain, Configuration, LEAD_TIME_VALUES,
    CONFIDENCE_STRINGS, Confidence, METRIC_STRINGS, Metric, DEFAULT_ZOOM,
    METRIC_SHORTHAND, CONFIDENCE_SHORTHAND)
from nwm_explorer.readers import MetricReader, DashboardState, NWMReader, USGSReader
from nwm_explorer.plotters import SiteMapPlotter
from nwm_explorer.histogram import HistogramGrid
from nwm_explorer.hydrographer import HydrographCard

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
        # Data readers
        self.metrics_reader = MetricReader(root)
        self.nwm_reader = NWMReader(root)
        self.usgs_reader = USGSReader(root)
    
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
            width=320,
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
            self.site_plotter.figure, config={"displaylogo": False})
        self.map_card = pn.Card(
            self.site_map,
            collapsible=False,
            hide_header=True
            )
        
        # Update data
        self.data = self.metrics_reader.query(self.state)

        def update_map(event):
            if event is None:
                return
            current_state = self.state
            self.data = self.metrics_reader.query(current_state)
            if self.data is None:
                self.status_feed.insert(0,
                    pn.pane.Alert("No data found", alert_type="warning"))
                return
            relayout_data = self.site_map.relayout_data
            col = METRIC_SHORTHAND[current_state.metric] + CONFIDENCE_SHORTHAND[current_state.confidence]
            
            if current_state.domain == self.last_domain and relayout_data is not None:
                self.site_plotter.update_colors(
                    values=self.data[col].to_numpy(),
                    label=self.state.metric_label,
                    relayout_data=relayout_data
                )
            else:
                self.site_plotter.update_points(
                    values=self.data[col].to_numpy(),
                    latitude=self.data["latitude"].to_numpy(),
                    longitude=self.data["longitude"].to_numpy(),
                    metric_label=current_state.metric_label,
                    zoom=DEFAULT_ZOOM[current_state.domain],
                    custom_data=self.data.select(["usgs_site_code", "nwm_feature_id"])
                )
                self.last_domain = current_state.domain
            self.site_map.object = self.site_plotter.figure
        self.filter_widgets.register_callback(update_map)

        # Setup histogram
        self.hcolumns = ["kge", "pearson", "rel_mean", "rel_var"]
        datasets = []
        for c in self.hcolumns:
            d = self.metrics_reader.query(self.state, [c, f"{c}_lower", f"{c}_upper"])
            datasets.append((
                d[c].to_numpy(),
                d[f"{c}_lower"].to_numpy(),
                d[f"{c}_upper"].to_numpy()
            ))
        labels = ["KGE", "Pearson", "Relative Mean", "Relative Variance"]
        specs = [
            (-1.0, 1.0, 0.2),
            (-1.0, 1.0, 0.2),
            (0.0, 2.0, 0.2),
            (0.0, 2.0, 0.2),
        ]
        self.hgrid = HistogramGrid(datasets, specs, labels, 2)

        def update_histograms(event, event_type: str = "normal"):
            if not event:
                return
            if event_type == "normal":
                if event in METRIC_STRINGS:
                    return
                if event in CONFIDENCE_STRINGS:
                    return
            bbox = None
            relayout_data = self.site_map.relayout_data
            if "map._derived" in relayout_data:
                bbox = {
                    "lat_max": relayout_data["map._derived"]["coordinates"][0][1],
                    "lat_min": relayout_data["map._derived"]["coordinates"][2][1],
                    "lon_max": relayout_data["map._derived"]["coordinates"][1][0],
                    "lon_min": relayout_data["map._derived"]["coordinates"][0][0],
                }
            datasets = []
            for c in self.hcolumns:
                d = self.metrics_reader.query(self.state, [c, f"{c}_lower", f"{c}_upper"])
                if bbox is not None:
                    d = d.filter(
                        pl.col("latitude") <= bbox["lat_max"],
                        pl.col("latitude") >= bbox["lat_min"],
                        pl.col("longitude") <= bbox["lon_max"],
                        pl.col("longitude") >= bbox["lon_min"],
                    )
                datasets.append((
                    d[c].to_numpy(),
                    d[f"{c}_lower"].to_numpy(),
                    d[f"{c}_upper"].to_numpy()
                ))
            self.hgrid.update_data(datasets)
        self.filter_widgets.register_callback(update_histograms)
        pn.bind(update_histograms, self.site_map.param.relayout_data, watch=True,
            event_type="relayout")
        
        # Setup hydrograph
        self.usgs_site_code = None
        self.nwm_feature_id = None
        self.hydrograph: HydrographCard = None
        self.hydrograph_card = pn.pane.Placeholder(pn.Card(
            pn.pane.Markdown(
                "# Click a site on the map to view its hydrograph.",
                align="center"
                ),
            collapsible=False,
            hide_header=True,
            height=265,
            width=1045
        ))

        def update_hydrograph(event, event_type: str = "normal"):
            if not event:
                return

            if event_type == "click":
                self.usgs_site_code = event["points"][0]["customdata"][0]
                self.nwm_feature_id = event["points"][0]["customdata"][1]
            elif event in DOMAIN_STRINGS:
                self.usgs_site_code = None
                self.nwm_feature_id = None
                return
            elif event not in DOMAIN_CONFIGURATION_MAPPING[self.state.domain]:
                return

            if self.usgs_site_code is None:
                return
            nwm_data = self.nwm_reader.query(self.state, self.nwm_feature_id)

            if nwm_data is None or nwm_data.is_empty():
                return
            usgs_data = self.usgs_reader.query(
                self.state.domain,
                self.usgs_site_code,
                nwm_data["value_time"].min(),
                nwm_data["value_time"].max()
            )
            xdata = [usgs_data["value_time"]]
            ydata = [usgs_data["value"]]
            names = [f"USGS-{self.usgs_site_code}"]
            
            if self.state.configuration in LEAD_TIME_VALUES:
                for (rt,), forecast in nwm_data.partition_by("reference_time", as_dict=True, include_key=False).items():
                    xdata.append(forecast["value_time"])
                    ydata.append(forecast["value"])
                    names.append(rt.strftime("%Y-%m-%d %HZ"))
            else:
                xdata.append(nwm_data["value_time"])
                ydata.append(nwm_data["value"])
                names.append("Analysis")

            if self.hydrograph is None:
                self.hydrograph = HydrographCard(
                    x=xdata,
                    y=ydata,
                    names=names,
                    y_title="Streamflow (cfs)"
                )
                self.hydrograph_card.object = self.hydrograph.servable()
            else:
                self.hydrograph.update_data(
                    x=xdata,
                    y=ydata,
                    names=names
                )
        self.filter_widgets.register_callback(update_hydrograph)
        pn.bind(update_hydrograph, self.site_map.param.click_data, watch=True,
            event_type="click")

        # Layout cards
        controls = pn.Column(self.filter_card, status_card)
        over_view = pn.Row(self.map_card, self.hgrid.servable())
        site_view = pn.Row(self.hydrograph_card)
        layout = pn.Row(
            controls,
            pn.Column(over_view, site_view)
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
