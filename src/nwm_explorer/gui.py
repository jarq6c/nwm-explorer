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
    METRIC_SHORTHAND, CONFIDENCE_SHORTHAND, CallbackType, METRIC_PLOTTING_LIMITS)
from nwm_explorer.readers import MetricReader, DashboardState, NWMReader, USGSReader
from nwm_explorer.site_map import SiteMapCard
from nwm_explorer.histogram import HistogramGrid
from nwm_explorer.hydrographer import HydrographCard
from nwm_explorer.barplot import BarPlot

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
            pn.bind(func, self.lead_time_filter[0], callback_type=CallbackType.lead_time, watch=True)

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
        pn.bind(func, self.evaluation_filter, callback_type=CallbackType.evaluation, watch=True)
        pn.bind(func, self.domain_filter, callback_type=CallbackType.domain, watch=True)
        pn.bind(func, self.configuration_filter, callback_type=CallbackType.configuration, watch=True)
        pn.bind(func, self.threshold_filter, callback_type=CallbackType.threshold, watch=True)
        pn.bind(func, self.metric_filter, callback_type=CallbackType.metric, watch=True)
        pn.bind(func, self.confidence_filter, callback_type=CallbackType.confidence, watch=True)
        pn.bind(func, self.lead_time_filter[0], callback_type=CallbackType.lead_time, watch=True)
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
        data = self.metrics_reader.query(self.state)
        column = METRIC_SHORTHAND[self.state.metric] + CONFIDENCE_SHORTHAND[self.state.confidence]
        self.site_map = SiteMapCard(
            latitude=data["latitude"].to_numpy(),
            longitude=data["longitude"].to_numpy(),
            custom_data=data.select(["usgs_site_code", "nwm_feature_id"]),
            values=data[column].to_numpy(),
            value_label=self.state.metric_label,
            value_limits=METRIC_PLOTTING_LIMITS[self.state.metric],
            custom_labels=["USGS Site Code", "NWM Feature ID"],
            default_zoom=DEFAULT_ZOOM[self.state.domain]
        )

        def update_map(event, callback_type: CallbackType) -> None:
            if not event:
                return
            
            data = self.metrics_reader.query(self.state)
            column = METRIC_SHORTHAND[self.state.metric] + CONFIDENCE_SHORTHAND[self.state.confidence]

            if callback_type in [CallbackType.domain, CallbackType.evaluation]:
                self.site_map.update_points(
                    latitude=data["latitude"].to_numpy(),
                    longitude=data["longitude"].to_numpy(),
                    custom_data=data.select(["usgs_site_code", "nwm_feature_id"]),
                    values=data[column].to_numpy(),
                    value_label=self.state.metric_label,
                    value_limits=METRIC_PLOTTING_LIMITS[self.state.metric],
                    custom_labels=["USGS Site Code", "NWM Feature ID"],
                    default_zoom=DEFAULT_ZOOM[self.state.domain]
                )
            else:
                self.site_map.update_values(
                    values=data[column].to_numpy(),
                    value_label=self.state.metric_label,
                    value_limits=METRIC_PLOTTING_LIMITS[self.state.metric]
                )
            self.site_map.refresh()
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

        def update_histograms(event, callback_type: CallbackType):
            if not event:
                return
            if callback_type == CallbackType.metric:
                return
            if callback_type == CallbackType.confidence:
                return

            datasets = []
            for c in self.hcolumns:
                d = self.metrics_reader.query(self.state, [c, f"{c}_lower", f"{c}_upper"])
                d = d.filter(
                    pl.col("latitude") <= self.site_map.lat_max,
                    pl.col("latitude") >= self.site_map.lat_min,
                    pl.col("longitude") <= self.site_map.lon_max,
                    pl.col("longitude") >= self.site_map.lon_min,
                )
                datasets.append((
                    d[c].to_numpy(),
                    d[f"{c}_lower"].to_numpy(),
                    d[f"{c}_upper"].to_numpy()
                ))
            self.hgrid.update_data(datasets)
            self.hgrid.refresh()
        self.filter_widgets.register_callback(update_histograms)
        pn.bind(update_histograms, self.site_map.relayout_data, watch=True,
            callback_type=CallbackType.relayout)
        
        # Setup hydrograph
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

        def update_hydrograph(event, callback_type: CallbackType):
            if not event:
                return
            
            if callback_type in [CallbackType.domain, CallbackType.configuration]:
                return
            
            usgs_site_code = self.site_map.selection.get("usgs_site_code", None)
            nwm_feature_id = self.site_map.selection.get("nwm_feature_id", None)

            if usgs_site_code is None:
                return
            
            nwm_data = self.nwm_reader.query(self.state, nwm_feature_id)

            if nwm_data is None or nwm_data.is_empty():
                return
            usgs_data = self.usgs_reader.query(
                self.state.domain,
                usgs_site_code,
                nwm_data["value_time"].min(),
                nwm_data["value_time"].max()
            )
            xdata = [usgs_data["value_time"]]
            ydata = [usgs_data["value"]]
            names = [f"USGS-{usgs_site_code}"]
            
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
                self.hydrograph.refresh()
        self.filter_widgets.register_callback(update_hydrograph)
        pn.bind(update_hydrograph, self.site_map.click_data, watch=True,
            callback_type=CallbackType.click)
        
        # Setup bar plot
        self.barplot: BarPlot = None
        self.barplot_card = pn.pane.Placeholder(pn.Card(
            pn.pane.Markdown(
                "## Click a site on the map to view its statistics.",
                align="center"
                ),
            collapsible=False,
            hide_header=True,
            height=265,
            width=475
        ))

        def update_barplot(event, callback_type: CallbackType):
            if not event:
                return

            if callback_type not in [CallbackType.click, CallbackType.configuration, CallbackType.metric]:
                return
            
            usgs_site_code = self.site_map.selection.get("usgs_site_code", None)
            nwm_feature_id = self.site_map.selection.get("nwm_feature_id", None)

            if usgs_site_code is None:
                return
            data = self.metrics_reader.query_site(self.state, nwm_feature_id)

            if data is None or data.is_empty():
                return
            if self.state.configuration in LEAD_TIME_VALUES:
                xdata = data["lead_time_hours_min"].to_numpy()
            else:
                xdata = [0]
            c = METRIC_SHORTHAND[self.state.metric]
            ydata = data[c].to_numpy()
            ydata_lower = data[f"{c}_lower"].to_numpy()
            ydata_upper = data[f"{c}_upper"].to_numpy()
            xlabel = "Minimum Lead Time (h)"
            ylabel = self.state.metric_label

            if self.barplot is None:
                self.barplot = BarPlot(
                    xdata=xdata,
                    ydata=ydata,
                    ydata_lower=ydata_lower,
                    ydata_upper=ydata_upper,
                    xlabel=xlabel,
                    ylabel=ylabel
                )
                self.barplot_card.object = self.barplot.servable()
            else:
                self.barplot.update_data(
                    xdata=xdata,
                    ydata=ydata,
                    ydata_lower=ydata_lower,
                    ydata_upper=ydata_upper,
                    xlabel=xlabel,
                    ylabel=ylabel
                )
                self.barplot.refresh()
        self.filter_widgets.register_callback(update_barplot)
        pn.bind(update_barplot, self.site_map.click_data, watch=True,
            callback_type=CallbackType.click)

        # Layout cards
        controls = pn.Column(self.filter_card, status_card)
        over_view = pn.Row(self.site_map.servable(), self.hgrid.servable())
        site_view = pn.Row(self.hydrograph_card, self.barplot_card)
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
