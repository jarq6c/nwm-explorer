"""Generate and serve exploratory evaluation dashboard."""
from pathlib import Path
import inspect

import polars as pl
import panel as pn
from panel.template import BootstrapTemplate

from nwm_explorer.logging.logger import get_logger
from nwm_explorer.evaluation.compute import EvaluationRegistry, PREDICTION_RESAMPLING
from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration, Metric
from nwm_explorer.interfaces.filters import FilteringWidgets, CallbackType
from nwm_explorer.data.routelink import get_routelink_readers
from nwm_explorer.plots.site_map import SiteMap
# from nwm_explorer.plots.histogram import Histogram

from nwm_explorer.data.mapping import ModelDomain, Metric

DEFAULT_ZOOM: dict[ModelDomain, int] = {
    ModelDomain.alaska: 5,
    ModelDomain.conus: 3,
    ModelDomain.hawaii: 6,
    ModelDomain.puertorico: 8
}
"""Default map zoom for each domain."""

DEFAULT_CENTER: dict[ModelDomain, dict[str, float]] = {
    ModelDomain.alaska: {"lat": 60.84683, "lon": -149.05659},
    ModelDomain.conus: {"lat": 38.83348, "lon": -93.97612},
    ModelDomain.hawaii: {"lat": 21.24988, "lon": -157.59606},
    ModelDomain.puertorico: {"lat": 18.21807, "lon": -66.32802}
}
"""Default map center for each domain."""

METRIC_PLOTTING_LIMITS: dict[Metric, tuple[float, float]] = {
    Metric.relative_mean_bias: (-1.0, 1.0),
    Metric.pearson_correlation_coefficient: (-1.0, 1.0),
    Metric.nash_sutcliffe_efficiency: (-1.0, 1.0),
    Metric.relative_mean: (0.0, 2.0),
    Metric.relative_standard_deviation: (0.0, 2.0),
    Metric.kling_gupta_efficiency: (-1.0, 1.0)
}
"""Mapping from Metrics to plotting limits (cmin, cmax)."""

class Dashboard:
    """Build a dashboard for exploring National Water Model output."""
    def __init__(self, root: Path, title: str):
        # Get logger
        name = __loader__.name + "." + inspect.currentframe().f_code.co_name
        logger = get_logger(name)

        # Setup template
        self.template = BootstrapTemplate(title=title)

        # Setup registry
        registry_file = root / "evaluation_registry.json"
        if registry_file.exists():
            logger.info(f"Reading {registry_file}")
            with registry_file.open("r") as fo:
                evaluation_registry = EvaluationRegistry.model_validate_json(fo.read())
        else:
            logger.info(f"No registry found at {registry_file}")
            self.template.main.append(pn.pane.Markdown("# Registry not found. Have you run an evaluation?"))
            return
        
        # Scan evaluation data
        self.routelinks = get_routelink_readers(root)
        self.data: dict[str, dict[ModelDomain, dict[ModelConfiguration, pl.LazyFrame]]] = {}
        for label, evaluation_spec in evaluation_registry.evaluations.items():
            self.data[label] = {}
            for domain, files in evaluation_spec.files.items():
                self.data[label][domain] = {}
                for configuration, ifile in files.items():
                    logger.info(f"Scanning {ifile}")
                    self.data[label][domain][configuration] = pl.scan_parquet(ifile)

        # Widgets
        self.filters = FilteringWidgets(evaluation_registry)
        self.map = SiteMap(
            center=DEFAULT_CENTER[self.filters.state.domain],
            zoom=DEFAULT_ZOOM[self.filters.state.domain]
        )
        self.freeze_updates = False
        # self.histogram = Histogram([
        #     Metric.kling_gupta_efficiency,
        #     Metric.pearson_correlation_coefficient,
        #     Metric.relative_mean,
        #     Metric.relative_standard_deviation
        # ])
        # self.histogram_columns = [m+c for m in self.histogram.columns for c in CONFIDENCE_STRINGS.values()]
        # self.histogram_callbacks = [
        #     CallbackType.relayout,
        #     CallbackType.lead_time,
        #     CallbackType.evaluation,
        #     CallbackType.configuration,
        #     CallbackType.domain
        #     ]

        # Callbacks
        def update_interface(event, callback_type: CallbackType) -> None:
            # Limit updates
            if self.freeze_updates:
                return
            self.freeze_updates = True

            if callback_type == CallbackType.relayout:
                if "map.center" in event:
                    self.map.layout["map"]["center"].update(event["map.center"])
                if "map.zoom" in event:
                    self.map.layout["map"].update(dict(zoom=event["map.zoom"]))
                self.freeze_updates = False
                return

            # Current state
            state = self.filters.state

            # Reset view
            if callback_type == CallbackType.double_click:
                # Reset map view
                self.map.layout["map"].update(dict(
                    center=DEFAULT_CENTER[state.domain],
                    zoom=DEFAULT_ZOOM[state.domain]
                ))
                self.map.refresh()
                self.freeze_updates = False
                return

            # Update domain
            if callback_type == CallbackType.domain:
                # Reset map view
                self.map.layout["map"].update(dict(
                    center=DEFAULT_CENTER[state.domain],
                    zoom=DEFAULT_ZOOM[state.domain]
                ))
            
            # Select data
            data = self.data[state.evaluation][state.domain][state.configuration]
            geometry = self.routelinks[state.domain].select(["nwm_feature_id", "latitude", "longitude"])

            # Filter data
            value_column = state.metric + state.confidence
            columns = [value_column, "nwm_feature_id", "usgs_site_code", "start_date", "end_date", "sample_size"]
            if state.configuration in PREDICTION_RESAMPLING:
                columns.append("lead_time_hours_min")
                data = data.filter(pl.col("lead_time_hours_min") == state.lead_time)
            data = data.select(columns).join(geometry, on="nwm_feature_id", how="left").with_columns(
                pl.col("start_date").dt.strftime("%Y-%m-%d"),
                pl.col("end_date").dt.strftime("%Y-%m-%d")
            ).collect()
            
            # Update map
            cmin, cmax = METRIC_PLOTTING_LIMITS[state.metric]
            self.map.update(
                values=data[value_column].to_numpy(),
                latitude=data["latitude"].to_numpy(),
                longitude=data["longitude"].to_numpy(),
                value_label=state.metric_label,
                cmin=cmin,
                cmax=cmax,
                custom_data=data.select(columns[1:]).to_pandas()
            )
            
            # Send changes to frontend
            self.map.refresh()
            self.freeze_updates = False
        self.filters.register_callback(update_interface)
        pn.bind(
            update_interface,
            self.map.pane.param.doubleclick_data,
            watch=True,
            callback_type=CallbackType.double_click
        )
        pn.bind(
            update_interface,
            self.map.pane.param.relayout_data,
            watch=True,
            callback_type=CallbackType.relayout
        )
        # TODO solve double click domain change histogram error

        # def update_histogram(event, callback_type: CallbackType) -> None:
        #     if event is None:
        #         return

        #     if callback_type not in self.histogram_callbacks:
        #         return
            
        #     # Get state
        #     state = self.filters.state

        #     # Select data
        #     geometry = self.routelinks[state.domain].select(["nwm_feature_id", "latitude", "longitude"])
        #     data = self.data[state.evaluation][state.domain][state.configuration].join(
        #         geometry, on="nwm_feature_id", how="left")
        #     if self.map.lat_min is not None:
        #         data = data.filter(
        #                 pl.col("latitude") <= self.map.lat_max,
        #                 pl.col("latitude") >= self.map.lat_min,
        #                 pl.col("longitude") <= self.map.lon_max,
        #                 pl.col("longitude") >= self.map.lon_min
        #             )
        #     if state.configuration in PREDICTION_RESAMPLING:
        #         data = data.filter(pl.col("lead_time_hours_min") == state.lead_time)
        #     data = data.select(self.histogram_columns).collect()
            
        #     # Update and refresh
        #     self.histogram.update(data)
        #     self.histogram.refresh()
        # self.filters.register_callback(update_histogram)
        # pn.bind(update_histogram, self.map.relayout_data, watch=True,
        #     callback_type=CallbackType.relayout)

        # Layout
        self.template.main.append(
            pn.Row(
                self.filters.servable(),
                self.map.servable(),
                # self.histogram.servable()
        ))
    
    def servable(self) -> BootstrapTemplate:
        return self.template

def generate_dashboard(
        root: Path,
        title: str
        ) -> BootstrapTemplate:
    return Dashboard(root, title).servable()

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
