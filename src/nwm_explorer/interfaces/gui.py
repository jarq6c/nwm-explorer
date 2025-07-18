"""Generate and serve exploratory evaluation dashboard."""
from pathlib import Path
import inspect

import polars as pl
import panel as pn
import numpy as np
import numpy.typing as npt
from panel.template import BootstrapTemplate

from nwm_explorer.logging.logger import get_logger
from nwm_explorer.evaluation.compute import EvaluationRegistry, PREDICTION_RESAMPLING
from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration
from nwm_explorer.interfaces.filters import FilteringWidgets, CallbackType
from nwm_explorer.data.routelink import get_routelink_readers

import plotly.graph_objects as go
import colorcet as cc

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

METRIC_PLOTTING_LIMITS: dict[str, tuple[float, float]] = {
    "relative_mean_bias": (-1.0, 1.0),
    "pearson_correlation_coefficient": (-1.0, 1.0),
    "nash_sutcliffe_efficiency": (-1.0, 1.0),
    "relative_mean": (0.0, 2.0),
    "relative_standard_deviation": (0.0, 2.0),
    "kling_gupta_efficiency": (-1.0, 1.0)
}
"""Mapping from Metrics to plotting limist (cmin, cmax)."""

class SiteMap:
    def __init__(self):
        # Viewport
        self.domain = ModelDomain.alaska

        # Map data
        self.data = [go.Scattermap(
            marker=dict(
                colorbar=dict(
                    title=dict(
                        side="right"
                        )
                    ),
                size=15,
                colorscale=cc.gouldian
            ),
            showlegend=False,
            name="",
            mode="markers"
        )]

        # Map layout
        self.layout = go.Layout(
            showlegend=False,
            height=540,
            width=850,
            margin=dict(l=0, r=0, t=0, b=0),
            map=dict(
                style="satellite-streets",
                center=DEFAULT_CENTER[self.domain],
                zoom=DEFAULT_ZOOM[self.domain]
            ),
            clickmode="event",
            modebar=dict(
                remove=["lasso", "select", "resetview"],
                orientation="v"
            ),
            dragmode="zoom"
        )

        # Map figure
        self.figure = dict(
            data=self.data,
            layout=self.layout
        )

        # Servable
        self.pane = pn.pane.Plotly(self.figure)

        # Update layout
        def apply_relayout_data(data) -> None:
            if data is None:
                return
            if "map.center" in data:
                self.layout["map"]["center"].update(data["map.center"])
            if "map.zoom" in data:
                self.layout["map"].update(dict(zoom=data["map.zoom"]))
        pn.bind(apply_relayout_data, self.pane.param.relayout_data, watch=True)

        # Reset view
        def reset_view(event) -> None:
            self.layout["map"]["center"].update(DEFAULT_CENTER[self.domain])
            self.layout["map"].update(dict(zoom=DEFAULT_ZOOM[self.domain]))
            self.refresh()
        pn.bind(reset_view, self.pane.param.doubleclick_data, watch=True)
    
    def update(
        self,
        values: npt.ArrayLike,
        latitude: npt.ArrayLike,
        longitude: npt.ArrayLike,
        value_name: str,
        cmin: float,
        cmax: float,
        domain: ModelDomain
        ) -> None:
        # Colors
        self.data[0]["marker"].update(dict(color=values, cmin=cmin, cmax=cmax))

        # Coordinates
        self.data[0].update(dict(lat=latitude, lon=longitude))

        # Title
        self.data[0]["marker"]["colorbar"]["title"].update(dict(text=value_name))

        # Domain change
        if domain != self.domain:
            self.layout["map"]["center"].update(DEFAULT_CENTER[domain])
            self.layout["map"].update(dict(zoom=DEFAULT_ZOOM[domain]))
            self.domain = domain
    
    def refresh(self) -> None:
        self.figure.update(dict(data=self.data, layout=self.layout))
        self.pane.object = self.figure
    
    def servable(self) -> pn.pane.Plotly:
        return pn.Card(
            self.pane,
            collapsible=False,
            hide_header=True
        )

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
        self.map = SiteMap()
        self.state = self.filters.state

        # Callbacks
        def update_map(event, callback_type: CallbackType) -> None:
            if event is None:
                return
            
            # Limit number of state updates
            if self.state == self.filters.state:
                return
            
            # Update state
            self.state = self.filters.state

            # Select data
            data = self.data[self.state.evaluation][self.state.domain][self.state.configuration]
            geometry = self.routelinks[self.state.domain].select(["nwm_feature_id", "latitude", "longitude"])

            # Filter data
            value_column = self.state.metric + self.state.confidence
            columns = [value_column, "start_date", "end_date", "nwm_feature_id", "usgs_site_code", "sample_size"]
            if self.state.configuration in PREDICTION_RESAMPLING:
                columns.append("lead_time_hours_min")
                data = data.filter(pl.col("lead_time_hours_min") == self.state.lead_time)
            data = data.select(columns).join(geometry, on="nwm_feature_id", how="left").collect()
            
            # Update map
            cmin, cmax = METRIC_PLOTTING_LIMITS[self.state.metric]
            self.map.update(
                values=data[value_column].to_numpy(),
                latitude=data["latitude"].to_numpy(),
                longitude=data["longitude"].to_numpy(),
                value_name=self.state.metric_label,
                cmin=cmin,
                cmax=cmax,
                domain=self.state.domain
            )
            self.map.refresh()
        self.filters.register_callback(update_map)

        # Layout
        self.template.main.append(
            pn.Row(
                self.filters.servable(),
                self.map.servable()
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
