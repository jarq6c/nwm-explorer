"""
An interactive mapping interface.
"""
from typing import TypedDict, Any
from dataclasses import dataclass, field
import warnings
from enum import StrEnum
from pathlib import Path
from datetime import datetime
from collections.abc import Callable

import polars as pl
import panel as pn
from panel.viewable import Viewer
import plotly.graph_objects as go
import colorcet as cc
from pydantic import BaseModel

class ModelDomain(StrEnum):
    """National Water Model domains."""
    alaska = "Alaska"
    conus = "CONUS"
    hawaii = "Hawaii"
    puertorico = "Puerto Rico"

class ModelConfiguration(StrEnum):
    """National Water Model configurations."""
    analysis_assim_extend_alaska_no_da = "analysis_assim_extend_alaska_no_da"
    analysis_assim_extend_no_da = "analysis_assim_extend_no_da"
    analysis_assim_hawaii_no_da = "analysis_assim_hawaii_no_da"
    analysis_assim_puertorico_no_da = "analysis_assim_puertorico_no_da"
    medium_range_mem1 = "medium_range_mem1"
    medium_range_blend = "medium_range_blend"
    medium_range_no_da = "medium_range_no_da"
    medium_range_alaska_mem1 = "medium_range_alaska_mem1"
    medium_range_blend_alaska = "medium_range_blend_alaska"
    medium_range_alaska_no_da = "medium_range_alaska_no_da"
    short_range = "short_range"
    short_range_alaska = "short_range_alaska"
    short_range_hawaii = "short_range_hawaii"
    short_range_hawaii_no_da = "short_range_hawaii_no_da"
    short_range_puertorico = "short_range_puertorico"
    short_range_puertorico_no_da = "short_range_puertorico_no_da"

DOMAIN_CONFIGURATIONS: dict[ModelDomain, dict[str, ModelConfiguration]] = {
    ModelDomain.alaska: {
        "Extended Analysis (MRMS, No-DA)": ModelConfiguration.analysis_assim_extend_alaska_no_da,
        "Medium Range Forecast (GFS, Deterministic)": ModelConfiguration.medium_range_alaska_mem1,
        "Medium Range Forecast (NBM, Deterministic)": ModelConfiguration.medium_range_blend_alaska,
        "Medium Range Forecast (GFS, Deterministic, No-DA)": ModelConfiguration.medium_range_alaska_no_da,
        "Short Range Forecast (HRRR)": ModelConfiguration.short_range_alaska
    },
    ModelDomain.conus: {
        "Extended Analysis (MRMS, No-DA)": ModelConfiguration.analysis_assim_extend_no_da,
        "Medium Range Forecast (GFS, Deterministic)": ModelConfiguration.medium_range_mem1,
        "Medium Range Forecast (NBM, Deterministic)": ModelConfiguration.medium_range_blend,
        "Medium Range Forecast (GFS, Deterministic, No-DA)": ModelConfiguration.medium_range_no_da,
        "Short Range Forecast (HRRR)": ModelConfiguration.short_range
    },
    ModelDomain.hawaii: {
        "Analysis (MRMS, No-DA)": ModelConfiguration.analysis_assim_hawaii_no_da,
        "Short Range Forecast (WRF-ARW)": ModelConfiguration.short_range_hawaii,
        "Short Range Forecast (WRF-ARW, No-DA)": ModelConfiguration.short_range_hawaii_no_da
    },
    ModelDomain.puertorico: {
        "Analysis (MRMS, No-DA)": ModelConfiguration.analysis_assim_puertorico_no_da,
        "Short Range Forecast (WRF-ARW)": ModelConfiguration.short_range_puertorico,
        "Short Range Forecast (WRF-ARW, No-DA)": ModelConfiguration.short_range_puertorico_no_da
    }
}
"""
Mapping from ModelDomain to pretty string representations of model configurations.
Pretty strings map to model ModelConfiguration enums.
"""

class Metric(StrEnum):
    """Metric names."""
    nash_sutcliffe_efficiency = "Nash-Sutcliffe efficiency"
    relative_mean_bias = "Relative mean bias"
    pearson_correlation_coefficient = "Pearson correlation coefficient"
    relative_mean = "Relative mean"
    relative_standard_deviation = "Relative standard deviation"
    kling_gupta_efficiency = "Kling-Gupta efficiency"

class MetricConfidence(StrEnum):
    """Metric value estimations (typically 95% confidence)."""
    _point = "Point"
    _lower = "Lower"
    _upper = "Upper"

class Coordinates(TypedDict):
    """
    TypedDict specifying a point location in WGS84 decimal coordinates.

    Attributes
    ----------
    lat: float
        Latitude.
    lon: float
        Longitude
    """
    lat: float
    lon: float

class MapFocus(TypedDict):
    """
    TypedDict specifying default center coordinates and zoom level
    for a plotly map.
    
    Attributes
    ----------
    center: Coordinates
        Coordinates of map center.
    zoom: float
        Zoom level.
    """
    center: Coordinates
    zoom: float

@dataclass
class DomainView:
    """
    Defines a default view for a specific domain.

    Attributes
    ----------
    label: str
        String label.
    layout: Layout
        Plotly layout that defines map center and default zoom.
    """
    label: str
    layout: go.Layout

def generate_domain_view(label: str, focus: MapFocus) -> DomainView:
    """
    Returns a plotly Layout with center and zoom determined by focus and
    uirevision label.

    Parameters
    ----------
    label: str, required
        Handle to use for uirevision (maintains map focus between updates).
    focus: MapFocus, required
        Specifies map center and zoom level.
    
    Returns
    -------
    Layout
    """
    return DomainView(
        label=label,
        layout=go.Layout(
            showlegend=False,
            height=540,
            width=850,
            margin=dict(l=0, r=0, t=0, b=0),
            map=dict(
                style="satellite-streets",
                **focus
            ),
            clickmode="event+select",
            modebar=dict(
                remove=["lasso", "select", "resetview"],
                orientation="v"
            ),
            dragmode="zoom",
            uirevision=label
    ))

@dataclass
class MapLayer:
    """
    Defines a single map layer.
    
    Attributes
    ----------
    store: pl.LazyFrame
        Polars LazyFrame pointing at data to plot.
    filters: list[pl.Expr]
        List of polars.Expr applied to store.
    latitude_column: str, default 'latitude'
        Column in data to use as latitude.
    longitude_column: str, default 'longitude'
        Column in data to use as longitude.
    color_column: str, optional
        Column in data used to color markers.
    custom_data_columns: list[str], optional
        Columns in data to display on hover.
    custom_data_labels: list[str], optional
        Labels to use with custom data on hover.
    size_column: str, optional
        Column in data used to set marker size.
    color_scale: list[str], default colorcet.gouldian
        Default colorscale.
    marker_size: float, default 15
        Marker size used for uniform sizing.
    marker_color: str, default 'black'
        Marker color used for uniform color.
    marker_symbol: str, default 'circle'
        Marker symbol used for uniform symbols. Currently, plotly maps only
        support 'circle'.
    colorbar_title: str, optional
        Title to display next to colorbar.
    colorbar_limits: tuple[float, float], optional
        Colorbar range.
    """
    store: pl.LazyFrame
    filters: list[pl.Expr] | None = None
    latitude_column: str = "latitude"
    longitude_column: str = "longitude"
    color_column: str | None = None
    custom_data_columns: list[str] | None = None
    custom_data_labels: list[str] | None = None
    size_column: str | None = None
    color_scale: list[str] = field(default_factory=lambda: cc.gouldian)
    marker_size: float = 15.0
    marker_color: str = "black"
    marker_symbol: str = "circle"
    colorbar_title: str | None = None
    colorbar_limits: tuple[float, float] | None = None
    _trace: go.Scattermap | None = None

    def render(self) -> go.Scattermap:
        """
        Renders the layer using plotly.

        Returns
        -------
        plotly.graph_objects.Scattermap
        """
        # Warn for uncleared trace
        if self._trace is not None:
            warnings.warn("Overwriting layer", RuntimeWarning)
        
        # Hover template
        hover_template = ""

        # Set columns
        columns = [
            self.latitude_column,
            self.longitude_column
        ]
        if self.size_column:
            columns.append(self.size_column)
        if self.color_column:
            columns.append(self.color_column)
            hover_template += (
                f"{self.colorbar_title}: "
                "%{marker.color:.2f}<br>"
                )
        if self.custom_data_columns:
            columns += self.custom_data_columns
            for idx, c in enumerate(self.custom_data_columns):
                if self.custom_data_labels:
                    l = self.custom_data_labels[idx]
                else:
                    l = c
                hover_template += f"{l}: " +  "%{customdata[" + str(idx) + "]}<br>"
        hover_template += "Longitude: %{lon}<br>Latitude: %{lat}"

        # Load data
        if self.filters:
            data = self.store.select(columns).filter(*self.filters).collect()
        else:
            data = self.store.select(columns).collect()

        # Set marker color
        if self.color_column:
            color = data[self.color_column]
        else:
            color = self.marker_color

        # Set marker size
        if self.size_column:
            size = data[self.color_column]
        else:
            size = self.marker_size

        # Setup markers
        if isinstance(color, str):
            markers = dict(
                color=color,
                size=size,
                symbol=self.marker_symbol
            )
        else:
            markers = dict(
                color=color,
                colorbar=dict(title=dict(side="right", text=self.colorbar_title)),
                size=size,
                colorscale=self.color_scale,
                symbol=self.marker_symbol
            )
            if self.colorbar_limits:
                markers.update(dict(
                    cmin=self.colorbar_limits[0],
                    cmax=self.colorbar_limits[1]
            ))

        # Instantiate map
        self._trace = go.Scattermap(
            lon=data[self.longitude_column],
            lat=data[self.latitude_column],
            showlegend=False,
            name="",
            mode="markers",
            marker=markers,
            customdata=data[self.custom_data_columns],
            hovertemplate=hover_template
        )
        return self._trace

    def update(
            self,
            store: pl.LazyFrame | None = None,
            filters: list[pl.Expr] | None = None,
            latitude_column: str | None = None,
            longitude_column: str | None = None,
            color_column: str | None = None,
            custom_data_columns: list[str] | None = None,
            custom_data_labels: list[str] | None = None,
            size_column: str | None = None,
            color_scale: list[str] | None = None,
            marker_size: float | None = None,
            marker_color: str | None = None,
            marker_symbol: str | None = None,
            colorbar_title: str | None = None,
            colorbar_limits: tuple[float, float] | None = None
        ) -> None:
        """
        Update layer properties. Assumes render has already been called.
    
        Parameters
        ----------
        store: pl.LazyFrame, optional
            Polars LazyFrame pointing at data to plot.
        latitude_column: str, optional
            Column in data to use as latitude.
        longitude_column: str, optional
            Column in data to use as longitude.
        color_column: str, optional
            Column in data used to color markers.
        custom_data_columns: list[str], optional
            Columns in data to display on hover.
        custom_data_labels: list[str], optional
            Labels to use with custom data on hover.
        size_column: str, optional
            Column in data used to set marker size.
        color_scale: list[str], optional
            Default colorscale.
        marker_size: float, optional
            Marker size used for uniform sizing.
        marker_color: str, optional
            Marker color used for uniform color.
        marker_symbol: str, optional
            Marker symbol used for uniform symbols. Currently, plotly maps only
            support 'circle'.
        colorbar_title: str, optional
            Title to display next to colorbar.
        colorbar_limits: tuple[float, float], optional
            Colorbar range.
        """
        # Set columns
        columns = []
        if latitude_column:
            self.latitude_column = latitude_column
            columns.append(latitude_column)
        if longitude_column:
            self.longitude_column = longitude_column
            columns.append(longitude_column)
        if color_column:
            self.color_column = color_column
            columns.append(color_column)
        if size_column:
            self.size_column = size_column
            columns.append(size_column)
        if custom_data_columns:
            self.custom_data_columns = custom_data_columns
            columns += custom_data_columns

        # Additional attributes
        if store:
            self.store = store
        if custom_data_labels:
            self.custom_data_labels = custom_data_labels
        if color_scale:
            self.color_scale = color_scale
        if marker_size:
            self.marker_size = marker_size
        if marker_color:
            self.marker_color = marker_color
        if marker_symbol:
            self.marker_symbol = marker_symbol
        if colorbar_title:
            self.colorbar_title = colorbar_title
        if colorbar_limits:
            self.colorbar_limits = colorbar_limits
        if filters:
            self.filters = filters

        # Check for trace
        if self._trace is None:
            return

        # Load data
        if columns:
            if self.filters:
                data = self.store.select(columns).filter(*self.filters).collect()
            else:
                data = self.store.select(columns).collect()

        # Build update
        hover_template = ""
        if latitude_column:
            self._trace.update({"lat": data[latitude_column]})
        if longitude_column:
            self._trace.update({"lon": data[longitude_column]})
        if color_column:
            hover_template += (
                f"{self.colorbar_title}: "
                "%{marker.color:.2f}<br>"
                )
            self._trace["marker"].update({"color": data[color_column]})
        if size_column:
            self._trace["marker"].update({"size": data[size_column]})
        if custom_data_columns:
            self._trace.update({"customdata": data[custom_data_columns]})
        if color_scale:
            self._trace["marker"].update({"colorscale": color_scale})
        if marker_size:
            self._trace["marker"].update({"size": marker_size})
        if marker_color:
            self._trace["marker"].update({"color": marker_color})
        if marker_symbol:
            self._trace["marker"].update({"symbol": marker_symbol})
        if colorbar_title:
            self._trace["marker"]["colorbar"]["title"]["text"] = colorbar_title
        if colorbar_limits:
            self._trace["marker"].update({"cmin": colorbar_limits[0]})
            self._trace["marker"].update({"cmax": colorbar_limits[1]})

        # Hover template
        if self.custom_data_columns:
            for idx, c in enumerate(self.custom_data_columns):
                if self.custom_data_labels:
                    l = self.custom_data_labels[idx]
                else:
                    l = c
                hover_template += f"{l}: " +  "%{customdata[" + str(idx) + "]}<br>"
        hover_template += "Longitude: %{lon}<br>Latitude: %{lat}"

        # Update hover template
        self._trace["hovertemplate"] = hover_template

    def clear(self) -> None:
        """
        Reset rendered layer. Sets self.trace to None.
        """
        self._trace = None

    @property
    def trace(self) -> go.Scattermap | None:
        """
        Returns underlying plotly object. Returns None if layer has not been
        rendered.

        Returns
        -------
        plotly.graph_objects.Scattermap | None
        """
        return self._trace

class SiteMap(Viewer):
    """
    Shows a plotly map in a card interface.

    Parameters
    ----------
    layers: dict[str, MapLayer]
        Dict of MapLayer keyed to hashable labels for each layer.
    domains: dict[str, MapFocus]
        Dict of MapFocus keyed to hashable labels for each focus area.
    default_domain: str, optional
        Sets default domain returned to when resetting map. Defaults to the
        first key in domains.
    params: any
        Additional keyword arguments passed to panel.viewable.Viewer.
    """
    def __init__(
            self,
            layers: dict[str, MapLayer],
            domains: dict[str, MapFocus],
            default_domain: str | None = None,
            **params
        ) -> None:
        super().__init__(**params)

        # Data
        self._stub_layer = go.Scattermap()
        self.layers = layers

        # Layouts
        domain_views = [generate_domain_view(l, f) for l, f in domains.items()]
        self.layouts = {d.label: d.layout for d in domain_views}

        # Default layout
        if default_domain is None:
            self.default_domain = domain_views[0].label
        else:
            self.default_domain = default_domain
        self.layout = self.layouts[self.default_domain]

        # Main figure (map)
        self.pane = pn.pane.Plotly({
            "data": [list(layers.values())[0].render()],
            "layout": self.layout
        })

        # Widgets
        self.domain_selector = pn.widgets.Select(name="Domain",
            options=list(domains.keys()))
        self.layer_selector = pn.widgets.CheckBoxGroup(
            name="Layers",
            options=list(layers.keys()),
            value=list(layers.keys())[0:1]
        )

        # Add/Remove layers
        def update_layers(layer_keys: list[str]) -> None:
            for k, v in self.layers.items():
                if k in layer_keys:
                    if v.trace is None:
                        v.render()
                else:
                    v.clear()
            self.refresh()
        pn.bind(update_layers, self.layer_selector.param.value, watch=True)

        # Switch domain view
        def switch_domain(domain: str) -> None:
            self.layout = self.layouts[domain]
            self.refresh()
        pn.bind(switch_domain, self.domain_selector.param.value, watch=True)

        # Handle double click
        def reset_layout(event) -> None:
            self.layout = self.layouts[self.default_domain]
            self.domain_selector.value = self.default_domain
        pn.bind(reset_layout, self.pane.param.doubleclick_data, watch=True)

        # Handle single click
        self.click_data = None
        def log_click_event(event) -> None:
            data = event["points"][0]
            key = list(self.layers.keys())[data["curveNumber"]]
            custom_data = data["customdata"]
            columns = self.layers[key].custom_data_columns
            self.click_data = {col: val for col, val in zip(columns, custom_data)}
            self.click_data["layer"] = key
        pn.bind(log_click_event, self.pane.param.click_data, watch=True)

    def refresh(self) -> None:
        """
        Send current state of data and layout to frontend.
        """
        # Check for layers
        data = [v.trace for v in self.layers.values() if v.trace]
        if not data:
            data = [self._stub_layer]

        # Update view
        self.pane.object = {
            "data": data,
            "layout": self.layout
        }
    
    def register_domain_callback(self, function: Callable[..., None], **params) -> None:
        """
        Register a callback that triggers when the domain changes.
        
        Parameters
        ----------
        function: Callable
            Function called when domain is updated. First argument must accept a
            string (domain).
        params:
            Additional keyword arguments passed to function.
        """
        pn.bind(function, self.domain_selector.param.value, **params, watch=True)

    def __panel__(self) -> pn.Card:
        return pn.Card(
            self.pane,
            collapsible=False,
            hide_header=True
        )
    
    @property
    def domain(self) -> str:
        """Currently selected domain."""
        return self.domain_selector.value

DOMAIN_VIEWS: dict[ModelDomain, MapFocus] = {
    ModelDomain.conus: MapFocus(center=Coordinates(lat=38.83348, lon=-93.97612), zoom=3),
    ModelDomain.alaska: MapFocus(center=Coordinates(lat=60.84683, lon=-149.05659), zoom=5),
    ModelDomain.hawaii: MapFocus(center=Coordinates(lat=21.24988, lon=-157.59606), zoom=6),
    ModelDomain.puertorico: MapFocus(center=Coordinates(lat=18.21807, lon=-66.32802), zoom=8)
}
"""Default map centers and zoom levels for NWM domains."""

METRIC_PLOTTING_LIMITS: dict[Metric, tuple[float, float]] = {
    Metric.relative_mean_bias: (-1.0, 1.0),
    Metric.pearson_correlation_coefficient: (-1.0, 1.0),
    Metric.nash_sutcliffe_efficiency: (-1.0, 1.0),
    Metric.relative_mean: (0.0, 2.0),
    Metric.relative_standard_deviation: (0.0, 2.0),
    Metric.kling_gupta_efficiency: (-1.0, 1.0)
}
"""Mapping from Metrics to plotting limits (cmin, cmax)."""

class EvaluationSpec(BaseModel):
    startDT: datetime
    endDT: datetime
    directory: Path
    files: dict[ModelDomain, dict[ModelConfiguration, Path]]

class EvaluationRegistry(BaseModel):
    evaluations: dict[str, EvaluationSpec]

class EditablePlayer(Viewer):
    """
    DiscretePlayer that refreshes when changing options.

    Parameters
    ----------
    params: any
        Keyword arguments passed to pn.widgets.DiscretePlayer.
    """
    def __init__(
            self,
            **params
        ) -> None:
        # Initialize
        self._params: dict[str, Any] = params
        self._container = pn.pane.Placeholder(
            pn.widgets.DiscretePlayer(**params)
        )

    def __panel__(self) -> pn.pane.Placeholder:
        return self._container

    def update(self, **params) -> None:
        """
        Use this method to update underlying parameters of DiscretePlayer.
        """
        # Handle options
        if ("options" in params) and ("value" not in params):
            values = params["options"]
            v = self._container.object.value
            self._params.update({"value": v if v in values else values[0]})

        # Apply remaining updates
        self._params.update(params)

        # Refresh widget
        self._container.object = pn.widgets.DiscretePlayer(**self._params)

LEAD_TIME_VALUES: dict[ModelConfiguration, list[int]] = {
    ModelConfiguration.medium_range_mem1: [l for l in range(0, 240, 24)],
    ModelConfiguration.medium_range_blend: [l for l in range(0, 240, 24)],
    ModelConfiguration.medium_range_no_da: [l for l in range(0, 240, 24)],
    ModelConfiguration.medium_range_alaska_mem1: [l for l in range(0, 240, 24)],
    ModelConfiguration.medium_range_blend_alaska: [l for l in range(0, 240, 24)],
    ModelConfiguration.medium_range_alaska_no_da: [l for l in range(0, 240, 24)],
    ModelConfiguration.short_range: [l for l in range(0, 18, 6)],
    ModelConfiguration.short_range_alaska: [l for l in range(0, 45, 5)],
    ModelConfiguration.short_range_hawaii: [l for l in range(0, 48, 6)],
    ModelConfiguration.short_range_hawaii_no_da: [l for l in range(0, 48, 6)],
    ModelConfiguration.short_range_puertorico: [l for l in range(0, 48, 6)],
    ModelConfiguration.short_range_puertorico_no_da: [l for l in range(0, 48, 6)]
}
"""Mapping from model ModelConfiguration enums to lists of lead time integers (hours)."""

def main(root: Path = Path("./data")):
    # Setup registry
    registry_file = root / "test_registry.json"
    with registry_file.open("r") as fo:
        evaluation_registry = EvaluationRegistry.model_validate_json(fo.read())

    # Evaluation selector
    evaluation_selector = pn.widgets.Select(
        name="Evaluation",
        options=list(evaluation_registry.evaluations.keys())
    )
    threshold_filter = pn.widgets.Select(
        name="Streamflow Threshold (≥)",
        options=[
            "100% AEP-USGS (All data)"
        ]
    )

    # Initialize state
    default_domain = ModelDomain.conus
    configuration_selector = pn.widgets.Select(
        name="Model Configuration",
        options=list(
        DOMAIN_CONFIGURATIONS[default_domain].keys()
        ))
    evaluation = evaluation_registry.evaluations[evaluation_selector.value]
    configuration = DOMAIN_CONFIGURATIONS[default_domain][configuration_selector.value]
    if configuration in LEAD_TIME_VALUES:
        lead_time_options = LEAD_TIME_VALUES[configuration]
    else:
        lead_time_options = [0]

    # Data and geometry
    geometry_file = root / "parquet" / default_domain.name / "routelink.parquet"
    geometry = pl.scan_parquet(geometry_file).select(
        ["nwm_feature_id", "latitude", "longitude"])
    data = pl.scan_parquet(evaluation.files[default_domain][configuration]).join(
        geometry, on="nwm_feature_id", how="left")

    # Initial metric layer
    initial_column = list(Metric)[0].name + list(MetricConfidence)[0].name

    # Setup map
    metrics_key = "Metrics"
    site_map = SiteMap(
        layers={
            metrics_key: MapLayer(
                store=data,
                color_column=initial_column,
                custom_data_columns=[
                    "usgs_site_code",
                    "nwm_feature_id"
                ],
                custom_data_labels=[
                    "USGS site code",
                    "NWM feature ID"
                ],
                colorbar_title=list(Metric)[0],
                colorbar_limits=(-1.0, 1.0)
            ),
            "USGS streamflow gages": MapLayer(
                store=pl.scan_parquet("data/site_information.parquet"),
                custom_data_columns=[
                    "site_name",
                    "usgs_site_code",
                    "HUC",
                    "drainage_area",
                    "contributing_drainage_area"
                ],
                custom_data_labels=[
                    "Site name",
                    "USGS site code",
                    "HUC",
                    "Drainage Area (sq.mi.)",
                    "Contrib. Drain. Area (sq.mi.)"
                ],
                marker_color="rgba(23, 225, 189, 0.75)",
                marker_size=10
            ),
            "National Inventory of Dams": MapLayer(
                store=pl.scan_parquet("data/NID.parquet"),
                custom_data_columns=[
                    "riverName",
                    "name",
                    "maxDischarge",
                    "normalStorage",
                    "maxStorage",
                    "drainageArea"
                ],
                custom_data_labels=[
                    "River Name",
                    "Dam Name",
                    "Maximum Discharge (CFS)",
                    "Normal Storage (ac-ft)",
                    "Maximum Storage (ac-ft)",
                    "Drainage Area (sq.mi.)"
                ],
                marker_color="rgba(255, 141, 0, 0.75)",
                marker_size=10
            )
        },
        domains=DOMAIN_VIEWS
    )

    # Update configurations
    def update_configurations(domain):
        if domain is None:
            return
        configuration_selector.options = list(
            DOMAIN_CONFIGURATIONS[domain].keys()
        )
    site_map.register_domain_callback(update_configurations)

    # Lead times
    lead_time_selector = EditablePlayer(
        name="Minimum lead time (hours)",
        options=lead_time_options,
        show_loop_controls=False,
        visible_buttons=["previous", "next"],
        width=300,
        value=lead_time_options[0]
    )
    def update_lead_times(configuration_selection: str) -> None:
        configuration = DOMAIN_CONFIGURATIONS[site_map.domain][configuration_selection]
        if configuration in LEAD_TIME_VALUES:
            lead_time_options = LEAD_TIME_VALUES[configuration]
        else:
            lead_time_options = [0]
        lead_time_selector.update(options=lead_time_options)
    pn.bind(update_lead_times, configuration_selector.param.value, watch=True)

    # Metric selector
    metric_selector = pn.widgets.Select(name=metrics_key, options=list(Metric))
    confidence_selector = pn.widgets.Select(name="95% confidence interval", options=list(MetricConfidence))
    def update_metric(metric: str, confidence: str) -> None:
        # Set column
        m = Metric(metric)
        column = m.name + MetricConfidence(confidence).name

        # Update rendered layer
        site_map.layers[metrics_key].update(
            color_column=column,
            colorbar_title=m,
            colorbar_limits=METRIC_PLOTTING_LIMITS[m]
        )

        # Update map
        if site_map.layers[metrics_key].trace is not None:
            site_map.refresh()
    pn.bind(update_metric, metric=metric_selector.param.value,
        confidence=confidence_selector.param.value, watch=True)

    # Serve the dashboard
    pn.serve(pn.Column(
        site_map,
        evaluation_selector,
        site_map.domain_selector,
        configuration_selector,
        threshold_filter,
        metric_selector,
        confidence_selector,
        lead_time_selector,
        site_map.layer_selector
        ))

if __name__ == "__main__":
    main()
