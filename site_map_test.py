"""
An interactive mapping interface.
"""
from typing import TypedDict
from dataclasses import dataclass, field
import warnings

import polars as pl
import panel as pn
from panel.viewable import Viewer
import plotly.graph_objects as go
import colorcet as cc

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
            clickmode="event",
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
    colorbar_title: str, optional
        Title to display next to colorbar.
    colorbar_limits: tuple[float, float], optional
        Colorbar range.
    """
    store: pl.LazyFrame
    latitude_column: str = "latitude"
    longitude_column: str = "longitude"
    color_column: str | None = None
    custom_data_columns: list[str] | None = None
    custom_data_labels: list[str] | None = None
    size_column: str | None = None
    color_scale: list[str] = field(default_factory=lambda: cc.gouldian)
    marker_size: float = 15.0
    marker_color: str = "black"
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
        hover_template = "Longitude: %{lon}<br>Latitude: %{lat}"

        # Set columns
        columns = [
            self.latitude_column,
            self.longitude_column
        ]
        if self.size_column:
            columns.append(self.size_column)
        if self.custom_data_columns:
            columns += self.custom_data_columns
            for idx, c in enumerate(self.custom_data_columns):
                if self.custom_data_labels:
                    l = self.custom_data_labels[idx]
                else:
                    l = c
                hover_template = f"{l}: " +  "%{customdata[" + str(idx) + "]}<br>" + hover_template
        if self.color_column:
            columns.append(self.color_column)
            hover_template = (
                f"{self.colorbar_title}: "
                "%{marker.color:.2f}<br>" + hover_template
                )

        # Load data
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
                size=size
            )
        else:
            markers = dict(
                color=color,
                colorbar=dict(title=dict(side="right", text=self.colorbar_title)),
                size=size,
                colorscale=self.color_scale
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
            latitude_column: str | None = None,
            longitude_column: str | None = None,
            color_column: str | None = None,
            custom_data_columns: list[str] | None = None,
            size_column: str | None = None,
            color_scale: list[str] | None = None,
            marker_size: float | None = None,
            marker_color: str | None = None,
            colorbar_title: str | None = None,
            colorbar_limits: tuple[float, float] | None = None
        ) -> None:
        """
        Update layer properties. Assumes render has already been called.
    
        Parameters
        ----------
        latitude_column: str, optional
            Column in data to use as latitude.
        longitude_column: str, optional
            Column in data to use as longitude.
        color_column: str, optional
            Column in data used to color markers.
        custom_data_columns: list[str], optional
            Columns in data to display on hover.
        size_column: str, optional
            Column in data used to set marker size.
        color_scale: list[str], optional
            Default colorscale.
        marker_size: float, optional
            Marker size used for uniform sizing.
        marker_color: str, optional
            Marker color used for uniform color.
        colorbar_title: str, optional
            Title to display next to colorbar.
        colorbar_limits: tuple[float, float], optional
            Colorbar range.
        """
        # Check for trace
        if self._trace is None:
            raise RuntimeError("Cannot update unrendered layer")

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

        # Load data
        if columns:
            data = self.store.select(columns).collect()

        # Build update
        if latitude_column:
            self._trace.update({"lat": data[latitude_column]})
        if longitude_column:
            self._trace.update({"lon": data[longitude_column]})
        if color_column:
            self._trace["marker"].update({"color": data[color_column]})
        if size_column:
            self._trace["marker"].update({"size": data[size_column]})
        if custom_data_columns:
            self._trace.update({"customdata": data[custom_data_columns]})
        if color_scale:
            self.color_scale = color_scale
            self._trace["marker"].update({"colorscale": color_scale})
        if marker_size:
            self.marker_size = marker_size
            self._trace["marker"].update({"size": marker_size})
        if marker_color:
            self.marker_color = marker_color
            self._trace["marker"].update({"color": marker_color})
        if colorbar_title:
            self.colorbar_title = colorbar_title
            self._trace["marker"]["colorbar"]["title"]["text"] = colorbar_title
        if colorbar_limits:
            self.colorbar_limits = colorbar_limits
            self._trace["marker"].update({"cmin": colorbar_limits[0]})
            self._trace["marker"].update({"cmax": colorbar_limits[1]})

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
        Dict of MapLayer keyed to labels.
    domains: dict[str, MapFocus]
        Dict of MapFocus keyed to labels.
    default_domain: str, optional
        Sets default domain returned to when resetting map.
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
            "data": [v.render() for v in self.layers.values()],
            "layout": self.layout
        })

        # Handle double click
        def reset_layout(event) -> None:
            self.layout = self.layouts[self.default_domain]
        pn.bind(reset_layout, self.pane.param.doubleclick_data, watch=True)

    def switch_domain(self, label: str) -> None:
        """
        Change map layout to another domain.

        Parameters
        ----------
        label: str
            DomainView label.
        """
        self.layout = self.layouts[label]
    
    def update_layer(self, layer_label: str, **kwargs) -> None:
        """
        Update map layer.

        Parameters
        ----------
        layer_label: str
            Layer key.
        kwargs: any
            Keyword arguments passed directly to MapLayer.update.
        """
        self.layers[layer_label].update(**kwargs)
    
    def add_layer(self, layer_label: str, layer: MapLayer) -> None:
        """
        Add map layer.

        Parameters
        ----------
        layer_label: str
            Layer key.
        layer: MapLayer
            MapLayer object.
        """
        self.layers[layer_label] = layer
    
    def remove_layer(self, layer_label: str) -> MapLayer | None:
        """
        Remove map layer.

        Parameters
        ----------
        layer_label: str
            Layer key.
        """
        return self.layers.pop(layer_label, None)

    def refresh(self) -> None:
        """
        Send current state of data and layout to frontend.
        """
        self.pane.object = {
            "data": [v.trace for v in self.layers.values()],
            "layout": self.layout
        }

    def __panel__(self) -> pn.Card:
        return pn.Card(
            self.pane,
            collapsible=False,
            hide_header=True
        )

def main():
    # Domains
    domains = {
        "All": MapFocus(center=Coordinates(lat=38.83348, lon=-100.97612), zoom=2),
        "Alaska": MapFocus(center=Coordinates(lat=60.84683, lon=-149.05659), zoom=5),
        "CONUS": MapFocus(center=Coordinates(lat=38.83348, lon=-93.97612), zoom=3),
        "Hawaii": MapFocus(center=Coordinates(lat=21.24988, lon=-157.59606), zoom=6),
        "Puerto Rico": MapFocus(center=Coordinates(lat=18.21807, lon=-66.32802), zoom=8)
    }

    # Data
    # import numpy as np
    # rng = np.random.default_rng(seed=2025)
    # N = 10_000
    # pl.DataFrame({
    #     "USGS site code": ["02"+str(i) for i in range(N)],
    #     "Latitude": rng.uniform(24, 52, N),
    #     "Longitude": rng.uniform(-124, -67, N),
    #     "Nash-Sutcliffe efficiency": rng.uniform(-1.0, 1.0, N),
    #     "Relative mean": rng.uniform(0.0, 2.0, N)
    # }).write_parquet("fake_data.parquet")
    data = pl.scan_parquet("fake_data.parquet")

    # Layers
    layers = {
        "metrics": MapLayer(
            store=data,
            latitude_column="Latitude",
            longitude_column="Longitude",
            color_column="Nash-Sutcliffe efficiency",
            custom_data_columns=["USGS site code"],
            colorbar_title="Nash-Sutcliffe efficiency",
            colorbar_limits=(-1.0, 1.0)
        )
    }

    # Setup map
    site_map = SiteMap(layers, domains)

    # Add domain selector
    domain_selector = pn.widgets.Select(name="Domain", options=list(domains.keys()))
    def update_domain(domain) -> None:
        site_map.switch_domain(domain)
        site_map.refresh()
    pn.bind(update_domain, domain_selector.param.value, watch=True)

    # Update selector when map is reset
    def reset_selector(event) -> None:
        domain_selector.value = site_map.default_domain
    pn.bind(reset_selector, site_map.pane.param.doubleclick_data, watch=True)

    # Metric selector
    limits = {
        "Nash-Sutcliffe efficiency": (-1.0, 1.0),
        "Relative mean": (0.0, 2.0)
    }
    metric_selector = pn.widgets.Select(name="Metric", options=list(limits.keys()))
    def update_metric(event) -> None:
        site_map.update_layer(
            "metrics",
            color_column=event,
            colorbar_title=event,
            colorbar_limits=limits[event]
        )
        site_map.refresh()
    pn.bind(update_metric, metric_selector.param.value, watch=True)

    # Layers
    extra_layers = {
        "USGS streamflow gages": MapLayer(
            store=pl.scan_parquet("data/site_information.parquet"),
            custom_data_columns=[
                "contributing_drainage_area",
                "drainage_area",
                "HUC",
                "site_name",
                "usgs_site_code"
            ],
            custom_data_labels=[
                "Contrib. Drain. Area (sq.mi.)",
                "Drainage Area (sq.mi.)",
                "HUC",
                "Site name",
                "USGS site code"
            ],
            marker_color="rgba(23, 225, 189, 0.75)",
            marker_size=10
        ),
        "National Inventory of Dams": MapLayer(
            store=pl.scan_parquet("data/NID.parquet"),
            custom_data_columns=[
                "name",
                "riverName",
                "maxStorage",
                "normalStorage",
                "maxDischarge",
                "drainageArea"
            ],
            custom_data_labels=[
                "Dam Name",
                "River Name",
                "Drainage Area (sq.mi.)",
                "Maximum Storage (ac-ft)",
                "Normal Storage (ac-ft)",
                "Maximum Discharge (CFS)"
            ],
            marker_color="rgba(255, 141, 0, 0.75)",
            marker_size=10
        )
    }
    checkbox = pn.widgets.CheckBoxGroup(
        name="Additional layers",
        options=list(extra_layers.keys()),
        inline=True
    )
    def update_layers(event) -> None:
        # Check each layer
        for k, v in extra_layers.items():
            # Add layer
            if k in event:
                # Render layer
                if v.trace is None:
                    v.render()

                # Add to map
                site_map.add_layer(k, v)
            else:
                # Remove from map
                site_map.remove_layer(k)

                # Clear layer
                v.clear()
        
        # Refresh
        site_map.refresh()
    pn.bind(update_layers, checkbox.param.value, watch=True)

    # Serve the dashboard
    pn.serve(pn.Column(site_map, domain_selector, metric_selector, checkbox))

if __name__ == "__main__":
    main()
