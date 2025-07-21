"""Plotting."""
import numpy as np
import numpy.typing as npt
import pandas as pd
import panel as pn
import plotly.graph_objects as go
import colorcet as cc

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
        self.lat_min = None
        self.lat_max = None
        self.lon_min = None
        self.lon_max = None
        def apply_relayout_data(data) -> None:
            if data is None:
                return
            if "map.center" in data:
                self.layout["map"]["center"].update(data["map.center"])
            if "map.zoom" in data:
                self.layout["map"].update(dict(zoom=data["map.zoom"]))
            if "map._derived" in data:
                self.lat_max = data["map._derived"]["coordinates"][0][1]
                self.lat_min = data["map._derived"]["coordinates"][2][1]
                self.lon_max = data["map._derived"]["coordinates"][1][0]
                self.lon_min = data["map._derived"]["coordinates"][0][0]
            if "width" in data:
                self.layout["map"]["center"].update(DEFAULT_CENTER[self.domain])
                self.layout["map"].update(dict(zoom=DEFAULT_ZOOM[self.domain]))
                self.refresh()
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
        value_label: str,
        cmin: float,
        cmax: float,
        domain: ModelDomain,
        custom_data: pd.DataFrame
        ) -> None:
        # Colors
        self.data[0]["marker"].update(dict(color=values, cmin=cmin, cmax=cmax))

        # ScatterMap
        self.data[0].update(dict(
            lat=latitude,
            lon=longitude,
            customdata=custom_data,
            hovertemplate=(
                f"<br>{value_label}: "
                "%{marker.color:.2f}<br>"
                "NWM Feature ID: %{customdata[0]}<br>"
                "USGS Site Code: %{customdata[1]}<br>"
                "Start Date: %{customdata[2]}<br>"
                "End Date: %{customdata[3]}<br>"
                "Samples: %{customdata[4]}<br>"
                "Longitude: %{lon}<br>"
                "Latitude: %{lat}"
        )))

        # Title
        self.data[0]["marker"]["colorbar"]["title"].update(dict(text=value_label))

        # Boundaries
        self.lat_min = np.min(latitude)
        self.lat_max = np.max(latitude)
        self.lon_min = np.min(longitude)
        self.lon_max = np.max(longitude)

        # Domain change
        if domain != self.domain:
            self.layout["map"]["center"].update(DEFAULT_CENTER[domain])
            self.layout["map"].update(dict(zoom=DEFAULT_ZOOM[domain]))
            self.domain = domain
    
    def refresh(self) -> None:
        self.figure.update(dict(data=self.data, layout=self.layout))
        self.pane.object = self.figure
    
    def servable(self) -> pn.Card:
        return pn.Card(
            self.pane,
            collapsible=False,
            hide_header=True
        )
    
    @property
    def relayout_data(self) -> dict:
        return self.pane.param.relayout_data
