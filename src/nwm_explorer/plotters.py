"""Plot managers."""
from dataclasses import dataclass
import plotly.graph_objects as go
import colorcet as cc
import numpy.typing as npt

from nwm_explorer.mappings import DEFAULT_ZOOM, Domain
from nwm_explorer.readers import RoutelinkReader

SITE_MAP_CUSTOM_DATA_COLUMNS: list[str] = [
    "usgs_site_code",
    "nwm_feature_id"
]
"""Custom data column labels for use with Plotly hover tooltips."""

SITE_MAP_HOVER_TEMPLATE: str = (
    "USGS Site Code: %{customdata[0]}<br>"
    "NWM Feature ID: %{customdata[1]}<br>"
    "Longitude: %{lon}<br>"
    "Latitude: %{lat}"
)
"""Plotly compatible hover template for site maps."""

@dataclass
class SiteMapPlotter:
    scatter: go.Scattermap | None = None
    layout: go.Layout | None = None

    def __post_init__(self) -> None:
        if self.scatter is None:
            self.scatter = go.Scattermap(
                showlegend=False,
                name="",
                mode="markers",
                marker=dict(
                    size=15,
                    colorscale=cc.gouldian
                    )
                )
        if self.layout is None:
            self.layout = go.Layout(
                showlegend=False,
                height=720,
                width=1280,
                margin=dict(l=0, r=0, t=50, b=0),
                map=dict(
                    style="satellite-streets",
                ),
                clickmode="event",
                modebar=dict(
                    remove=["lasso", "select"]
                ),
                dragmode="zoom"
            )

        self.figure = {
            "data": [self.scatter],
            "layout": self.layout
        }

    def update_points(
            self,
            domain: Domain,
            geometry: npt.ArrayLike,
            values: npt.ArrayLike,
            metric_label: str,
            routelink_reader: RoutelinkReader
            ) -> None:
        self.scatter.update(dict(
            lat=geometry.y,
            lon=geometry.x,
            customdata=routelink_reader.select_columns(
                domain,
                SITE_MAP_CUSTOM_DATA_COLUMNS
            ),
            hovertemplate=(
                SITE_MAP_HOVER_TEMPLATE +
                f"<br>{metric_label}: " +
                "%{marker.color:.2f}"
            )
        ))
        self.scatter["marker"].update(dict(
            color=values,
            colorbar=dict(
                title=dict(
                    text=metric_label,
                    side="right"
                    )
                ),
                cmin=-1.0,
                cmax=1.0
        ))
        self.layout["map"].update(dict(
            center={
                "lat": geometry.y.mean(),
                "lon": geometry.x.mean()
                },
            zoom=DEFAULT_ZOOM[domain]
        ))
