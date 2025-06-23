"""Plot managers."""
from dataclasses import dataclass
from typing import Any
import plotly.graph_objects as go
import colorcet as cc
import numpy as np
import numpy.typing as npt
import polars as pl

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
            values: npt.ArrayLike,
            latitude: npt.ArrayLike,
            longitude: npt.ArrayLike,
            metric_label: str,
            zoom: int,
            custom_data: pl.DataFrame
            ) -> None:
        self.scatter.update(dict(
            lat=latitude,
            lon=longitude,
            customdata=custom_data,
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
                "lat": np.mean(latitude),
                "lon": np.mean(longitude)
                },
            zoom=zoom
        ))

    def update_colors(
            self,
            values: npt.ArrayLike,
            label: str,
            relayout_data: dict[str, Any]
            ) -> None:
        self.scatter.update(dict(
            hovertemplate=(
                SITE_MAP_HOVER_TEMPLATE +
                f"<br>{label}: " +
                "%{marker.color:.2f}"
            )
        ))
        self.scatter["marker"].update(dict(
            color=values,
            colorbar=dict(
                title=dict(
                    text=label,
                    side="right"
                    )
                ),
                cmin=-1.0,
                cmax=1.0
        ))
        if "map.center" in relayout_data:
            self.layout["map"]["center"].update({
                "lat": relayout_data["map.center"]["lat"],
                "lon": relayout_data["map.center"]["lon"]
            })
            self.layout["map"].update({
                "zoom": relayout_data["map.zoom"]
            })
