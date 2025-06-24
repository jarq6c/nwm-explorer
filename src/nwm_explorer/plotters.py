"""Plot managers."""
from dataclasses import dataclass
from typing import Any, TypedDict
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

class FigurePatch(TypedDict):
    """
    A plotly figure patch.

    Keys
    ----
    data: list[go.Trace]
        A list of plotly traces.
    layout: go.Layout
        Plotly layout.
    """
    data: list[go.Trace]
    layout: go.Layout

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
                height=600,
                width=900,
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

    @property
    def figure(self) -> FigurePatch:
        return {
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

def generate_histogram(
        x: npt.ArrayLike,
        xmin: float,
        xmax: float,
        bin_width: float
    ) -> tuple[npt.NDArray[np.float64], list[str]]:
    nbins = int((xmax - xmin) / bin_width)
    counts, bin_edges = np.histogram(
        a=x,
        bins=nbins,
        range=(xmin, xmax),
        density=False
    )
    bin_centers = np.linspace(xmin + bin_width / 2, xmax - bin_width / 2, nbins)

    below_minimum = x[x < xmin].size
    counts = np.insert(counts, 0, below_minimum)
    bin_centers = np.insert(bin_centers, 0, xmin - bin_width / 2)

    above_maximum = x[x > xmax].size
    counts = np.append(counts, above_maximum)
    bin_centers = np.append(bin_centers, xmax + bin_width / 2)

    probabilities = counts / np.sum(counts)

    xlabels = [f"< {xmin:.1f}"]
    for i in range(len(bin_edges) - 1):
        left = f"{bin_edges[i]:.1f}"
        right = f"{bin_edges[i+1]:.1f}"
        xlabels.append(left + " to " + right)
    xlabels.append(f"> {xmax:.1f}")

    return xlabels, probabilities

@dataclass
class HistogramPlotter:
    histogram: go.Bar | None = None
    layout: go.Layout | None = None

    def __post_init__(self) -> None:
        # config={"displayModeBar": False}
        if self.histogram is None:
            self.histogram = go.Bar(
                showlegend=False,
                name=""
                )
        if self.layout is None:
            self.layout = go.Layout(
                showlegend=False,
                height=250,
                width=300,
                margin=dict(l=0, r=0, t=0, b=0),
                yaxis=dict(
                    title=dict(
                            text="Relative Frequency (95% Confidence)"
                        )
            ))

    @property
    def figure(self) -> FigurePatch:
        return {
            "data": [self.histogram],
            "layout": self.layout
        }

    def update_bars(
            self,
            values: npt.ArrayLike,
            values_lower: npt.ArrayLike,
            values_upper: npt.ArrayLike,
            vmin: float,
            vmax: float,
            bin_width: float,
            xtitle: str
            ) -> None:
        labels, vprobs = generate_histogram(
            values, vmin, vmax, bin_width
        )
        _, vprobs_lower = generate_histogram(
            values_lower, vmin, vmax, bin_width
        )
        _, vprobs_upper = generate_histogram(
            values_upper, vmin, vmax, bin_width
        )
        estimates = np.vstack((vprobs, vprobs_lower, vprobs_upper))
        point = np.median(estimates, axis=0)
        upper = np.max(estimates, axis=0) - point
        lower = point - np.min(estimates, axis=0)

        # Update
        self.histogram.update(dict(
            x=labels,
            y=point,
            error_y=dict(
                type="data",
                symmetric=False,
                array=upper,
                arrayminus=lower
            )))
        self.layout.update(dict(
            xaxis=dict(
                title=dict(
                    text=xtitle
            ))))
