"""Generate and serve exploratory applications."""
from pathlib import Path
from typing import Any
import panel as pn
from panel.template import BootstrapTemplate
import plotly.graph_objects as go
import numpy as np
import numpy.typing as npt
import pandas as pd
import geopandas as gpd
import colorcet as cc

from nwm_explorer.readers import RoutelinkReader
from nwm_explorer.mappings import Domain

DEFAULT_ZOOM: dict[Domain, int] = {
    Domain.alaska: 5,
    Domain.conus: 3,
    Domain.hawaii: 6,
    Domain.puertorico: 8
}
"""Default map zoom for each domain."""

ROUTELINK_CUSTOM_DATA_COLUMNS: list[str] = [
    "usgs_site_code",
    "nwm_feature_id"
]
"""Custom data column labels for use with Plotly hover tooltips."""

ROUTELINK_HOVER_TEMPLATE: str = (
    "USGS Site Code: %{customdata[0]}<br>"
    "NWM Feature ID: %{customdata[1]}<br>"
    "Longitude: %{lon}<br>"
    "Latitude: %{lat}"
)
"""Plotly compatible hover template for site maps."""

def generate_map(
        geometry: gpd.GeoSeries,
        statistics: npt.ArrayLike,
        default_zoom: int = 2,
        customdata: pd.DataFrame | None = None,
        hovertemplate: str | None = None,
        stat_label: str = "Statistic",
        stat_lims: tuple[float, float] | None = None
        ) -> tuple[go.Scattermap, go.Layout]:
    """
    Generate a map of points.

    Parameters
    ----------
    geodata: geopandas.GeoSeries
        GeoSeries of POINT geometry.
    """
    # Site map
    scatter_map = go.Scattermap(
        showlegend=False,
        name="",
        lat=geometry.y,
        lon=geometry.x,
        mode="markers",
        marker=dict(
            size=15,
            color=statistics,
            colorscale=cc.gouldian,
            colorbar=dict(title=dict(
                text=stat_label, side="right")),
            cmin=stat_lims[0] if stat_lims else None,
            cmax=stat_lims[1] if stat_lims else None
            ),
        customdata=customdata,
        hovertemplate=(
            hovertemplate +
            f"<br>{stat_label}: " +
            "%{marker.color:.2f}"
        )
    )

    # Layout
    layout = go.Layout(
        showlegend=False,
        height=720,
        width=1280,
        margin=dict(l=0, r=0, t=50, b=0),
        map=dict(
            style="satellite-streets",
            center={
                "lat": geometry.y.mean(),
                "lon": geometry.x.mean()
                },
            zoom=default_zoom
        ),
        clickmode="event",
        modebar=dict(
            remove=["lasso", "select"]
        ),
        dragmode="zoom"
    )
    return scatter_map, layout

def generate_dashboard(
        root: Path,
        title: str
        ) -> BootstrapTemplate:
    # Data
    rr = RoutelinkReader(root)
    domain_list = rr.domains
    geometry = rr.geometry(domain_list[0])
    metric_labels = ["Mean relative bias", "Nash-Sutcliffe Efficiency"]

    # Widgets
    domain_selector = pn.widgets.Select(
        name="Select Domain",
        options=domain_list,
        value=domain_list[0]
    )
    metric_selector = pn.widgets.Select(
        name="Select Metric",
        options=metric_labels,
        value=metric_labels[0]
    )

    # Panes
    rng = np.random.default_rng(seed=2025)
    scatter_map, layout = generate_map(
        geometry=geometry,
        statistics=rng.uniform(-1.0, 1.0, len(geometry)),
        default_zoom=DEFAULT_ZOOM[domain_list[0]],
        customdata=rr.select_columns(
                domain_list[0],
                ROUTELINK_CUSTOM_DATA_COLUMNS
            ),
        hovertemplate=ROUTELINK_HOVER_TEMPLATE,
        stat_label=metric_labels[0],
        stat_lims=(-1.0, 1.0)
        )
    figure_data = {"data": [scatter_map], "layout": layout}
    site_map = pn.pane.Plotly(figure_data)

    # Callbacks
    def domain_callbacks(domain):
        # Update map
        geometry = rr.geometry(domain)
        scatter_map.update(dict(
            lat=geometry.y,
            lon=geometry.x,
            customdata=rr.select_columns(
                    domain,
                    ROUTELINK_CUSTOM_DATA_COLUMNS
                ),
            hovertemplate=(
                ROUTELINK_HOVER_TEMPLATE +
                f"<br>{metric_selector.value}: " +
                "%{marker.color:.2f}"
            )
        ))
        scatter_map["marker"].update(dict(
            color=rng.uniform(-1.0, 1.0, len(geometry)),
            colorbar=dict(
                title=dict(
                    text=metric_selector.value,
                    side="right"
                    )
                ),
                # cmin=cmin,
                # cmax=cmax
        ))
        layout["map"].update(dict(
            center={
                "lat": geometry.y.mean(),
                "lon": geometry.x.mean()
                },
            zoom=DEFAULT_ZOOM[domain]
        ))
        site_map.object = figure_data
    pn.bind(domain_callbacks, domain_selector, watch=True)

    # Layout
    template = BootstrapTemplate(title=title)
    template.sidebar.append(domain_selector)
    template.sidebar.append(metric_selector)
    template.main.append(site_map)

    return template

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
