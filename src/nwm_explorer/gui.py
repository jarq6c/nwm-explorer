"""Generate and serve exploratory applications."""
from pathlib import Path
from typing import Any
import panel as pn
from panel.template import BootstrapTemplate
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd

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
"""Custom data columns for use with Plotly hover tooltips."""

ROUTELINK_HOVER_TEMPLATE: str = (
    "USGS Site Code: %{customdata[0]}<br>"
    "NWM Feature ID: %{customdata[1]}<br>"
    "Longitude: %{lon}<br>"
    "Latitude: %{lat}<br>"
)
"""Plotly compatible hover template for site maps."""

def generate_map(
        geometry: gpd.GeoSeries,
        default_zoom: int = 2,
        customdata: pd.DataFrame | None = None,
        hovertemplate: str | None = None
        ) -> dict[str, Any]:
    """
    Generate a map of points.

    Parameters
    ----------
    geodata: geopandas.GeoSeries
        GeoSeries of POINT geometry.
    """
    # Site map
    data = []
    data.append(go.Scattermap(
        showlegend=False,
        name="",
        lat=geometry.y,
        lon=geometry.x,
        mode="markers",
        marker=dict(
            size=15,
            color="cyan"
            ),
        selected=dict(
            marker=dict(
                color="cyan"
            )
        ),
        customdata=customdata,
        hovertemplate=hovertemplate
    ))

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
    return {"data": data, "layout": layout}

def generate_dashboard(
        root: Path,
        title: str
        ) -> BootstrapTemplate:
    # Data
    rr = RoutelinkReader(root)
    domain_list = rr.domains
    initial_domain = domain_list[0]
    geometry = rr.geometry(initial_domain)

    # Widgets
    domain_selector = pn.widgets.Select(
        name="Select Domain",
        options=domain_list,
        value=initial_domain
    )

    # Panes
    site_map = pn.pane.Plotly(generate_map(
        geometry=geometry,
        default_zoom=DEFAULT_ZOOM[initial_domain],
        customdata=rr.select_columns(
                initial_domain,
                ROUTELINK_CUSTOM_DATA_COLUMNS
            ),
        hovertemplate=ROUTELINK_HOVER_TEMPLATE
        ))

    # Callbacks
    def domain_callbacks(domain):
        # Update map
        site_map.object = generate_map(
            geometry=rr.geometry(domain),
            default_zoom=DEFAULT_ZOOM[domain],
            customdata=rr.select_columns(
                    domain,
                    ROUTELINK_CUSTOM_DATA_COLUMNS
                ),
            hovertemplate=ROUTELINK_HOVER_TEMPLATE
            )
    pn.bind(domain_callbacks, domain_selector, watch=True)

    # Layout
    template = BootstrapTemplate(title=title)
    template.sidebar.append(domain_selector)
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
