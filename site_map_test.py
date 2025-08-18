"""
An interactive mapping interface.
"""
from typing import TypedDict, Protocol
from dataclasses import dataclass

import polars as pl

import numpy.typing as npt
import panel as pn
from panel.viewable import Viewer
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
import colorcet as cc

class ColorString(Protocol):
    "Plotly compatible color string."

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

def generate_scatter_map(
        longitude: npt.ArrayLike,
        latitude: npt.ArrayLike,
        color: ColorString | npt.ArrayLike | None = None,
        colorbar_title: str | None = None,
        colorbar_limits: tuple[float, float] | None = None,
        custom_data: pl.DataFrame | None = None
        ) -> go.Scattermap:
    # Set default color
    if color is None:
        color = "black"

    # Setup markers
    if isinstance(color, str):
        markers = dict(
            color=color,
            size=15
        )
    else:
        markers = dict(
            color=color,
            colorbar=dict(title=dict(side="right", text=colorbar_title)),
            size=15,
            colorscale=cc.gouldian
        )
        if colorbar_limits is not None:
            markers.update(dict(
                cmin=colorbar_limits[0],
                cmax=colorbar_limits[1]
        ))

    # Instantiate map
    return go.Scattermap(
        lon=longitude,
        lat=latitude,
        showlegend=False,
        name="",
        mode="markers",
        marker=markers,
        customdata=custom_data
    )

class SiteMap(Viewer):
    """
    Shows a plotly map in a card interface.

    Parameters
    ----------
    domains: dict[str, MapFocus]
        Dict of MapFocus keyed to domain labels.
    default_domain: str, optional
        Sets default domain returned to when resetting map.
    """
    def __init__(
            self,
            data: list[BaseTraceType],
            domains: dict[str, MapFocus],
            default_domain: str | None = None,
            **params
        ) -> None:
        super().__init__(**params)

        # Data
        self.data: list[BaseTraceType] = data

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
            "data": self.data,
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

    def refresh(self) -> None:
        """
        Send current state of data and layout to frontend.
        """
        self.pane.object = {
            "data": self.data,
            "layout": self.layout
        }

    def __panel__(self) -> pn.Card:
        return pn.Card(
            self.pane,
            collapsible=False,
            hide_header=True
        )

def main():
    # Domain defaults
    domains = {
        "All": MapFocus(center=Coordinates(lat=38.83348, lon=-100.97612), zoom=2),
        "Alaska": MapFocus(center=Coordinates(lat=60.84683, lon=-149.05659), zoom=5),
        "CONUS": MapFocus(center=Coordinates(lat=38.83348, lon=-93.97612), zoom=3),
        "Hawaii": MapFocus(center=Coordinates(lat=21.24988, lon=-157.59606), zoom=6),
        "Puerto Rico": MapFocus(center=Coordinates(lat=18.21807, lon=-66.32802), zoom=8)
    }

    # Data
    df = pl.DataFrame({
        "USGS site code": ["02146470"],
        "Latitude": [35.16444444],
        "Longitude": [-80.8530556],
        "Nash-Sutcliffe efficiency": [0.55]
    })

    # Scatter map
    scatter_map = generate_scatter_map(
        df["Longitude"],
        df["Latitude"],
        df["Nash-Sutcliffe efficiency"],
        "Nash-Sutcliffe efficiency",
        (-1.0, 1.0),
        df[["USGS site code"]]
    )

    # Setup map
    site_map = SiteMap([scatter_map], domains)

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

    # Serve the dashboard
    pn.serve(pn.Column(site_map, domain_selector))

if __name__ == "__main__":
    main()
