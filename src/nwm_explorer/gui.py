"""Generate and serve exploratory applications."""
from pathlib import Path
from typing import Any
import panel as pn
from panel.template import BootstrapTemplate
import plotly.graph_objects as go
import geopandas as gpd

from nwm_explorer.readers import RoutelinkReader

def generate_map(geometry: gpd.GeoSeries) -> dict[str, Any]:
    """
    Generate a map of points.

    Parameters
    ----------
    geodata: geopandas.GeoSeries
        GeoSeries of POINT geometry.
    """
    # Site highlighter
    data = []
    data.append(go.Scattermap(
        showlegend=False,
        name="",
        lat=geometry.y[:1],
        lon=geometry.x[:1],
        mode="markers",
        marker=dict(
            size=25,
            color="magenta"
            ),
        selected=dict(
            marker=dict(
                color="magenta"
            )
        ),
    ))

    # Site map
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
        # customdata=geodata[[
        #     "LEFT FEATURE NAME",
        #     "LEFT FEATURE DESCRIPTION",
        #     "RIGHT FEATURE NAME"
        #     ]],
        # hovertemplate=
        # "LEFT FEATURE DESCRIPTION: %{customdata[1]}<br>"
        # "LEFT FEATURE NAME: %{customdata[0]}<br>"
        # "RIGHT FEATURE NAME: %{customdata[2]}<br>"
        # "LONGITUDE: %{lon}<br>"
        # "LATITUDE: %{lat}<br>"
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
            zoom=2
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
    initial_site_list = rr.site_list(initial_domain)
    geometry = rr.geometry(initial_domain)

    # Widgets
    domain_selector = pn.widgets.Select(
        name="Select Domain",
        options=domain_list,
        value=initial_domain
    )
    usgs_site_code_selector = pn.widgets.AutocompleteInput(
        name="USGS Site Code",
        options=initial_site_list,
        search_strategy="includes",
        placeholder=f"Select USGS Site Code"
    )

    # Panes
    site_map = pn.pane.Plotly(generate_map(geometry))
    # readout = pn.pane.Markdown(f"# Number of sites: {len(initial_site_list)}")

    # # Callbacks
    # def update_readout(domain):
    #     site_list = rr.site_list(domain)
    #     readout.object = f"# Number of sites: {len(site_list)}"
    #     usgs_site_code_selector.options = site_list
    # pn.bind(update_readout, domain_selector, watch=True)

    # Layout
    template = BootstrapTemplate(title=title)
    template.sidebar.append(domain_selector)
    template.sidebar.append(usgs_site_code_selector)
    # template.main.append(readout)
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
