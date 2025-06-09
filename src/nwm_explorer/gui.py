"""Generate and serve exploratory applications."""
from pathlib import Path
import panel as pn
from panel.template import BootstrapTemplate
import numpy as np

from nwm_explorer.readers import RoutelinkReader, MetricsReader
from nwm_explorer.plotters import SiteMapPlotter

def generate_dashboard(
        root: Path,
        title: str
        ) -> BootstrapTemplate:
    # Data
    routelink_reader = RoutelinkReader(root)
    metrics_reader = MetricsReader(root)
    domain_list = routelink_reader.domains
    metric_labels = metrics_reader.metrics
    configurations = metrics_reader.configurations
    periods = metrics_reader.periods

    # Plotters
    site_map = SiteMapPlotter()

    # Widgets
    domain_selector = pn.widgets.Select(
        name="Select Domain",
        options=domain_list,
        value=domain_list[0]
    )
    configuration_selector = pn.widgets.Select(
        name="Select Model Configuration",
        options=configurations,
        value=configurations[0]
    )
    period_selector = pn.widgets.Select(
        name="Select Evaluation Period",
        options=periods,
        value=periods[0]
    )
    metric_selector = pn.widgets.Select(
        name="Select Metric",
        options=metric_labels,
        value=metric_labels[0]
    )

    # Panes
    rng = np.random.default_rng(seed=2025)
    geometry = routelink_reader.geometry(domain_list[0])
    site_map.update_points(
        domain=domain_list[0],
        values=rng.uniform(-1.0, 1.0, len(geometry)),
        metric_label=metric_selector.value,
        routelink_reader=routelink_reader
    )
    site_map_pane = pn.pane.Plotly(site_map.figure)

    # Callbacks
    def domain_callbacks(domain):
        geometry = routelink_reader.geometry(domain)
        # Update map
        lat, lon, zoom = site_map.update_points(
            domain=domain,
            values=rng.uniform(-1.0, 1.0, len(geometry)),
            metric_label=metric_selector.value,
            routelink_reader=routelink_reader
        )
        site_map_pane.relayout_data.update({
            "map.center": {"lat": lat, "lon": lon}})
        site_map_pane.relayout_data.update({"map.zoom": zoom})
        site_map_pane.object = site_map.figure
    pn.bind(domain_callbacks, domain_selector, watch=True)

    def metric_callbacks(metric_label):
        # Update map
        site_map.update_colors(
            values=rng.uniform(-1.0, 1.0, len(geometry)),
            metric_label=metric_label,
            relayout_data=site_map_pane.relayout_data
        )
        site_map_pane.object = site_map.figure
    pn.bind(metric_callbacks, metric_selector, watch=True)

    # Layout
    template = BootstrapTemplate(title=title)
    template.sidebar.append(domain_selector)
    template.sidebar.append(configuration_selector)
    template.sidebar.append(period_selector)
    template.sidebar.append(metric_selector)
    template.main.append(site_map_pane)

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
