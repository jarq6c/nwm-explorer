"""Basic functions to build and launch dashboards."""
from pathlib import Path

import panel as pn

from .dashboard import Dashboard

def generate_dashboard(
        root: Path,
        title: str
        ) -> Dashboard:
    """Instantiate and return a dashboard."""
    return Dashboard(root, title)

def generate_dashboard_closure(
        root: Path,
        title: str
        ) -> Dashboard:
    """Build and return a closure function that generates new dashboards."""
    def closure():
        return generate_dashboard(root, title)
    return closure

def serve_dashboards(
        root: Path,
        title: str
        ) -> None:
    """Serve dashboards."""
    # Slugify title
    slug = title.lower().replace(" ", "-")

    # Serve
    endpoints = {
        slug: generate_dashboard_closure(root, title)
    }
    pn.serve(endpoints)
