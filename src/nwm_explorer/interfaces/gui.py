"""Generate and serve exploratory evaluation dashboard."""
from pathlib import Path
import panel as pn
from panel.template import BootstrapTemplate

class Dashboard:
    """Build a dashboard for exploring National Water Model output."""
    def __init__(self, root: Path, title: str):
        self.template = BootstrapTemplate(title=title)
    
    def servable(self) -> BootstrapTemplate:
        return self.template

def generate_dashboard(
        root: Path,
        title: str
        ) -> BootstrapTemplate:
    return Dashboard(root, title).servable()

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
