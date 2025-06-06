"""Generate and serve exploratory applications."""
from pathlib import Path
import panel as pn
from panel.template import BootstrapTemplate

from nwm_explorer.readers import read_routelinks

def generate_dashboard(root: Path = Path("data")) -> BootstrapTemplate:
    # Widgets
    domain_selector = pn.widgets.Select(
        name="Select Domain",
        options=[]
    )

    # Layout
    template = BootstrapTemplate(title="Title")
    template.sidebar.append(domain_selector)

    # Callbacks
    routelinks = read_routelinks(root)
    domain_selector.options = list(routelinks.keys())
    
    return template

def generate_dashboard_closure(root: Path = Path("data")) -> BootstrapTemplate:
    def closure():
        return generate_dashboard(root)
    return closure

def serve_dashboards(root: Path = Path("data")):
    endpoints = {
        "nwm-explorer": generate_dashboard_closure(root)
    }
    pn.serve(endpoints)
