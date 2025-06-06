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

    # Panes
    readout = pn.pane.Markdown("# Number of sites: ")

    # Layout
    template = BootstrapTemplate(title="Title")
    template.sidebar.append(domain_selector)
    template.main.append(readout)

    # Callbacks
    routelinks = read_routelinks(root)
    domain_selector.options = list(routelinks.keys())
    def update_readout(domain):
        number_of_sites = routelinks[domain].select("usgs_site_code").count().collect().item(0, 0)
        readout.object = f"# Number of sites: {number_of_sites}"
    pn.bind(update_readout, domain_selector, watch=True)

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
