"""Generate and serve exploratory applications."""
from pathlib import Path
import panel as pn
from panel.template import BootstrapTemplate

from nwm_explorer.readers import read_routelinks

def generate_dashboard(root: Path = Path("data")) -> BootstrapTemplate:
    # Data
    routelinks = read_routelinks(root)
    domain_list = list(routelinks.keys())
    initial_domain = domain_list[0]
    initial_site_list = routelinks[initial_domain].select("usgs_site_code").collect()["usgs_site_code"].to_list()

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
    n = routelinks[initial_domain].select("usgs_site_code").count().collect().item(0, 0)
    readout = pn.pane.Markdown(f"# Number of sites: {n}")

    # Layout
    template = BootstrapTemplate(title="National Water Model Explorer")
    template.sidebar.append(domain_selector)
    template.sidebar.append(usgs_site_code_selector)
    template.main.append(readout)

    # Callbacks
    def update_readout(domain):
        number_of_sites = routelinks[domain].select("usgs_site_code").count().collect().item(0, 0)
        readout.object = f"# Number of sites: {number_of_sites}"
        site_list = routelinks[domain].select("usgs_site_code").collect()["usgs_site_code"].to_list()
        usgs_site_code_selector.options = site_list
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
