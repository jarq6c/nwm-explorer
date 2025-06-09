"""Generate and serve exploratory applications."""
from pathlib import Path
import panel as pn
from panel.template import BootstrapTemplate

from nwm_explorer.readers import RoutelinkReader

def generate_dashboard(
        root: Path,
        title: str
        ) -> BootstrapTemplate:
    # Data
    rr = RoutelinkReader(root)
    domain_list = rr.domains
    initial_domain = domain_list[0]
    initial_site_list = rr.site_list(initial_domain)

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
    readout = pn.pane.Markdown(f"# Number of sites: {len(initial_site_list)}")

    # Layout
    template = BootstrapTemplate(title=title)
    template.sidebar.append(domain_selector)
    template.sidebar.append(usgs_site_code_selector)
    template.main.append(readout)

    # Callbacks
    def update_readout(domain):
        site_list = rr.site_list(domain)
        readout.object = f"# Number of sites: {len(site_list)}"
        usgs_site_code_selector.options = site_list
    pn.bind(update_readout, domain_selector, watch=True)

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
    endpoints = {
        "nwm-explorer": generate_dashboard_closure(root, title)
    }
    pn.serve(endpoints)
