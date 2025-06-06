"""Generate and serve exploratory applications."""
from pathlib import Path
import panel as pn
from panel.template import BootstrapTemplate

from nwm_explorer.readers import read_routelinks

def generate_dashboard(directory: Path = Path("data")) -> BootstrapTemplate:
    routelinks = read_routelinks(directory)
    template = BootstrapTemplate(title="Title")
    return template

def serve_dashboards(directory: Path = Path("data")):
    pn.serve(generate_dashboard)
