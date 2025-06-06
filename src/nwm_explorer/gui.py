"""Generate and serve exploratory applications."""
from pathlib import Path
import panel as pn
from panel.template import BootstrapTemplate

def start_dashboard(directory: Path = Path("data")):
    pn.serve(BootstrapTemplate(title="Title").servable())
