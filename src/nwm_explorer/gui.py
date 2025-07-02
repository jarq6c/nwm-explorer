"""Generate and serve exploratory applications."""
from pathlib import Path
import panel as pn
from panel.template import BootstrapTemplate
from nwm_explorer.filters import FilteringWidgets, FilterState
from nwm_explorer.status import StatusFeed

class Dashboard:
    """Build a dashboard for exploring National Water Model output."""
    def __init__(self, root: Path, title: str):
        # Filters
        self.filters = FilteringWidgets()
        
        # Status
        self.status = StatusFeed()
        self.status.success("Initialized")

        # Layout dashboard
        layout = pn.Column(
            self.filters.servable(),
            self.status.servable()
            )
        self.template = BootstrapTemplate(title=title)
        self.template.main.append(layout)

    @property
    def state(self) -> FilterState:
        """Current dashboard state."""
        return self.filters.state
    
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
