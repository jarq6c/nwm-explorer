"""Generate and serve exploratory evaluation dashboard."""
from pathlib import Path
import panel as pn
from panel.template import BootstrapTemplate
from nwm_explorer.filters import FilteringWidgets, FilterState
from nwm_explorer.status import StatusFeed
from nwm_explorer.site_map import SiteMap
from nwm_explorer.mappings_new import CallbackType, Domain

import polars as pl

class Dashboard:
    """Build a dashboard for exploring National Water Model output."""
    def __init__(self, root: Path, title: str):
        # Filters
        self.filters = FilteringWidgets()
        
        # Status
        self.status = StatusFeed()
        self.status.success("Initialized")

        # Site map
        self.site_maps: dict[Domain, SiteMap] = {}
        for idx, d in enumerate(list(Domain)):
            self.site_maps[d] = SiteMap(
                latitude=[33.20563855835994+5*idx],
                longitude=[-87.54662400303468],
                custom_data=pl.DataFrame({"A": ["1234"], "B": [4325]}),
                values=[0.567],
                value_label=f"KGE {idx}",
                value_limits=(0.0, 1.0),
                custom_labels=[f"First {idx}", f"Second {idx}"],
                default_zoom=15
            ).servable()
        self.site_map = pn.pane.Placeholder(self.site_maps[self.filters.state.domain])

        # Site map updates
        def update_site_map(event, callback_type: CallbackType) -> None:
            if event is None:
                return
            
            # Change domain
            if callback_type == CallbackType.domain:
                self.site_map.object = self.site_maps[self.filters.state.domain]
        self.filters.register_callback(update_site_map)

        # Layout dashboard
        layout = pn.Row(
            pn.Column(
                self.filters.servable(),
                self.status.servable()
            ),
            self.site_map
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
