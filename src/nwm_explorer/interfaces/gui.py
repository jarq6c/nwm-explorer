"""Generate and serve exploratory evaluation dashboard."""
from pathlib import Path
import inspect

import polars as pl
import panel as pn
from panel.template import BootstrapTemplate

from nwm_explorer.logging.logger import get_logger
from nwm_explorer.evaluation.compute import EvaluationRegistry
from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration

PRETTY_DOMAINS: dict[ModelDomain, str] = {
    ModelDomain.alaska: "Alaska",
    ModelDomain.conus: "CONUS",
    ModelDomain.hawaii: "Hawaii",
    ModelDomain.puertorico: "Puerto Rico"
}
"""Mapping from ModelDomain to pretty strings."""

DOMAIN_LOOKUP: dict[str, ModelDomain] = {v: k for k, v in PRETTY_DOMAINS.items()}
"""Reverse look-up from pretty strings to ModelDomain."""

class Dashboard:
    """Build a dashboard for exploring National Water Model output."""
    def __init__(self, root: Path, title: str):
        # Get logger
        name = __loader__.name + "." + inspect.currentframe().f_code.co_name
        logger = get_logger(name)

        # Setup template
        self.template = BootstrapTemplate(title=title)

        # Setup registry
        registry_file = root / "evaluation_registry.json"
        if registry_file.exists():
            logger.info(f"Reading {registry_file}")
            with registry_file.open("r") as fo:
                evaluation_registry = EvaluationRegistry.model_validate_json(fo.read())
        else:
            logger.info(f"No registry found at {registry_file}")
            self.template.main.append(pn.pane.Markdown("# Registry not found. Have you run an evaluation?"))
            return
        
        # Scan evaluation data
        self.data: dict[str, dict[ModelDomain, dict[ModelConfiguration, pl.LazyFrame]]] = {}
        for label, evaluation_spec in evaluation_registry.evaluations.items():
            self.data[label] = {}
            for domain, files in evaluation_spec.files.items():
                self.data[label][domain] = {}
                for configuration, ifile in files.items():
                    logger.info(f"Scanning {ifile}")
                    self.data[label][domain][configuration] = pl.scan_parquet(ifile)

        # Setup widgets
        self.evaluation_filter = pn.widgets.Select(
            name="Evaluation",
            options=list(self.data.keys())
        )
        self.domain_filter = pn.widgets.Select(
            name="Domain",
            options=[]
        )
        self.configuration_filter = pn.widgets.Select(
            name="Configuration",
            options=[]
        )

        # Update domain
        def update_domains(eval) -> None:
            self.domain_filter.options = [PRETTY_DOMAINS[v] for v in self.data[eval].values()]
        pn.bind(update_domains, self.evaluation_filter, watch=True)

        # Layout
        filters = pn.Card(pn.Column(
            self.evaluation_filter,
            self.domain_filter,
            self.configuration_filter
            ),
            title="Filters",
            collapsible=False
        )
        self.template.main.append(filters)
    
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
