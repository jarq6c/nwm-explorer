"""Generate and serve exploratory evaluation dashboard."""
from pathlib import Path
import inspect

import panel as pn
from panel.template import BootstrapTemplate

from nwm_explorer.logging.logger import get_logger
from nwm_explorer.evaluation.compute import EvaluationRegistry

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
        
        print(evaluation_registry)
    
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
