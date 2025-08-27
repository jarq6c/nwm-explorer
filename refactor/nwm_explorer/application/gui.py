"""
Web-based application used to vizualize and explore hydrological model
evaluations, primarily for the National Water Model.
"""
import inspect
from pathlib import Path
from typing import Any
from collections.abc import Callable

import panel as pn
from panel.viewable import Viewer
from panel.template import BootstrapTemplate

from nwm_explorer.logging.loggers import get_logger

class Dashboard(Viewer):
    """Build a dashboard for exploring National Water Model output."""
    def __init__(self, root: Path, title: str, **params: dict[str, Any]):
        # Apply parameters
        super().__init__(**params)

        # Get logger
        name = __loader__.name + "." + inspect.currentframe().f_code.co_name
        logger = get_logger(name)

        # Setup template
        logger.info("Build template")
        self.template = BootstrapTemplate(
            title=title,
            collapsed_sidebar=True
        )
    
    def __panel__(self) -> BootstrapTemplate:
        return self.template

def generate_dashboard(
        root: Path,
        title: str
        ) -> Dashboard:
    """
    Returns a new servable dashboard to requesting endpoint.
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)
    logger.info("Build dashboard")
    return Dashboard(root, title)

def generate_dashboard_closure(
        root: Path,
        title: str
        ) -> Callable[[], Dashboard]:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)
    logger.info("Generating dashboard closure")
    def closure() -> Dashboard:
        return generate_dashboard(root, title)
    return closure

def serve_dashboard(
        root: Path,
        title: str
        ) -> None:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Slugify title
    slug = title.lower().replace(" ", "-")

    # Serve
    endpoints = {
        slug: generate_dashboard_closure(root, title)
    }
    logger.info("Serving dashboards")
    pn.serve(endpoints)
