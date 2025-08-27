"""
Web-based application used to vizualize and explore hydrological model
evaluations, primarily for the National Water Model.
"""
import inspect
from pathlib import Path
from collections.abc import Callable

import panel as pn

from nwm_explorer.logging.loggers import get_logger
from nwm_explorer.layouts.dashboard import Dashboard, generate_dashboard

def generate_dashboard_closure(
        root: Path,
        title: str
        ) -> Callable[[], Dashboard]:
    """
    Generates a partial function that returns Dashboards with root and title
    applied. This is required to generate dashboards with different parameters
    at run time.

    Parameters
    ----------
    root: Path
        Path to root data directory used by dashboards.
    title: str
        Title that appears in the dashboard header.
    
    Returns
    -------
    Callable[[], Dashboard]
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)
    logger.info("Generating dashboard closure")
    def closure() -> Dashboard:
        return generate_dashboard(root, title)
    return closure

def serve_dashboards(
        root: Path,
        title: str
        ) -> None:
    """
    Serve new dashboards at an endpoint determined by title.

    Parameters
    ----------
    root: Path
        Path to root data directory used by dashboards.
    title: str
        Title that appears in browser tab and dashboard header. Dashboards are
        served at an endpoint that is a slugified version of the title. (
        "National Water Model Evaluations" is served at
        www.myhost.com/national-water-model-evaluations).
    """
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
