"""
Web-based application used to vizualize and explore hydrological model
evaluations, primarily for the National Water Model.
"""
import inspect
from collections.abc import Callable

import panel as pn

from nwm_explorer.logging.loggers import get_logger
from nwm_explorer.layouts.dashboard import Dashboard, generate_dashboard
from nwm_explorer.application.api import EvaluationRegistry

def generate_dashboard_closure(
        registry: EvaluationRegistry
        ) -> Callable[[], Dashboard]:
    """
    Generates a partial function that returns Dashboards with registry applied.
    This is required to generate dashboards with different parameters
    at run time.

    Parameters
    ----------
    registry: EvaluationRegistry
        Registry used by dashboards.
    
    Returns
    -------
    Callable[[], Dashboard]
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)
    logger.info("Generating dashboard closure")
    def closure() -> Dashboard:
        return generate_dashboard(registry)
    return closure

def serve_dashboards(
        registry: EvaluationRegistry
        ) -> None:
    """
    Serve new dashboards at an endpoint determined by slug given in registry.

    Parameters
    ----------
    registry: EvaluationRegistry
        Registry used by dashboards.
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Serve
    endpoints = {
        registry.dashboard_configuration.slug: generate_dashboard_closure(registry)
    }
    logger.info("Serving dashboards")
    pn.serve(endpoints)
