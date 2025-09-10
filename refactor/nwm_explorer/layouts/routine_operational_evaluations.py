"""
Routine operational evaluation layout.
"""
import inspect
from typing import Any

import panel as pn
from panel.viewable import Viewer

from nwm_explorer.logging.loggers import get_logger
from nwm_explorer.application.api import EvaluationRegistry
from nwm_explorer.panes.filters import Filters

class RoutineOperationalEvaluationLayout(Viewer):
    """
    Dashboard main area layout for routine operational evaluations.

    Parameters
    ----------
    registry: EvaluationRegistry
        EvaluationRegistry used throughout dashboard.
    params: any
        Additional keyword arguments passed directly to panel.viewable.Viewer.
    """
    def __init__(self, registry: EvaluationRegistry, **params: dict[str, Any]):
        # Apply parameters
        super().__init__(**params)

        # Get logger
        name = __loader__.name + "." + inspect.currentframe().f_code.co_name
        logger = get_logger(name)
        logger.info("Build routine operational evaluations")

        # Cards
        self.filters = Filters(registry)
        self.site_map = pn.pane.Markdown("Site map")
        self.histograms = [
            pn.pane.Markdown("Histogram"),
            pn.pane.Markdown("Histogram"),
            pn.pane.Markdown("Histogram"),
            pn.pane.Markdown("Histogram")
            ]
        self.site_information = pn.pane.Markdown("Site information")
        self.hydrograph = pn.pane.Markdown("Hydrograph")
        self.site_metrics = pn.pane.Markdown("Site metrics")

    def __panel__(self) -> pn.Card:
        return pn.Column(
            pn.Row(self.filters, self.site_map, pn.GridBox(*self.histograms, ncols=2)),
            pn.Row(self.site_information, self.hydrograph, self.site_metrics)
        )
