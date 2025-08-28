"""
Routine operational evaluation layout.
"""
import inspect
from typing import Any

import panel as pn
from panel.viewable import Viewer

from nwm_explorer.logging.loggers import get_logger

class RoutineOperationalEvaluationLayout(Viewer):
    """
    Dashboard main area layout for routine operational evaluations.

    Parameters
    ----------
    params: any
        Additional keyword arguments passed directly to panel.viewable.Viewer.
    """
    def __init__(self, **params: dict[str, Any]):
        # Apply parameters
        super().__init__(**params)

        # Get logger
        name = __loader__.name + "." + inspect.currentframe().f_code.co_name
        logger = get_logger(name)
        logger.info("Build routine operational evaluations")

        # Main display
        self.main = pn.pane.Markdown()
    
    def __panel__(self) -> pn.Card:
        return pn.Card(self.main, title="Routine Operational Evaluations")
