"""
Post-event evaluation layout.
"""
import inspect
from typing import Any

import panel as pn
from panel.viewable import Viewer

from nwm_explorer.logging.loggers import get_logger
from nwm_explorer.panes.configuration import ConfigurationPane
from nwm_explorer.application.api import EvaluationRegistry

class PostEventEvaluationLayout(Viewer):
    """
    Dashboard main area layout for post-event evaluations.

    Parameters
    ----------
    params: any
        Additional keyword arguments passed directly to panel.viewable.Viewer.
    """
    def __init__(
            self,
            registry: EvaluationRegistry,
            configuration: ConfigurationPane,
            **params: dict[str, Any]
            ):
        # Apply parameters
        super().__init__(**params)

        # Get logger
        name = __loader__.name + "." + inspect.currentframe().f_code.co_name
        logger = get_logger(name)
        logger.info("Build post-event evaluations")
    
    def __panel__(self) -> pn.Card:
        return pn.Card(title="Post-event Evaluations")
