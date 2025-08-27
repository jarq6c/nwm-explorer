"""
Configuration options relevant to the entire application.
"""
import inspect
from typing import Any

import panel as pn
from panel.viewable import Viewer

from nwm_explorer.logging.loggers import get_logger

class ConfigurationPane(Viewer):
    """
    Provides a card interface with different configuration options, relevant
    to the entire application.

    Parameters
    ----------
    mode_options: list[str]
        Selectable dashboard mode options.
    params: any
        Additional keyword arguments passed directly to panel.viewable.Viewer.
    """
    def __init__(
            self,
            mode_options: list[str],
            **params: dict[str, Any]
        ):
        # Apply parameters
        super().__init__(**params)

        # Get logger
        name = __loader__.name + "." + inspect.currentframe().f_code.co_name
        logger = get_logger(name)
        logger.info("Build configuration pane")

        # Dashboard mode
        self.mode_selector = pn.widgets.Select(
            name="Dashboard Mode",
            options=mode_options
            )
    
    def __panel__(self) -> pn.Card:
        return pn.Card(
            pn.Column(
                self.mode_selector
            ),
            collapsible=False,
            title="Configuration"
        )
