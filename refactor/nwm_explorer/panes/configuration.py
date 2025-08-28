"""
Configuration options relevant to the entire application.
"""
import inspect
from typing import Any

import panel as pn
from panel.viewable import Viewer
from param.parameterized import Parameters

from nwm_explorer.logging.loggers import get_logger

class TitledCheckBoxGroup(Viewer):
    """
    CheckBoxGroup with a rendered title.

    Parameters
    ----------
    params: any
        Keyword arguments passed to pn.widgets.CheckBoxGroup.
    """
    def __init__(
            self,
            **params
        ) -> None:
        self._check_box_group = pn.widgets.CheckBoxGroup(**params)

    def __panel__(self) -> pn.Column:
        return pn.Column(
            self._check_box_group.name,
            self._check_box_group
        )

    @property
    def param(self) -> Parameters:
        return self._check_box_group.param

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
            streamflow_options: list[str],
            precipitation_options: list[str],
            map_layers: list[str],
            **params: dict[str, Any]
        ):
        # Apply parameters
        super().__init__(**params)

        # Get logger
        name = __loader__.name + "." + inspect.currentframe().f_code.co_name
        logger = get_logger(name)
        logger.info("Build configuration pane")

        # Widgets
        self.widgets = {
            # Dashboard mode
            "mode_selector": pn.widgets.Select(
                name="Dashboard Mode",
                options=mode_options
                ),

            # Streamflow units
            "streamflow_selector": pn.widgets.Select(
                name="Streamflow measurement units",
                options=streamflow_options
                ),

            # Precipitation units
            "precipitation_selector": pn.widgets.Select(
                name="Precipitation measurement units",
                options=precipitation_options
                ),

            # Map layers
            "layer_selector": TitledCheckBoxGroup(
                name="Map layers",
                options=map_layers,
                value=map_layers[0:1]
            )
        }

        # Add widgets as attributes
        for k, v in self.widgets.items():
            setattr(self, k, v)

    def __panel__(self) -> pn.Card:
        return pn.Card(
            pn.Column(*[v for v in self.widgets.values()]),
            collapsible=False,
            title="Configuration"
        )
