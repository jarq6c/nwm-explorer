"""Sidebar options that affect application behavior."""
from typing import Callable

import panel as pn
from panel.viewable import Viewer

from .constants import AxisType, UNIT_CONVERSION_FUNCTIONS

class StreamflowOptions(Viewer):
    """WidetBox for selecting streamflow plot parameters."""
    def __init__(self, **params):
        super().__init__(**params)

        # Setup
        self._widgets = {
            "axis_scale": pn.widgets.Select(
                name="Scale",
                options=["Linear", "Log"],
                value="Linear"
            ),
            "measurement_units": pn.widgets.Select(
                name="Measurement units",
                options=list(UNIT_CONVERSION_FUNCTIONS.keys()),
                value=list(UNIT_CONVERSION_FUNCTIONS.keys())[0]
            )
        }

    def __panel__(self):
        return pn.WidgetBox("# Streamflow options", *list(self._widgets.values()))

    def bind(self, function: Callable) -> None:
        """Bind function to widgets."""
        for k, w in self._widgets.items():
            pn.bind(function, w.param.value, watch=True, callback_type=k)

    @property
    def axis_scale(self) -> AxisType:
        """Currently select axis type."""
        return AxisType(self._widgets["axis_scale"].value.lower())

    @property
    def units(self) -> str:
        """Currently select measurment units."""
        return self._widgets["measurement_units"].value
