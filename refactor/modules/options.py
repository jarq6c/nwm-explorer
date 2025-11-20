"""Sidebar options that affect application behavior."""
from typing import Callable

import panel as pn
from panel.viewable import Viewer

class StreamflowOptions(Viewer):
    """WidetBox for selecting streamflow plot parameters."""
    def __init__(self, **params):
        super().__init__(**params)

        # Setup
        self._widgets = [
            pn.widgets.Select(
                name="Scale",
                options=["Linear", "Log"],
                value="Linear"
            ),
            pn.widgets.Select(
                name="Measurement units",
                options=["CFS", "CMS", "CFS/sq.mi.", "inch/h"],
                value="CFS"
            )
        ]

    def __panel__(self):
        return pn.WidgetBox("# Streamflow options", *self._widgets)

    def bind(self, function: Callable) -> None:
        """Bind function to widgets."""
        for w in self._widgets:
            pn.bind(function, w.param.value, watch=True, callback_type="streamflow")
