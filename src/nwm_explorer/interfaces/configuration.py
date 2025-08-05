"""Site configuration widgets."""
from enum import StrEnum
from typing import Callable
from dataclasses import dataclass

import panel as pn

from nwm_explorer.interfaces.filters import CallbackType, EventHandler

CMS_FACTOR: float = 0.3048 ** 3.0
"""Conversion factor from CFS to CMS."""

INH_FACTOR: float = 3.0 / 1936.0
"""Conversion factor from CFS to inches per hour, must divide by area in square miles."""

class MeasurementUnits(StrEnum):
    """Measurement units enums."""
    cfs = "CFS"
    cms = "CMS"
    inh = "inch/h"
    cfs_sqmi = "CFS/sq.mi."

MEASUREMENT_UNIT_STRINGS: dict[str, MeasurementUnits] = {
    "Cubic feet per second per square mile": MeasurementUnits.cfs_sqmi,
    "Cubic feet per second": MeasurementUnits.cfs,
    "Cubic meters per second": MeasurementUnits.cms,
    "Inches per hour": MeasurementUnits.inh
}
"""Mapping from pretty strings to MeasurementUnits."""

@dataclass
class SiteConfigurationState:
    """Site Configuration state variables."""
    units: MeasurementUnits

class ConfigurationWidgets:
    def __init__(self, layers: list[str] | None = None):
        # Filtering options
        self.callbacks: list[EventHandler] = []
        self.units_selector = pn.widgets.RadioBoxGroup(
            name="Discharge Units",
            options=list(MEASUREMENT_UNIT_STRINGS.keys())
        )
        self.layer_selector = pn.widgets.CheckBoxGroup(
            name="Additional Map Layers",
            options=layers if layers is not None else []
        )

    @property
    def state(self) -> SiteConfigurationState:
        """Returns current state of site options."""
        return SiteConfigurationState(
            units=MEASUREMENT_UNIT_STRINGS[self.units_selector.value]
        )

    def servable(self) -> pn.Card:
        return pn.Column(
            pn.pane.Markdown("# Dashboard Configuration"),
            pn.Card(
                self.units_selector,
                title="Measurement Units",
                collapsible=False
            ),
            pn.Card(
                self.layer_selector,
                title="Additional Map Layers",
                collapsible=False
            )
        )

    def register_callback(self, func: Callable) -> None:
        """Register callback function."""
        pn.bind(func, self.units_selector, callback_type=CallbackType.measurement_units, watch=True)
        pn.bind(func, self.layer_selector, callback_type=CallbackType.layers, watch=True)
        self.callbacks.append(func)
