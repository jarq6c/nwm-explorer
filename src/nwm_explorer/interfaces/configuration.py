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
    inh = "in."

MEASUREMENT_UNIT_STRINGS: dict[str, MeasurementUnits] = {
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
    def __init__(self):
        # Filtering options
        self.callbacks: list[EventHandler] = []
        self.units_selector = pn.widgets.RadioBoxGroup(
            name="Measurement Units",
            options=list(MEASUREMENT_UNIT_STRINGS.keys())
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
            )
        )

    def register_callback(self, func: Callable) -> None:
        """Register callback function."""
        pn.bind(func, self.units_selector, callback_type=CallbackType.measurement_units, watch=True)
        self.callbacks.append(func)
