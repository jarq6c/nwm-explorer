"""Sidebar options that affect application behavior."""
from typing import Callable
from pathlib import Path

import panel as pn
from panel.viewable import Viewer

from .constants import AxisType, MeasurementUnits
from .usgs import load_site_information

UNIT_CONVERSIONS: dict[str, float] = {
    MeasurementUnits.CUBIC_FEET_PER_SECOND: 1.0,
    MeasurementUnits.CUBIC_METERS_PER_SECOND: 0.3048 ** 3.0,
    MeasurementUnits.CUBIC_FEET_PER_SECOND_PER_SQUARE_MILE: 1.0,
    MeasurementUnits.INCHES_PER_HOUR: 1.0,
    MeasurementUnits.CUMULATIVE_INCHES_PER_HOUR: 1.0
}
"""Measurement unit conversion functions. Assumes base units of CFS"""

def compute_conversion_factor(
        root: Path,
        usgs_site_code: str,
        measurement_units: str
) -> float:
    """
    Compute conversion factor for given site code and measurement units.
    """
    # Extract area
    if measurement_units in [
        MeasurementUnits.CUBIC_FEET_PER_SECOND_PER_SQUARE_MILE,
        MeasurementUnits.INCHES_PER_HOUR,
        MeasurementUnits.CUMULATIVE_INCHES_PER_HOUR]:
        # Load site information
        cols = ["drainage_area", "contributing_drainage_area"]
        df = load_site_information(root, usgs_site_code, cols)
        area = df.min_horizontal().item(0)
        if area is None:
            return 0.0
        return UNIT_CONVERSIONS[measurement_units] / area
    return UNIT_CONVERSIONS[measurement_units]

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
                options=list(UNIT_CONVERSIONS.keys()),
                value=list(UNIT_CONVERSIONS.keys())[0]
            )
        }

    def __panel__(self):
        return pn.Column("# Streamflow options", *list(self._widgets.values()))

    def bind(self, function: Callable) -> None:
        """Bind function to widgets."""
        for k, w in self._widgets.items():
            pn.bind(function, w.param.value, watch=True, callback_type=k)

    @property
    def axis_scale(self) -> AxisType:
        """Currently select axis type."""
        return AxisType(self._widgets["axis_scale"].value.lower())

    @property
    def measurement_units(self) -> str:
        """Currently select measurment units."""
        return self._widgets["measurement_units"].value
