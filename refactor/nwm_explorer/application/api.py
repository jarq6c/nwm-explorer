"""
Objects and methods needed to support programmatic interaction with various
application components.
"""
from pathlib import Path
from enum import StrEnum
from pydantic import BaseModel

class ModelDomain(StrEnum):
    """Symbols that refer to modeling spatial domains."""
    conus = "CONUS"
    alaska = "Alaska"
    hawaii = "Hawaii"
    puertorico = "Puerto Rico"

class ModelConfiguration(StrEnum):
    """Symbols that refer to model configurations."""
    ana_mrms = "Analysis (MRMS/Stage IV)"
    short_range = "Short range forecast (HRRR)"
    medium_range = "Medium range forecast (GFS)"

class DashboardConfiguration(BaseModel):
    """
    Dashboard configuration options.

    Attributes
    ----------
    title: str
        Dashboard title displayed on header.
    slug: str
        Dashboard slug appended to hostname in the URL.
    """
    title: str
    slug: str

class EvaluationRegistry(BaseModel):
    """
    Centralized object used to track various options used through the application.

    Attributes
    ----------
    dashboard_configuration: DashboardConfiguration
        Dashboard configuration options.
    evaluations: dict[str, EvaluationSpec]
        Dict of EvaluationSpec keyed to hashable str.
    """
    dashboard_configuration: DashboardConfiguration
    evaluations: dict[str, dict[ModelDomain, dict[ModelConfiguration, Path]]]
