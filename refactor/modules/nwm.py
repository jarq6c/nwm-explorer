"""Download and process National Water Model output."""
from enum import StrEnum

class ModelDomain(StrEnum):
    """Model domains."""
    ALASKA = "alaska"
    HAWAII = "hawaii"
    CONUS = "conus"
    PUERTO_RICO = "puertorico"
