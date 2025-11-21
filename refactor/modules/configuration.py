"""
Application-wide configuration.
"""
from pathlib import Path
import inspect

from pydantic import BaseModel

from .logger import get_logger

class MapLayer(BaseModel):
    """
    Model for additional map layers.

    Attributes:
    name: str
        Name to display.
    path: pathlib.Path
        Path to GeoParquet file.
    columns: list[str], optional
        List of column values to display on hover.
    """
    name: str
    path: Path
    columns: list[str] | None = None

class Configuration(BaseModel):
    """
    Application configuration options.

    Attributes
    ----------
    title: str
        Application name.
    endpoint: str
        Endpoint for service.
    root: pathlib.Path
        Root data directory.
    key: str
        USGS API key.
    map_layers: list[MapLayer]
        List of additional map layers to show on map.
    """
    title: str
    endpoint: str
    root: Path
    usgs_api_key: str
    map_layers: list[MapLayer]

def load_configuration(configuration_file: Path) -> Configuration:
    """
    Returns Configuration object after validating file.

    Parameters
    ----------
    configuration_file: pathlib.Path
        Path to configuration JSON file.

    Returns
    -------
    Configuration
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Load configuration
    logger.info("Reading %s", configuration_file)
    with configuration_file.open("r") as fo:
        return Configuration.model_validate_json(fo.read())
