"""
Application-wide configuration.
"""
from pathlib import Path
import inspect
from datetime import datetime

from pydantic import BaseModel

from nwm_explorer.logger import get_logger

class Evaluation(BaseModel):
    """
    Model for evaluation parameters.

    Attributes
    ----------
    label: str
        Machine-friendly label used to generate parquet store.
    start_time: pandas.Timestamp
        First reference time.
    end_time: pandas.Timestamp
        Last reference time.
    """
    label: str
    start_time: datetime
    end_time: datetime

class MapLayer(BaseModel):
    """
    Model for additional map layers.

    Attributes
    ----------
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
    usgs_api_key: str
        USGS API key.
    map_layers: list[MapLayer]
        List of additional map layers to show on map.
    evaluations: list[Evaluation]
        List of evaluations to run.
    processes: int, optional, default 1
        Number of parallel processes to use for computation.
    sites_per_chunk: int, optional, default 1
        Maximum number of sites to load into memory at once.
    retries: int, optional, default 3
        Number of times to retry downloads.
    """
    title: str
    endpoint: str
    root: Path
    usgs_api_key: str
    map_layers: list[MapLayer]
    evaluations: list[Evaluation]
    processes: int = 1
    sites_per_chunk: int = 1
    retries: int = 3

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
