"""
Application-wide configuration.
"""
from pathlib import Path
import inspect

from pydantic import BaseModel

from .logger import get_logger

class Configuration(BaseModel):
    """
    Application configuration options.

    Attributes
    ----------
    key: str
        USGS API key.
    """
    usgs_api_key: str

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
