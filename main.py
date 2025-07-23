"""Download and manage USGS site information."""
from pathlib import Path
import inspect
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import polars as pl

from nwm_explorer.data.download import download_files
from nwm_explorer.logging.logger import get_logger

BASE_URL: str = "https://waterservices.usgs.gov/nwis/site/?format=rdb&siteOutput=expanded&siteStatus=all&parameterCd=00060&huc="

def build_urls() -> list[str]:
    return [BASE_URL+f"{h}".zfill(2) for h in range(1, 22)]

print(build_urls())
