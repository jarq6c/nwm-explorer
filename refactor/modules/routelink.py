"""Download and process RouteLink information."""
from pathlib import Path
from tempfile import TemporaryDirectory
import inspect

import pandas as pd
import polars as pl
from yarl import URL

from .logger import get_logger
from .downloads import download_files

ROUTELINK_URL: str = (
    "https://www.hydroshare.org/resource"
    "/1fe9975004ce4b5097d41939afa14f84/data/contents/RouteLink.h5"
)
"""URL to pandas-compatible RouteLink HDF5 source file."""

ROUTELINK_PARQUET: Path = Path("./data/routelink.parquet")
"""Path to polars-compatible RouteLink parquet file used by application."""

def download_routelink(
        url: str | URL = ROUTELINK_URL,
        file_path: Path = ROUTELINK_PARQUET
) -> pl.LazyFrame:
    """
    Download RouteLink file.

    Parameters
    ----------
    url: str | URL
        Source URL.
    file_path: pathlib.Path
        Destination file path.
    
    Returns
    -------
    polars.LazyFrame
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Check for file
    if file_path.exists():
        logger.info("%s exists", file_path)
        return pl.scan_parquet(file_path)
    logger.info("Downloading %s", file_path)

    # Create parents
    file_path.parent.mkdir(exist_ok=True, parents=True)

    # Download RouteLink
    with TemporaryDirectory() as td:
        # Temporary download path
        ofile = Path(td) / "routelink.h5"

        # Download
        download_files(
            (url, ofile),
        )

        # Clean-up
        df = pd.read_hdf(ofile)
        df = df[df["usgs_site_code"].str.isdigit()]
        short = df["usgs_site_code"].str.len() <= 7
        df.loc[short, "usgs_site_code"] = "0" + df.loc[short, "usgs_site_code"]

        # Save
        enumerated_site_code = pl.Enum(df["usgs_site_code"].to_list())
        data = pl.DataFrame(
            df,
            schema_overrides={"usgs_site_code": enumerated_site_code}
        )
        data.write_parquet(file_path)

    # Scan and return
    return pl.scan_parquet(file_path)
