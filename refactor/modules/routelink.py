"""Download and process RouteLink information."""
from pathlib import Path
from tempfile import TemporaryDirectory
import inspect
import tarfile

import pandas as pd
import polars as pl
from yarl import URL

from .logger import get_logger
from .downloads import download_files
from .nwm import ModelDomain
from .usgs import enumerate_sites

ROUTELINK_URL: str = (
    "https://www.hydroshare.org/resource"
    "/1fe9975004ce4b5097d41939afa14f84/data/contents/RouteLinks.tar.gz"
)
"""URL to RouteLink CSV tarball."""

ROUTELINK_PARQUET: Path = Path("routelink.parquet")
"""Default path to polars-compatible RouteLink parquet file used by application."""

ROUTELINK_FILENAMES: dict[ModelDomain, str] = {
    ModelDomain.ALASKA: "RouteLink_AK.csv",
    ModelDomain.CONUS: "RouteLink_CONUS.csv",
    ModelDomain.HAWAII: "RouteLink_HI.csv",
    ModelDomain.PUERTO_RICO: "RouteLink_PRVI.csv"
}
"""Mapping from domains to routelink files names."""

def download_routelink(
        root: Path,
        url: str | URL = ROUTELINK_URL
) -> pl.LazyFrame:
    """
    Download RouteLink file.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    url: str | URL
        Source URL.
    
    Returns
    -------
    polars.LazyFrame
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Check for file
    file_path = root / ROUTELINK_PARQUET
    if file_path.exists():
        logger.info("Scanning %s", file_path)
        return pl.scan_parquet(file_path)
    logger.info("Downloading %s", file_path)

    # Download RouteLink
    with TemporaryDirectory() as td:
        # Temporary download path
        ofile = Path(td) / "routelink.tar.gz"

        # Download
        download_files(
            (url, ofile),
        )

        logger.info("Extracting routelink files")
        odir = Path(td) / "routelinks"
        odir.mkdir()
        with tarfile.open(ofile, "r:gz") as tf:
            tf.extractall(odir)

        logger.info("Processing routelink files")
        dfs = []
        for d, fn in ROUTELINK_FILENAMES.items():
            ifile = odir / f"csv/{fn}"
            df = pd.read_csv(
                ifile,
                comment="#",
                dtype=str
            )
            df["domain"] = d
            dfs.append(df)

        # Clean-up
        data = pd.concat(dfs, ignore_index=True)
        data = data[data["usgs_site_code"].str.isdigit()]
        short = data["usgs_site_code"].str.len() <= 7
        data.loc[short, "usgs_site_code"] = "0" + data.loc[short, "usgs_site_code"]

        # Save
        enumerated_site_code = enumerate_sites(root)
        pl_data = pl.DataFrame(
            data,
            schema_overrides={
                "usgs_site_code": enumerated_site_code,
                "domain": ModelDomain,
                "nwm_feature_id": pl.Int64,
                "latitude": pl.Float64,
                "longitude": pl.Float64
                },
            strict=False
        ).drop_nulls("usgs_site_code")
        pl_data.write_parquet(file_path)

    # Scan
    logger.info("Scanning %s", file_path)
    return pl.scan_parquet(file_path)
