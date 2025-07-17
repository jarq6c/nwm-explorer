"""Testing interfaces and data models."""
from pathlib import Path
import tarfile
import inspect

import polars as pl

from nwm_explorer.data.download import download_files
from nwm_explorer.logging.logger import get_logger
from nwm_explorer.data.mapping import ModelDomain

ROUTELINKS_URL: str = (
    "https://www.hydroshare.org/resource/"
    "e9fe66730d184bdfbaea19639bd7cb55/data/"
    "contents/RouteLinks.tar.gz"
    )
"""NWM RouteLinks on HydroShare."""

ROUTELINK_FILENAMES: dict[ModelDomain, str] = {
    ModelDomain.alaska: "RouteLink_AK.csv",
    ModelDomain.conus: "RouteLink_CONUS.csv",
    ModelDomain.hawaii: "RouteLink_HI.csv",
    ModelDomain.puertorico: "RouteLink_PRVI.csv"
}
"""Mapping from domains to routelink files names."""

def build_routelink_filepath(root: Path, domain: ModelDomain) -> Path:
    return root / "parquet" / domain / "routelink.parquet"

def build_routelink_filepaths(root: Path) -> dict[ModelDomain, Path]:
    """Returns mapping from domains to parquet filepaths."""
    return {d: build_routelink_filepath(root, d) for d in ROUTELINK_FILENAMES}

def get_routelink_reader(root: Path, domain: ModelDomain) -> pl.LazyFrame:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)
    
    # Get file path
    fp = build_routelink_filepath(root, domain)
    logger.info(f"Scanning {fp}")
    return pl.scan_parquet(fp)

def get_routelink_readers(root: Path) -> dict[ModelDomain, pl.LazyFrame]:
    """Returns mapping from ModelDomain to polars.LazyFrame."""
    return {d: get_routelink_reader(root, d) for d in ROUTELINK_FILENAMES}

def download_routelinks(root: Path, retries: int = 10) -> None:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Create root directory
    logger.info(f"Checking {root.absolute()}")
    root.mkdir(exist_ok=True, parents=True)

    # Check for files
    logger.info("Looking for existing routelink files")
    all_files_exist = False
    routelink_filepaths = build_routelink_filepaths(root)
    for fp in routelink_filepaths.values():
        all_files_exist = fp.exists()
    
    if all_files_exist:
        logger.info("Routelink files exist for all domains")
        return

    # Download and process routelink
    logger.info("Downloading routelink files")
    directory = root / "routelinks"
    filepath = directory / "Routelinks.tar.gz"
    directory.mkdir(exist_ok=True)
    download_files((ROUTELINKS_URL, filepath), auto_decompress=False, retries=retries)

    logger.info("Extracting routelink files")
    with tarfile.open(filepath, "r:gz") as tf:
        tf.extractall(directory)
    
    logger.info("Processing routelink files")
    for d, fn in ROUTELINK_FILENAMES.items():
        ofile = routelink_filepaths[d]
        if ofile.exists():
            print(f"Skipping {ofile}")
            continue
        ofile.parent.mkdir(exist_ok=True, parents=True)
        ifile = directory / f"csv/{fn}"
        df = pl.read_csv(
            ifile,
            comment_prefix="#",
            schema_overrides={"usgs_site_code": pl.String}
        )
        
        if d == ModelDomain.conus:
            df = df.with_columns(
                    pl.col("usgs_site_code").replace("8313150", "08313150")
                )

        # Limit usgs_site_code to digits
        pdf = df.to_pandas()
        pdf = pdf[pdf["usgs_site_code"].str.isdigit()]
        df = pl.DataFrame(pdf)
        df.write_parquet(ofile)
        ifile.unlink()
    
    logger.info("Cleaning up routelink files")
    filepath.unlink()
    (directory / "csv").rmdir()
    directory.rmdir()
