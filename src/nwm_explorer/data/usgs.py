"""Testing interfaces and data models."""
from pathlib import Path
import tarfile
import inspect

import polars as pl

from nwm_explorer.data.download import download_files
from nwm_explorer.logging.logger import get_logger
from nwm_explorer.data.mapping import ModelDomain

# TODO make this file work

usgs_URL: str = (
    "https://www.hydroshare.org/resource/"
    "e9fe66730d184bdfbaea19639bd7cb55/data/"
    "contents/usgs.tar.gz"
    )
"""NWM usgs on HydroShare."""

usgs_FILENAMES: dict[ModelDomain, str] = {
    ModelDomain.alaska: "usgs_AK.csv",
    ModelDomain.conus: "usgs_CONUS.csv",
    ModelDomain.hawaii: "usgs_HI.csv",
    ModelDomain.puertorico: "usgs_PRVI.csv"
}
"""Mapping from domains to usgs files names."""

def build_usgs_filepath(root: Path, domain: ModelDomain) -> Path:
    return root / "parquet" / domain / "usgs.parquet"

def build_usgs_filepaths(root: Path) -> dict[ModelDomain, Path]:
    """Returns mapping from domains to parquet filepaths."""
    return {d: build_usgs_filepath(root, d) for d in usgs_FILENAMES}

def get_usgs_reader(root: Path, domain: ModelDomain) -> pl.LazyFrame:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)
    
    # Get file path
    fp = build_usgs_filepath(root, domain)
    logger.info(f"Scanning {fp}")
    return pl.scan_parquet(fp)

def get_usgs_readers(root: Path) -> dict[ModelDomain, pl.LazyFrame]:
    """Returns mapping from ModelDomain to polars.LazyFrame."""
    return {d: get_usgs_reader(root, d) for d in usgs_FILENAMES}

def download_usgs(root: Path) -> None:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Create root directory
    logger.info(f"Checking {root.absolute()}")
    root.mkdir(exist_ok=True, parents=True)

    # Check for files
    logger.info("Looking for existing usgs files")
    all_files_exist = False
    usgs_filepaths = build_usgs_filepaths(root)
    for fp in usgs_filepaths.values():
        all_files_exist = fp.exists()
    
    if all_files_exist:
        logger.info("usgs files exist for all domains")
        return

    # Download and process usgs
    logger.info("Downloading usgs files")
    directory = root / "usgs"
    filepath = directory / "usgs.tar.gz"
    directory.mkdir(exist_ok=True)
    download_files((usgs_URL, filepath), auto_decompress=False)

    logger.info("Extracting usgs files")
    with tarfile.open(filepath, "r:gz") as tf:
        tf.extractall(directory)
    
    logger.info("Processing usgs files")
    for d, fn in usgs_FILENAMES.items():
        ofile = usgs_filepaths[d]
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
        df.write_parquet(ofile)
        ifile.unlink()
    
    logger.info("Cleaning up usgs files")
    filepath.unlink()
    (directory / "csv").rmdir()
    directory.rmdir()
