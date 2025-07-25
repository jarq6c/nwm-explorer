"""Download and manage USGS site information."""
from pathlib import Path
import inspect
from concurrent.futures import ProcessPoolExecutor
from typing import Literal

import pandas as pd
import polars as pl

from nwm_explorer.data.download import download_files
from nwm_explorer.logging.logger import get_logger
from nwm_explorer.data.mapping import ModelDomain
from nwm_explorer.data.usgs import tsv_gz_validator

HUC_URL: str = "https://waterservices.usgs.gov/nwis/site/?format=rdb&siteOutput=expanded&siteStatus=all&parameterCd=00060&huc="
SITE_URL: str = "https://waterservices.usgs.gov/nwis/site/?format=rdb&siteOutput=expanded&siteStatus=all&parameterCd=00060&sites="

COLUMNS: dict[str, str] = {
    "site_no": "usgs_site_code",
    "station_nm": "site_name",
    "dec_lat_va": "latitude",
    "dec_long_va": "longitude",
    "district_cd": "district_code",
    "state_cd": "state_code",
    "county_cd": "county_code",
    "huc_cd": "HUC",
    "drain_area_va": "drainage_area",
    "contrib_drain_area_va": "contributing_drainage_area"
}
"""Mapping from USGS site service columns to new column labels."""

DTYPES: dict[str, type] = {
    "latitude": float,
    "longitude": float,
    "drainage_area": float,
    "contributing_drainage_area": float
}
"""Mapping from columns to data types."""

def generate_site_info_urls(
        url_type: Literal["HUC", "SITE"] = "HUC",
        site_list: list[str] | None = None
) -> list[str]:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    logger.info("Building url list")
    if url_type == "HUC":
        return [HUC_URL+f"{h}".zfill(2) for h in range(1, 22)]
    return [SITE_URL+s for s in site_list]

def process_site_tsv(filepath: Path) -> pd.DataFrame:
    """
    Process a NWIS IV API TSV file.

    Parameters
    ----------
    filepaths: list[Path]
        Path to file to process.

    Returns
    -------
    pandas.DataFrame
    """
    if not filepath.exists():
        return pd.DataFrame()
    df = pd.read_csv(
        filepath,
        comment="#", 
        dtype=str,
        sep="\t"
        )[list(COLUMNS.keys())].rename(columns=COLUMNS).iloc[1:, :]
    
    for c, t in DTYPES.items():
        df[c] = df[c].astype(t)
    return df

def download_site_info(
    root: Path,
    routelinks: dict[ModelDomain, pl.LazyFrame],
    jobs: int,
    retries: int = 10
    ) -> None:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Check for file
    ofile = root / "site_information.parquet"
    if ofile.exists():
        logger.info(f"Found {ofile}")
        return

    # File details
    logger.info("Generating huc file details")
    temporary_directory = root / "temp"
    temporary_directory.mkdir(exist_ok=True)
    urls = generate_site_info_urls()
    file_paths = [temporary_directory / f"huc_{h}.tsv.gz" for h in range(1, 22)]
    logger.info(f"Saving TSV files to {temporary_directory}")
    download_files(
        *list(zip(urls, file_paths)),
        limit=10,
        timeout=3600, 
        headers={"Accept-Encoding": "gzip"},
        auto_decompress=False,
        file_validator=tsv_gz_validator,
        retries=retries
    )

    # Process data
    logger.info("Processing files")
    chunksize = max(1, len(file_paths) // jobs)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        huc_df = pd.concat(pool.map(
            process_site_tsv, file_paths, chunksize=chunksize), ignore_index=True)
        
    # Compare to routelink
    site_lists = []
    for _, r in routelinks.items():
        site_lists.append(r.select("usgs_site_code").collect().to_pandas())
    rl_sites = pd.concat(site_lists, ignore_index=True)
    rl_sites = rl_sites[~rl_sites["usgs_site_code"].isin(huc_df["usgs_site_code"])]

    # Download remaining sites
    site_list = rl_sites["usgs_site_code"].to_list()
    urls = generate_site_info_urls("SITE", site_list)
    file_paths = [temporary_directory / f"site_{s}.tsv.gz" for s in site_list]
    logger.info(f"Saving TSV files to {temporary_directory}")
    download_files(
        *list(zip(urls, file_paths)),
        limit=10,
        timeout=3600, 
        headers={"Accept-Encoding": "gzip"},
        auto_decompress=False,
        file_validator=tsv_gz_validator,
        retries=retries
    )

    # Process data
    logger.info("Processing files")
    chunksize = max(1, len(file_paths) // jobs)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        site_df = pd.concat(pool.map(
            process_site_tsv, file_paths, chunksize=chunksize), ignore_index=True)
    
    # Combine all data
    huc_df = pd.concat([huc_df, site_df], ignore_index=True)

    # Add site url
    huc_df["monitoring_url"] = "https://waterdata.usgs.gov/monitoring-location/USGS-" + huc_df["usgs_site_code"]

    # Saving
    logger.info(f"Saving {ofile}")
    pl.DataFrame(huc_df).write_parquet(ofile)

    # Clean-up
    logger.info("Cleaning up")
    for tfile in temporary_directory.glob("*"):
        if tfile.exists():
            tfile.unlink()
    temporary_directory.rmdir()

def scan_site_info(root: Path) -> pl.LazyFrame:
    return pl.scan_parquet(root / "site_information.parquet")
