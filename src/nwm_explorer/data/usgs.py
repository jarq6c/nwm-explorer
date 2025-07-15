"""Testing interfaces and data models."""
from pathlib import Path
import inspect
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import polars as pl

from nwm_explorer.data.download import download_files
from nwm_explorer.logging.logger import get_logger
from nwm_explorer.data.mapping import ModelDomain

TIMEZONE_MAPPING: dict[str, str] = {
    "AKST": "America/Anchorage",
    "AKDT": "America/Anchorage",
    "HST": "America/Adak",
    "HDT": "America/Adak",
    "AST": "America/Puerto_Rico",
    "CDT": "America/Chicago",
    "CST": "America/Chicago",
    "EDT": "America/New_York",
    "EST": "America/New_York",
    "MST": "America/Phoenix",
    "MDT": "America/Denver",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles"
}
"""Mapping from common timezone strings to IANA compatible strings."""

def tsv_gz_validator(filepath: Path) -> None:
    """
    Validate that given filepath opens and closes without raising.

    Parameters
    ----------
    filepath: Path
        Path to file.
    
    Returns
    -------
    None
    """
    pd.read_csv(
        filepath,
        comment="#", 
        dtype=str,
        sep="\t",
        header=None,
        nrows=1
        )

def process_nwis_tsv(filepath: Path) -> pd.DataFrame:
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
    df = pd.read_csv(
        filepath,
        comment="#", 
        dtype=str,
        sep="\t",
        header=None,
        ).iloc[2:, 1:5]

    if df.iloc[:, -1].isna().all():
        return pd.DataFrame()

    df = df.set_axis(
        ["usgs_site_code", "value_time", "timezone", "value"],
        axis="columns")
    df = df[df["usgs_site_code"].str.isdigit()]
    df["value_time"] = pd.to_datetime(df["value_time"])
    
    # Deal with time zones
    for tz in df["timezone"].unique():
        mapped_tz = TIMEZONE_MAPPING.get(tz, tz)
        daylight = tz.endswith("DT")
        df.loc[df["timezone"] == tz, "value_time"] = df.loc[
            df["timezone"] == tz, "value_time"].dt.tz_localize(
                mapped_tz, ambiguous=daylight).dt.tz_convert(
                    "UTC").dt.tz_localize(None)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["usgs_site_code", "value_time", "value"]].dropna()
    return df

def process_nwis_tsv_parallel(
        filepaths: list[Path],
        max_processes: int = 1
    ) -> pd.DataFrame:
    """
    Process a collection of USGS NWIS IV API TSV files and return a
    dataframe, in parallel.

    Parameters
    ----------
    filepaths: list[Path]
        List of filepaths to process.
    max_processes: int, optional, default 1
        Maximum number of cores to use simultaneously.

    Returns
    -------
    pandas.DataFrame
    """
    chunksize = max(1, len(filepaths) // max_processes)
    with ProcessPoolExecutor(max_workers=max_processes) as pool:
        df = pd.concat(pool.map(
            process_nwis_tsv, filepaths, chunksize=chunksize), ignore_index=True)
    df["usgs_site_code"] = df["usgs_site_code"].astype("category")
    return df

def generate_usgs_urls(
        site_list: list[str],
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp
) -> list[str]:
    """
    Generate a list of USGS NWIS RDB URLs for each site in site_list.

    Parameters
    ----------
    site_list: list[str]
        List of USGS site codes.
    start_datetime: pd.Timestamp
        startDT.
    end_datetime: pd.Timestamp
        endDT.
    
    Returns
    -------
    list[str]
    """
    urls = []
    prefix = "https://waterservices.usgs.gov/nwis/iv/?format=rdb&"
    start_str = start_datetime.strftime("%Y-%m-%dT%H:%MZ")
    end_str = end_datetime.strftime("%Y-%m-%dT%H:%MZ")
    suffix = "&siteStatus=all&parameterCd=00060"

    for site in site_list:
        middle = f"sites={site}&startDT={start_str}&endDT={end_str}"
        urls.append(prefix+middle+suffix)
    return urls

def get_usgs_reader(
    root: Path,
    domain: ModelDomain,
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    ) -> pl.LazyFrame:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Generate observational time periods
    logger.info(f"Scanning USGS {domain} {startDT} to {endDT}")
    logger.info("Generating observational time periods")
    padded_start = startDT - pd.Timedelta("31d")
    padded_end = endDT + pd.Timedelta("31d")
    months = pd.date_range(
        start=padded_start.strftime("%Y%m01"),
        end=padded_end.strftime("%Y%m01"),
        freq="MS"
    )

    # File details
    logger.info("Generating file details")
    file_paths = []
    for idx in range(len(months)-1):
        filename = months[idx].strftime("usgs.%Y%m.parquet")
        fp = root / "parquet" / domain / filename
        if fp.exists():
            file_paths.append(fp)
    
    # Scan data
    return pl.scan_parquet(file_paths).filter(
        pl.col("value_time") >= startDT,
        pl.col("value_time") <= endDT
    )

def get_usgs_readers(
    root: Path,
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    ) -> dict[ModelDomain, pl.LazyFrame]:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Generate observational time periods
    return {d: get_usgs_reader(root, d, startDT, endDT) for d in list(ModelDomain)}

def download_usgs(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    root: Path,
    routelinks: dict[ModelDomain, pl.LazyFrame],
    jobs: int
    ) -> None:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Generate observational time periods
    logger.info("Generating observational time periods")
    padded_start = startDT - pd.Timedelta("31d")
    padded_end = endDT + pd.Timedelta("31d")
    months = pd.date_range(
        start=padded_start.strftime("%Y%m01"),
        end=padded_end.strftime("%Y%m01"),
        freq="MS"
    )

    # Sites to download
    logger.info("Reading routelinks")
    sites = {d: df.select("usgs_site_code").collect()["usgs_site_code"].to_numpy() for d, df in routelinks.items()}

    # File details
    logger.info("Generating file details")
    temporary_directory = root / "temp"
    temporary_directory.mkdir(exist_ok=True)
    logger.info(f"Saving TSV files to {temporary_directory}")
    for idx in range(len(months)-1):
        filename = months[idx].strftime("usgs.%Y%m.parquet")

        for d in sites:
            ofile = root / "parquet" / d / filename
            logger.info(f"Building {ofile}")
            if ofile.exists():
                logger.info(f"Found {ofile}")
                continue
            urls = generate_usgs_urls(sites[d], months[idx], months[idx+1])
            file_paths = [temporary_directory / f"usgs-{s}.tsv.gz" for s in sites[d]]

            logger.info("Downloading USGS data")
            download_files(
                *list(zip(urls, file_paths)),
                limit=10,
                timeout=3600, 
                headers={"Accept-Encoding": "gzip"},
                auto_decompress=False,
                file_validator=tsv_gz_validator
            )

            logger.info("Processing USGS data")
            file_paths = list(temporary_directory.glob("*.tsv.gz"))
            try:
                data = process_nwis_tsv_parallel(
                    file_paths,
                    jobs
                )
                logger.info(f"Saving {ofile}")
                pl.DataFrame(data).write_parquet(ofile)
            except ValueError:
                logger.info("Unable to process files")
            except KeyError:
                logger.info("Unable to process files")

            logger.info("Cleaning up TSV files")
            for fp in file_paths:
                if fp.exists():
                    logger.info(str(fp))
                    fp.unlink()

    # Clean-up
    logger.info(f"Cleaning up {temporary_directory}")
    try:
        temporary_directory.rmdir()
    except OSError:
        logger.info(f"Unable to clean-up {temporary_directory}")
