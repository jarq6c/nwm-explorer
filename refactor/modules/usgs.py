"""Retrieve and organize USGS streamflow observations."""
from pathlib import Path
import inspect
import json
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import us
import pandas as pd
import polars as pl

from .logger import get_logger
from .downloads import download_files

NWIS_BASE_URL: str = "https://waterservices.usgs.gov/nwis/iv/?format=json&siteStatus=all"
"""NWIS IV API returning json and all site statuses."""

STATE_LIST: list[us.states.State] = us.states.STATES + [us.states.PR]
"""List of US states."""

SUBDIRECTORY: str = "usgs"
"""Subdirectory that indicates root of USGS output parquet store."""

def json_validator(ifile: Path) -> None:
    """
    Open and read json file data.
    """
    with ifile.open("r") as fi:
        json.loads(fi.read())

def build_usgs_filepath(
        root: Path,
        state_code: str,
        date: pd.Timestamp
) -> Path:
    """Build and return USGS hive-partitioned parquet file path."""
    # Build path
    state = f"state_code={state_code}"
    year = f"year={date.year}"
    month = f"month={date.month}"
    day = f"D{date.day}.parquet"
    return root / SUBDIRECTORY / state / year / month / day

def generate_usgs_url(
        date: pd.Timestamp,
        state_code: str
) -> str:
    """Returns download URL with parameters."""
    start = date.floor(freq="1d").strftime("%Y-%m-%dT%H:%MZ")
    end = (date.floor(freq="1d") + pd.Timedelta(hours=23, minutes=59)).strftime("%Y-%m-%dT%H:%MZ")
    return NWIS_BASE_URL + f"&stateCd={state_code}&startDT={start}&endDT={end}"

@dataclass
class JSONJob:
    """
    Parameters to keep track of input files and output files for processing
    USGS data.
    """
    ifile: Path
    ofile: Path

def process_json(job: JSONJob) -> None:
    """Process a JSON file."""
    # Load raw data
    with job.ifile.open("r") as fi:
        data = json.loads(fi.read())

    dfs = []
    for site in data["value"]["timeSeries"]:
        usgs_site_code = site["sourceInfo"]["siteCode"][0]["value"]
        for idx, series in enumerate(site["values"]):
            for value in series["value"]:
                # Update series
                value["usgs_site_code"] = usgs_site_code
                value["series"] = idx
                value["qualifiers"] = str(value["qualifiers"])
                dfs.append(value)
    pl.from_dicts(dfs).with_columns(
        pl.col("value").cast(pl.Float32),
        pl.col("usgs_site_code").cast(pl.Categorical),
        pl.col("series").cast(pl.Int32),
        pl.col("qualifiers").cast(pl.Categorical),
        pl.col("dateTime").str.to_datetime("%Y-%m-%dT%H:%M:%S%.3f%:z",
            time_unit="ms").dt.replace_time_zone(None)
    ).rename({
        "value": "observed_cfs",
        "dateTime": "value_time"
    }).write_parquet(job.ofile)

def download_usgs(
        start: pd.Timestamp,
        end: pd.Timestamp,
        root: Path,
        jobs: int = 1,
        retries: int = 3
    ):
    """
    Download and process NWM output.

    Parameters
    ----------
    start: pandas.Timestamp
        First reference date to retrieve and process.
    end: pandas.Timestamp
        Last refrence date to retrieve and process.
    routelink: polars.DataFrame
        Crosswalk from NWM channel feature IDs to USGS site codes.
    root: pathlib.Path
        Root data directory.
    jobs: int, optional, default 1
        Number of parallel process used to process NWM output.
    retries: int, optional, default 3
        Number of times to retry NetCDF file downloads.
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # List of dates to retrieve
    reference_dates = pd.date_range(
        start=start.floor("1d"),
        end=end.ceil("1d"),
        freq="1d"
    )

    # Prepare downloads and output file paths
    urls: list[str] = []
    ofiles: list[Path] = []
    for state in STATE_LIST:
        # Generate file path and check for existence
        state_code = state.abbr.lower()

        # Process each reference day
        for rd in reference_dates:
            # Output file
            ofile = build_usgs_filepath(root, state_code, rd)

            # Skip existing files
            if ofile.exists():
                logger.info("Skipping existing file %s", ofile)
                continue

            logger.info("Preparing to build %s", ofile)
            ofiles.append(ofile)
            ofile.parent.mkdir(exist_ok=True, parents=True)
            urls.append(generate_usgs_url(rd, state_code))

    # Temporary directory
    odir = root / "temp"
    odir.mkdir(exist_ok=True)
    logger.info("Downloading to %s", odir)

    # Download and process
    with TemporaryDirectory(
        prefix="usgs_",
        dir=odir
        ) as td:
        for ofile, url in zip(ofiles, urls):
            # Download file
            json_file = Path(td) / str(ofile).replace(
                "/", "_").replace("=", "_").replace("parquet", "json")

            logger.info("Downloading %s", json_file)
            download_files(
                (url, json_file),
                limit=1,
                timeout=3600, 
                headers={"Accept-Encoding": "gzip"},
                file_validator=json_validator,
                retries=retries
            )

            # Process
            logger.info("Building %s", ofile)
            process_json(JSONJob(json_file, ofile))
            return

def scan_usgs(root: Path) -> pl.LazyFrame:
    """
    Return polars.LazyFrame of USGS observations.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    """
    return pl.scan_parquet(
        root / f"{SUBDIRECTORY}/",
        hive_schema={
            "state_code": pl.Enum([s.abbr.lower() for s in STATE_LIST]),
            "year": pl.Int32,
            "month": pl.Int32
        }
    )
