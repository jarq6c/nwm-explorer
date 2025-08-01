"""Retrieve and organize USGS streamflow observations."""
from pathlib import Path
import inspect
import json
from dataclasses import dataclass

import us
import pandas as pd
import polars as pl

from nwm_explorer.data.mapping import ModelDomain
from nwm_explorer.logging.logger import get_logger
from nwm_explorer.data.download import download_files

NWIS_BASE_URL: str = "https://waterservices.usgs.gov/nwis/iv/?format=json&siteStatus=all"
"""NWIS IV API returning json and all site statuses."""

STATE_LIST: list[us.states.State] = us.states.STATES + [us.states.PR]
"""List of US states."""

STATE_DOMAIN: dict[us.states.State, ModelDomain] = {
    us.states.AK: ModelDomain.alaska,
    us.states.HI: ModelDomain.hawaii,
    us.states.PR: ModelDomain.puertorico
}
"""Mapping from US state to NWM domain."""

DOMAIN_STATE_LOOKUP: dict[ModelDomain, us.states.State] = {v: k for k, v in STATE_DOMAIN.items()}
"""Reverse look-up from NWM domain to US state."""

def json_validator(ifile: Path) -> None:
    with ifile.open("r") as fi:
        json.loads(fi.read())

def generate_usgs_filepath(
        root: Path,
        date: pd.Timestamp,
        stateCd: str
) -> Path:
    """Returns a standardized filepath."""
    # Look up state
    state = us.states.lookup(stateCd)

    # Map to model domain
    domain = STATE_DOMAIN.get(state, ModelDomain.conus)

    # Directory name
    directory = date.strftime("usgs.%Y%m%d")

    # File name
    file_name = f"{stateCd}_streamflow_cfs.parquet"

    # Full path
    return root / f"parquet/{domain}/{directory}/{file_name}"

def generate_usgs_url(
        date: pd.Timestamp,
        stateCd: str
) -> str:
    """Returns download parameters."""
    startDT=date.floor(freq="1d").strftime("%Y-%m-%dT%H:%MZ")
    endDT=(date.floor(freq="1d") + pd.Timedelta(hours=23, minutes=59)).strftime("%Y-%m-%dT%H:%MZ")
    return NWIS_BASE_URL + f"&stateCd={stateCd}&startDT={startDT}&endDT={endDT}"

@dataclass
class JSONJob:
    ifile: Path
    ofile: Path

def process_json(job: JSONJob) -> None:
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
        "value": "observed",
        "dateTime": "value_time"
    }).write_parquet(job.ofile)

def download_usgs(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    root: Path,
    retries: int = 10
    ):
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # List of dates to retrieve
    reference_dates = pd.date_range(
        start=startDT.floor("1d"),
        end=endDT.ceil("1d"),
        freq="1d"
    )

    # Setup temp directory
    tdir = root / "temp"
    tdir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Downloading files to {tdir}")

    # Download data by state and reference day
    for rd in reference_dates:
        # Start building download parameters
        urls = []
        temp_files = []
        json_jobs = []
        date_string = rd.strftime("usgs.%Y%m%d")

        # Partition by state
        for s in STATE_LIST:
            # Generate file path and check for existence
            fp = generate_usgs_filepath(root, rd, s.abbr.lower())
            if fp.exists():
                logger.info(f"{fp} exists, skipping download")
                continue

            # Create parents
            fp.parent.mkdir(exist_ok=True, parents=True)

            # Set download parameters
            urls.append(generate_usgs_url(rd, s.abbr.lower()))

            # Processing
            tfile = tdir / f"{date_string}_{s}.json"
            temp_files.append(tfile)
            json_jobs.append(JSONJob(tfile, fp))
        
        # Check for files to download
        if len(urls) == 0:
            continue
    
        # Download
        logger.info(f"Downloading {rd}")
        download_files(
            *list(zip(urls, temp_files)),
            limit=1,
            timeout=3600, 
            headers={"Accept-Encoding": "gzip"},
            file_validator=json_validator,
            retries=retries
        )

        # Process
        logger.info(f"Processing {rd}")
        for j in json_jobs:
            logger.info(f"{j.ifile} -> {j.ofile}")
            process_json(j)

    # Clean-up
    logger.info(f"Cleaning up {tdir}")
    for ofile in tdir.glob("*.json"):
        ofile.unlink()
    tdir.rmdir()

def get_usgs_reader(
    root: Path,
    domain: ModelDomain,
    reference_dates: list[pd.Timestamp]
    ) -> pl.LazyFrame:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)
    
    # Get file path
    logger.info(f"Scanning {domain} {reference_dates[0]} to {reference_dates[-1]}")

    # Look-up state
    state = DOMAIN_STATE_LOOKUP.get(domain, None)

    # Assume CONUS, otherwise
    if state is None:
        state_list = us.states.STATES_CONTIGUOUS
    else:
        state_list = [state]
    
    # Build file paths
    file_paths = []
    for s in state_list:
        for rd in reference_dates:
            file_paths.append(generate_usgs_filepath(root, rd, s.abbr.lower()))

    # Scan
    return pl.scan_parquet([fp for fp in file_paths if fp.exists()])

def get_usgs_readers(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    root: Path
    ) -> dict[ModelDomain, pl.LazyFrame]:
    """Returns mapping from ModelDomain to polars.LazyFrame."""
    # List of dates to retrieve
    reference_dates = pd.date_range(
        start=startDT.floor("1d"),
        end=endDT.ceil("1d"),
        freq="1d"
    ).to_list()

    return {d: get_usgs_reader(root, d, reference_dates) for d in list(ModelDomain)}
