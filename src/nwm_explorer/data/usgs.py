"""Retrieve and organize USGS streamflow observations."""
from pathlib import Path
from typing import TypedDict
import inspect

import us
import pandas as pd
import polars as pl
from hydrotools.nwis_client.iv import IVDataService

from nwm_explorer.data.mapping import ModelDomain
from nwm_explorer.logging.logger import get_logger

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

class DownloadParameters(TypedDict):
    """Typed dict containing parameters for IVDataService.get."""
    stateCd: str
    startDT: pd.Timestamp
    endDT: pd.Timestamp

def generate_usgs_download_parameters(
        date: pd.Timestamp,
        stateCd: str
) -> DownloadParameters:
    """Returns download parameters."""
    return DownloadParameters(
        stateCd=stateCd,
        startDT=date.floor(freq="1d"),
        endDT=date.floor(freq="1d") + pd.Timedelta(hours=23, minutes=59)
    )

def download_usgs_file(
        parameters: DownloadParameters,
        file_path: Path,
        cache_path: Path
) -> None:
    """Download USGS observations and save to parquet."""
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    if file_path.exists():
        logger.info(f"{file_path} exists, skipping download")
        return
    
    logger.info(f"Retrieving {file_path}")
    client = IVDataService(cache_filename=cache_path)
    df = pl.DataFrame(client.get(**parameters))

    logger.info(f"Saving {file_path}")
    file_path.parent.mkdir(exist_ok=True, parents=True)
    df.write_parquet(file_path)

def download_usgs(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    root: Path
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

    # Cache
    nwis_cache = root / "nwisiv_cache.sqlite"

    # Download data by state and reference day
    for rd in reference_dates:
        for s in STATE_LIST:
            # Generate file path and check for existence
            fp = generate_usgs_filepath(root, rd, s.abbr.lower())
            if fp.exists():
                logger.info(f"{fp} exists, skipping download")
                continue

            # Download
            p = generate_usgs_download_parameters(rd, s.abbr.lower())
            download_usgs_file(p, fp, nwis_cache)
    
    # Clean-up cache
    logger.info(f"Cleaning up {nwis_cache}")
    if nwis_cache.exists():
        nwis_cache.unlink()

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
