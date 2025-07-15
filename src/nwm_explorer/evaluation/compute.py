"""Compute standard evaluation metrics."""
from pathlib import Path
import inspect

import polars as pl
import pandas as pd

from nwm_explorer.logging.logger import get_logger
from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration
from nwm_explorer.data.nwm import generate_reference_dates, build_nwm_file_details

OBSERVATION_RESAMPLING: dict[ModelDomain, tuple[str]] = {
    ModelDomain.alaska: ("1d", "5h"),
    ModelDomain.conus: ("1d", "6h"),
    ModelDomain.hawaii: ("1d", "6h"),
    ModelDomain.puertorico: ("1d", "6h")
}
"""Mapping used to resample observations."""

PREDICTION_RESAMPLING: dict[ModelConfiguration, tuple[pl.Duration, str]] = {
    ModelConfiguration.medium_range_mem1: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_blend: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_no_da: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_alaska_mem1: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_blend_alaska: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_alaska_no_da: (pl.duration(hours=24), "1d"),
    ModelConfiguration.short_range: (pl.duration(hours=6), "6h"),
    ModelConfiguration.short_range_alaska: (pl.duration(hours=5), "5h"),
    ModelConfiguration.short_range_hawaii: (pl.duration(hours=6), "6h"),
    ModelConfiguration.short_range_hawaii_no_da: (pl.duration(hours=6), "6h"),
    ModelConfiguration.short_range_puertorico: (pl.duration(hours=6), "6h"),
    ModelConfiguration.short_range_puertorico_no_da: (pl.duration(hours=6), "6h")
}
"""Mapping used for computing lead time and sampling frequency."""

def compute_metrics(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    root: Path,
    routelinks: dict[ModelDomain, pl.LazyFrame],
    jobs: int
    ) -> None:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Generate reference dates
    logger.info("Generating reference dates")
    reference_dates = generate_reference_dates(startDT, endDT)

    # Features to extract
    logger.info("Reading routelinks")
    features = {d: df.select("nwm_feature_id").collect()["nwm_feature_id"].to_numpy() for d, df in routelinks.items()}

    # File details
    logger.info("Generating file details")
    file_details = build_nwm_file_details(root, reference_dates, features)

    # Download
    logger.info("Pairing NWM data")
    for fd in file_details:
        if fd.path.exists():
            logger.info(f"Skipping existing file {fd.path}")
            break
