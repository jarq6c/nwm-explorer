"""Compute standard evaluation metrics."""
from pathlib import Path
import inspect

import polars as pl
import pandas as pd
import numpy as np

from nwm_explorer.logging.logger import get_logger
from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration
from nwm_explorer.data.nwm import generate_reference_dates, build_nwm_file_details
from nwm_explorer.data.usgs import get_usgs_reader

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

    # Read routelinks
    logger.info("Reading routelinks")
    crosswalk = {d: df.select(["usgs_site_code", "nwm_feature_id"]).collect() for d, df in routelinks.items()}

    # File details
    logger.info("Generating file details")
    file_details = build_nwm_file_details(root, reference_dates)

    # Download
    logger.info("Pairing NWM data")
    for fd in file_details:
        if fd.path.exists():
            logger.info(f"Loading {fd.path}")
            sim = pl.read_parquet(fd.path)
            first = sim["value_time"].min()
            last = sim["value_time"].max()

            logger.info("Loading observations")
            obs = get_usgs_reader(root, fd.domain, first, last).collect()
            xwalk = crosswalk[fd.domain]

            logger.info(f"Resampling {fd.path}")
            if fd.configuration in PREDICTION_RESAMPLING:
                sampling_duration = PREDICTION_RESAMPLING[fd.configuration][0]
                resampling_frequency = PREDICTION_RESAMPLING[fd.configuration][1]
                hours = sampling_duration / pl.duration(hours=1)
                sim = sim.sort(
                    ("nwm_feature_id", "reference_time", "value_time")
                ).with_columns(
                    ((pl.col("value_time").sub(
                        pl.col("reference_time")
                        ) / sampling_duration).floor() *
                            hours).alias("lead_time_hours_min")
                ).group_by_dynamic(
                    "value_time",
                    every=resampling_frequency,
                    group_by=("nwm_feature_id", "reference_time")
                ).agg(
                    pl.col("predicted").max(),
                    pl.col("lead_time_hours_min").min()
                )
            else:
                # NOTE This will result in two simulation values per
                #  reference day. Handle this before computing metrics (max).
                sim = sim.sort(
                    ("nwm_feature_id", "value_time")
                ).group_by_dynamic(
                    "value_time",
                    every="1d",
                    group_by="nwm_feature_id"
                ).agg(
                    pl.col("predicted").max()
                )
            print(sim.head())
            break
