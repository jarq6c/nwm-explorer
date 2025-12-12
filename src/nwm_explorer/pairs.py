"""Methods to pair predictions with observations."""
from pathlib import Path
import functools
import inspect

import pandas as pd
import polars as pl

from nwm_explorer.routelink import download_routelink
from nwm_explorer.nwm import scan_nwm
from nwm_explorer.usgs import scan_usgs, STATE_LIST
from nwm_explorer.logger import get_logger
from nwm_explorer.constants import (ModelConfiguration, LRU_CACHE_SIZES, SUBDIRECTORIES,
    GROUP_SPECIFICATIONS)

class PairingError(Exception):
    """Custom exception raised for pairing errors."""
    def __init__(self, message: str = "Pairing error occured."):
        self.message = message
        super().__init__(self.message)

@functools.lru_cache(LRU_CACHE_SIZES["pairs"])
def routelink_cache(root: Path) -> pl.DataFrame:
    """Scan and collect routelink."""
    return download_routelink(root).select(
        ["nwm_feature_id", "usgs_site_code"]
    ).collect()

@functools.lru_cache(LRU_CACHE_SIZES["pairs"])
def observations_cache(root: Path) -> pl.LazyFrame:
    """Scan observations."""
    return scan_usgs(root)

@functools.lru_cache(LRU_CACHE_SIZES["pairs"])
def load_and_map_observations_chunk(
        root: Path,
        state_code: str,
        year: int,
        month: int
) -> pl.DataFrame:
    """
    Return polars.DataFrame of USGS observations. Add nwm_feature_id column.
    Cache DataFrame.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    start_time: str
        Start date. pandas.Timestamp compatible.
    end_time: str
        End date. pandas.Timestamp compatible.

    Returns
    -------
    polars.DataFrame
    """
    # Load routelink
    routelink = routelink_cache(root)

    # Scan observations
    observations = observations_cache(root)

    # Collect observations
    return observations.filter(
        pl.col("state_code") == state_code,
        pl.col("year") == year,
        pl.col("month") == month,
        pl.col("usgs_site_code").is_in(routelink["usgs_site_code"].implode())
    ).select(
        ["usgs_site_code", "value_time", "observed_cfs"]
    ).collect()

@functools.lru_cache(LRU_CACHE_SIZES["pairs"])
def load_and_map_observations(
        root: Path,
        start_time: str,
        end_time: str,
        window_interval: int
) -> pl.DataFrame:
    """
    Return polars.DataFrame of USGS observations. Add nwm_feature_id column.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    start_time: str
        Start date. pandas.Timestamp compatible.
    end_time: str
        End date. pandas.Timestamp compatible
    window_interval: int
        Time interval to aggregate over in hours.

    Returns
    -------
    polars.DataFrame
    """
    # Load routelink
    routelink = routelink_cache(root)

    # Expand range
    first = pd.Timestamp(start_time).to_period("M").start_time
    last = pd.Timestamp(end_time).to_period("M").start_time

    # Check end month
    if first == last:
        last += pd.Timedelta("31D")
        last = last.to_period("M").start_time

    # Load data
    dataframes = []
    for m in pd.date_range(first, last, freq="1MS"):
        for state_code in [s.abbr.lower() for s in STATE_LIST]:
            # Collect observations
            df = load_and_map_observations_chunk(
                root, state_code, m.year, m.month
            ).unique(
                ["usgs_site_code", "value_time"]
            ).sort(
                ["usgs_site_code", "value_time"]
            ).group_by_dynamic(
                "value_time",
                every=f"{window_interval}h",
                group_by="usgs_site_code"
            ).agg(
                pl.col("observed_cfs").min().alias("observed_cfs_min"),
                pl.col("observed_cfs").median().alias("observed_cfs_median"),
                pl.col("observed_cfs").max().alias("observed_cfs_max"),
                pl.col("value_time").min().alias("observed_value_time_min"),
                pl.col("value_time").max().alias("observed_value_time_max")
            ).with_columns(
                nwm_feature_id=pl.col("usgs_site_code").replace_strict(
                    routelink["usgs_site_code"].implode(),
                    routelink["nwm_feature_id"].implode()
                )
            )
            dataframes.append(df)

    # Concatenate
    return pl.concat(dataframes).filter(
        pl.col("value_time") >= start_time,
        pl.col("value_time") <= end_time
    )

def build_pairs_filepath(
    root: Path,
    configuration: ModelConfiguration,
    reference_date: pd.Timestamp
    ) -> Path:
    """Build and return hive-partitioned parquet file path."""
    config = f"configuration={configuration}"
    year = f"year={reference_date.year}"
    month = f"month={reference_date.month}"
    partition = f"P{reference_date.day}.parquet"
    subdirectory = SUBDIRECTORIES["pairs"]
    return root / subdirectory / config / year / month / partition

def pair_nwm_usgs(
        root: Path,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
        ) -> None:
    """
    Resample and pair NWM output to USGS observations.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    start_date: pandas.Timestamp
        First reference date to pair.
    end_date: int
        Last reference date to pair.
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Check for minimum data
    period = end_date - start_date
    if period < pd.Timedelta("30d"):
        logger.info("Cannot pair %s to %s", str(start_date), str(end_date))
        raise PairingError("Minimum period to pair is 30 days")

    # Pair
    prediction_store = scan_nwm(root)
    date_range = pd.date_range(
        start=start_date,
        end=end_date,
        freq="1MS"
    )
    logger.info("Pairing %s to %s", str(start_date), str(end_date))
    # Process each reference month
    for reference_date in date_range:
        # Process each configuration
        for config, specs in GROUP_SPECIFICATIONS.items():
            # Check for ofile
            ofile = build_pairs_filepath(
                root, config, reference_date
            )
            if ofile.exists():
                logger.info("Skipping existing: %s", ofile)
                continue
            logger.info("Building %s", ofile)

            # Aggregations
            aggregations = [
                pl.col("predicted_cfs").min().alias("predicted_cfs_min"),
                pl.col("predicted_cfs").median().alias("predicted_cfs_median"),
                pl.col("predicted_cfs").max().alias("predicted_cfs_max"),
                pl.col("value_time").min().alias("predicted_value_time_min"),
                pl.col("value_time").max().alias("predicted_value_time_max"),
            ]
            if "reference_time" not in specs.group_by_columns:
                aggregations.append(pl.col("reference_time").min())

            # Predictions
            logger.info("Resampling predictions")
            pred = prediction_store.filter(
                pl.col("configuration") == config,
                pl.col("year") == reference_date.year,
                pl.col("month") == reference_date.month
            ).select(
                specs.select_columns
            ).collect().sort(
                specs.sort_columns
            ).group_by_dynamic(
                specs.index_column,
                every=f"{specs.window_interval}h",
                group_by=specs.group_by_columns
            ).agg(*aggregations)

            # Check for data
            if pred.is_empty():
                logger.info("Found no predictions")
                continue

            # Get date range
            first = pred["value_time"].min()
            last = pred["value_time"].max()

            # Observations
            logger.info("Resampling observations")
            obs = load_and_map_observations(
                    root=root,
                    start_time=first,
                    end_time=last,
                    window_interval=specs.window_interval
                )

            # Pair and save
            logger.info("Pairing observations and predictions")
            ofile.parent.mkdir(exist_ok=True, parents=True)
            pairs = pred.join(obs, on=["nwm_feature_id", "value_time"],
                how="left").drop_nulls()

            # Save
            logger.info("Saving %s", ofile)
            pairs.write_parquet(ofile)

def scan_pairs_no_cache(root: Path) -> pl.LazyFrame:
    """
    Return polars.LazyFrame of paired NWM predictions and USGS observations.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.

    Returns
    -------
    polars.LazyFrame
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    logger.info("Scanning pairs")
    subdirectory = SUBDIRECTORIES["pairs"]
    return pl.scan_parquet(
        root / f"{subdirectory}/",
        hive_schema={
            "configuration": pl.Enum(ModelConfiguration),
            "year": pl.Int32,
            "month": pl.Int32
        }
    )

@functools.lru_cache(LRU_CACHE_SIZES["pairs"])
def scan_pairs_cache(root: Path) -> pl.LazyFrame:
    """
    Return polars.LazyFrame of paired NWM predictions and USGS observations.
    Cache result.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.

    Returns
    -------
    polars.LazyFrame
    """
    return scan_pairs_no_cache(root)

def scan_pairs(root: Path, cache: bool = False) -> pl.LazyFrame:
    """
    Return polars.LazyFrame of paired NWM predictions and USGS observations.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    cache: bool, optional, default False
        Whether to cache the resulting LazyFrame for subsequent calls.

    Returns
    -------
    polars.LazyFrame
    """
    if cache:
        return scan_pairs_cache(root)
    return scan_pairs_no_cache(root)
