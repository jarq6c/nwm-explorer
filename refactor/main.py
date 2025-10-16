"""Methods to pair predictions with observations."""
from pathlib import Path
from dataclasses import dataclass
import functools
import inspect

import pandas as pd
import polars as pl

from modules.routelink import download_routelink
from modules.nwm import scan_nwm, ModelConfiguration
from modules.usgs import scan_usgs, STATE_LIST
from modules.logger import get_logger

LRU_CACHE_SIZE: int = 60
"""Maximum size of functools.lru_cache."""

SUBDIRECTORY: str = "pairs"
"""Subdirectory that indicates root of pairs parquet store."""

@dataclass
class NWMGroupSpecification:
    """
    A dataclass that holds specifications for time-based polars groupby
    operations.
    """
    index_column: str = "value_time"
    group_by_columns: list[str] | None = None
    select_columns: list[str] | None = None
    sort_columns: list[str] | None = None
    window_interval: str = "1d"
    state_code: str | None = None

    def __post_init__(self) -> None:
        if self.group_by_columns is None:
            self.group_by_columns = ["nwm_feature_id", "reference_time"]

        if self.select_columns is None:
            self.select_columns = self.group_by_columns + [self.index_column, "predicted_cfs"]

        if self.sort_columns is None:
            self.sort_columns = self.group_by_columns + [self.index_column]

GROUP_SPECIFICATIONS: dict[ModelConfiguration, NWMGroupSpecification] = {
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_ALASKA_NO_DA: NWMGroupSpecification(
        group_by_columns=["nwm_feature_id"],
        state_code="ak",
        select_columns=["nwm_feature_id", "reference_time", "value_time", "predicted_cfs"]
    ),
    ModelConfiguration.MEDIUM_RANGE_ALASKA_MEM_1: NWMGroupSpecification(
        state_code="ak"
    ),
    ModelConfiguration.MEDIUM_RANGE_BLEND_ALASKA: NWMGroupSpecification(
        state_code="ak"
    ),
    ModelConfiguration.MEDIUM_RANGE_ALASKA_NO_DA: NWMGroupSpecification(
        state_code="ak"
    ),
    ModelConfiguration.SHORT_RANGE_ALASKA: NWMGroupSpecification(
        window_interval="5h",
        state_code="ak"
    ),
    ModelConfiguration.ANALYSIS_ASSIM_HAWAII_NO_DA: NWMGroupSpecification(
        group_by_columns=["nwm_feature_id"],
        state_code="hi",
        select_columns=["nwm_feature_id", "reference_time", "value_time", "predicted_cfs"]
    ),
    ModelConfiguration.SHORT_RANGE_HAWAII: NWMGroupSpecification(
        window_interval="6h",
        state_code="hi"
    ),
    ModelConfiguration.SHORT_RANGE_HAWAII_NO_DA: NWMGroupSpecification(
        window_interval="6h",
        state_code="hi"
    ),
    ModelConfiguration.ANALYSIS_ASSIM_PUERTO_RICO_NO_DA: NWMGroupSpecification(
        group_by_columns=["nwm_feature_id"],
        state_code="pr",
        select_columns=["nwm_feature_id", "reference_time", "value_time", "predicted_cfs"]
    ),
    ModelConfiguration.SHORT_RANGE_PUERTO_RICO: NWMGroupSpecification(
        window_interval="6h",
        state_code="pr"
    ),
    ModelConfiguration.SHORT_RANGE_PUERTO_RICO_NO_DA: NWMGroupSpecification(
        window_interval="6h",
        state_code="pr"
    ),
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_NO_DA: NWMGroupSpecification(
        group_by_columns=["nwm_feature_id"],
        select_columns=["nwm_feature_id", "reference_time", "value_time", "predicted_cfs"]
    ),
    ModelConfiguration.MEDIUM_RANGE_MEM_1: NWMGroupSpecification(),
    ModelConfiguration.MEDIUM_RANGE_BLEND: NWMGroupSpecification(),
    ModelConfiguration.MEDIUM_RANGE_NO_DA: NWMGroupSpecification(),
    ModelConfiguration.SHORT_RANGE: NWMGroupSpecification(
        window_interval="6h"
    )
}
"""Mapping from ModelConfiguration to group-by specifications."""

@functools.lru_cache(LRU_CACHE_SIZE)
def routelink_cache(root: Path) -> pl.DataFrame:
    """Scan and collect routelink."""
    return download_routelink(root).select(
        ["nwm_feature_id", "usgs_site_code"]
    ).collect()

@functools.lru_cache(LRU_CACHE_SIZE)
def observations_cache(root: Path) -> pl.LazyFrame:
    """Scan observations."""
    return scan_usgs(root)

@functools.lru_cache(LRU_CACHE_SIZE)
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
        End date. pandas.Timestamp compatible
    window_interval: str
        Time interval to aggregate over.

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

@functools.lru_cache(LRU_CACHE_SIZE)
def load_and_map_observations(
        root: Path,
        start_time: str,
        end_time: str,
        window_interval: str
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
    window_interval: str
        Time interval to aggregate over.

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
                every=window_interval,
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
    return root / SUBDIRECTORY / config / year / month / partition

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
                every=specs.window_interval,
                group_by=specs.group_by_columns
            ).agg(
                pl.col("predicted_cfs").min().alias("predicted_cfs_min"),
                pl.col("predicted_cfs").median().alias("predicted_cfs_median"),
                pl.col("predicted_cfs").max().alias("predicted_cfs_max"),
                pl.col("value_time").min().alias("predicted_value_time_min"),
                pl.col("value_time").max().alias("predicted_value_time_max")
            )

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
            # pairs.write_parquet(ofile)
            print(pairs)
        break

if __name__ == "__main__":
    mroot = Path("/ised/nwm_explorer_data")

    pair_nwm_usgs(
        root = mroot,
        start_date = pd.Timestamp("2023-10-01"),
        end_date = pd.Timestamp("2025-09-30")
    )

    # mpairs = pl.scan_parquet(
    #     mroot / f"{SUBDIRECTORY}/",
    #     hive_schema={
    #         "configuration": pl.Enum(ModelConfiguration),
    #         "year": pl.Int32,
    #         "month": pl.Int32
    #     }
    # )
    # print(mpairs.tail().collect())
