"""Methods to pair predictions with observations."""
from pathlib import Path
from dataclasses import dataclass
import functools
import inspect

import pandas as pd
import polars as pl

from modules.routelink import download_routelink
from modules.nwm import scan_nwm, ModelConfiguration
from modules.usgs import scan_usgs
from modules.logger import get_logger

LRU_CACHE_SIZE: int = 10
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
    window_interval: str = "1d"
    state_code: str | None = None

    def __post_init__(self) -> None:
        if self.group_by_columns is None:
            self.group_by_columns = ["nwm_feature_id", "reference_time"]

    @property
    def select_columns(self) -> list[str]:
        """A list of columns to select."""
        return self.group_by_columns + [self.index_column, "predicted_cfs"]

    @property
    def sort_columns(self) -> list[str]:
        """A list of columns to sort by."""
        return self.group_by_columns + [self.index_column]

GROUP_SPECIFICATIONS: dict[ModelConfiguration, NWMGroupSpecification] = {
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_ALASKA_NO_DA: NWMGroupSpecification(
        group_by_columns=["nwm_feature_id"],
        state_code="ak"
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
        state_code="hi"
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
        state_code="pr"
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
        group_by_columns=["nwm_feature_id"]
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
def load_and_map_observations_by_state(
        root: Path,
        state_code: str,
        year: int,
        month: int,
        window_interval: str
) -> pl.DataFrame:
    """
    Return polars.DataFrame of USGS observations. Cache DataFrame.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    state_code: str
        Two character lower case state abbreviation.
    year: int
        Integer year.
    month: int
        Integer month.
    
    Returns
    -------
    polars.DataFrame
    """
    # Load routelink
    routelink = download_routelink(root).select(
        ["nwm_feature_id", "usgs_site_code"]
    ).collect()

    # Collect observations
    return scan_usgs(root
        ).filter(
            pl.col("state_code") == state_code,
            pl.col("year") == year,
            pl.col("month") == month,
            pl.col("usgs_site_code").is_in(routelink["usgs_site_code"].implode())
        ).select(
            ["usgs_site_code", "value_time", "observed_cfs"]
        ).collect().unique(
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

@functools.lru_cache(LRU_CACHE_SIZE)
def load_and_map_observations_conus(
        root: Path,
        year: int,
        month: int,
        window_interval: str
) -> pl.DataFrame:
    """
    Return polars.DataFrame of USGS observations. Cache DataFrame.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    state_code: str
        Two character lower case state abbreviation.
    year: int
        Integer year.
    month: int
        Integer month.
    
    Returns
    -------
    polars.DataFrame
    """
    # Load routelink
    routelink = download_routelink(root).select(
        ["nwm_feature_id", "usgs_site_code"]
    ).collect()

    # Collect observations
    return scan_usgs(root
        ).filter(
            ~pl.col("state_code").is_in(["ak", "hi", "pr"]),
            pl.col("year") == year,
            pl.col("month") == month,
            pl.col("usgs_site_code").is_in(routelink["usgs_site_code"].implode())
        ).select(
            ["usgs_site_code", "value_time", "observed_cfs"]
        ).collect().unique(
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

def main(
        root: Path = Path("/ised/nwm_explorer_data"),
        start_date: pd.Timestamp = pd.Timestamp("2023-10-01"),
        end_date: pd.Timestamp = pd.Timestamp("2025-09-30")
        ) -> None:
    """Main."""
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
    for reference_date in date_range:
        for config, specs in GROUP_SPECIFICATIONS.items():
            # Check for ofile
            ofile = build_pairs_filepath(
                root, config, reference_date
            )
            if ofile.exists():
                logger.info("Skipping existing: %s", ofile)
                continue
            logger.info("Building %s", ofile)

            # Observations
            logger.info("Resampling observations")
            if specs.state_code is None:
                obs = load_and_map_observations_conus(
                        root=root,
                        year=reference_date.year,
                        month=reference_date.month,
                        window_interval=specs.window_interval
                    )
            else:
                obs = load_and_map_observations_by_state(
                        root=root,
                        state_code=specs.state_code,
                        year=reference_date.year,
                        month=reference_date.month,
                        window_interval=specs.window_interval
                    )

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

            # Pair and save
            logger.info("Pairing observations and predictions")
            ofile.parent.mkdir(exist_ok=True, parents=True)
            pairs = pred.join(obs, on=["nwm_feature_id", "value_time"],
                how="left").drop_nulls()

            # Save
            logger.info("Saving %s", ofile)
            pairs.write_parquet(ofile)

if __name__ == "__main__":
    main()
