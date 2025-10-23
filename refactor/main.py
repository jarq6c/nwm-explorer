"""Methods to evaluate pairs."""
from pathlib import Path
import inspect
from typing import Generator

import polars as pl
import pandas as pd
import numpy as np
import numpy.typing as npt

from modules.nwm import ModelConfiguration
from modules.pairs import scan_pairs, GROUP_SPECIFICATIONS
from modules.logger import get_logger
from modules.routelink import download_routelink

def load_pool(
        root: Path,
        configuration: ModelConfiguration,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        lead_time_interval: int,
        features: npt.NDArray[np.int64]
        ) -> pl.DataFrame:
    """
    Load and group forecast pairs into lead time pools.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    configuration: ModelConfiguration
        NWM Model configuration.
    start_time: pandas.Timestamp
        First reference time.
    end_time: pandas.Timestamp
        Last reference time.
    lead_time_interval: int
        Lead time scale to aggregate over in hours.
    features: numpy.ndarray[int64], required
        Array of channel features to retrieve from parquet store.
    
    Returns
    -------
    polars.DataFrame
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Load data
    logger.info("Loading pairs")
    dataframes = []
    for m in pd.date_range(start_time, end_time, freq="1MS"):
        dataframes.append(
            scan_pairs(root, cache=True).filter(
                pl.col("configuration") == configuration,
                pl.col("year") == m.year,
                pl.col("month") == m.month,
                pl.col("reference_time") >= start_time,
                pl.col("reference_time") <= end_time,
                pl.col("nwm_feature_id").is_in(features)
            ).collect()
        )

    # Compute lead time and pool pairs
    logger.info("Pooling pairs")
    return pl.concat(
        dataframes
    ).with_columns(
        lead_time_hours_min=(pl.col("predicted_value_time_min").sub(pl.col("reference_time")) /
            pl.duration(hours=lead_time_interval)).floor().cast(pl.Int32).mul(lead_time_interval)
    ).sort([
        "configuration",
        "nwm_feature_id",
        "lead_time_hours_min",
        "value_time"]
    ).group_by_dynamic(
        "value_time",
        every=f"{lead_time_interval}h",
        group_by=["configuration", "nwm_feature_id", "lead_time_hours_min"]
    ).agg(
        pl.col("predicted_cfs_min").min(),
        pl.col("predicted_cfs_median").median(),
        pl.col("predicted_cfs_max").max(),
        pl.col("predicted_value_time_min").min(),
        pl.col("predicted_value_time_max").max(),
        pl.col("reference_time").min().alias("reference_time_min"),
        pl.col("reference_time").max().alias("reference_time_max"),
        pl.col("usgs_site_code").first(),
        pl.col("observed_cfs_min").min(),
        pl.col("observed_cfs_median").median(),
        pl.col("observed_cfs_max").max(),
        pl.col("observed_value_time_min").min(),
        pl.col("observed_value_time_max").max()
    )

def prediction_pool_generator(
        root: Path,
        configuration: ModelConfiguration,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        lead_time_interval: int,
        sites_per_chunk: int = 1
        ) -> Generator[pd.DataFrame]:
    """
    Iteratively, load and group forecast pairs into lead time pools

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    configuration: ModelConfiguration
        NWM Model configuration.
    start_time: pandas.Timestamp
        First reference time.
    end_time: pandas.Timestamp
        Last reference time.
    lead_time_interval: int
        Lead time scale to aggregate over in hours.
    
    Returns
    -------
    Generator[pandas.DataFrame]
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Generate feature list
    feature_list = download_routelink(
        root
    ).select(
        "nwm_feature_id"
    ).collect()["nwm_feature_id"].to_numpy()

    # Feature chunks
    number_of_chunks = len(feature_list) // sites_per_chunk + 1
    feature_chunks = np.array_split(feature_list, number_of_chunks)

    # Yield pool generator
    for features in feature_chunks:
        # Load data
        data = load_pool(
            root=root,
            configuration=configuration,
            start_time=start_time,
            end_time=end_time,
            lead_time_interval=lead_time_interval,
            features=features
        )

        # Check for data
        if data.is_empty():
            logger.info("Empty pool, trying again")
            continue
        else:
            yield data.to_pandas()

def main() -> None:
    """Main."""
    # Process each configuration
    for config, specs in GROUP_SPECIFICATIONS.items():
        if config != ModelConfiguration.MEDIUM_RANGE_MEM_1:
            continue
        print(config)
        # Get a generator
        for df in prediction_pool_generator(
            root=Path("/ised/nwm_explorer_data"),
            configuration=config,
            start_time=pd.Timestamp("2023-10-01"),
            end_time=pd.Timestamp("2025-09-30"),
            lead_time_interval=specs.window_interval,
            sites_per_chunk=250
        ):
            # print(df)
            print(df.info(memory_usage="deep"))
        # df.write_csv("test_eval.csv")
        break

if __name__ == "__main__":
    main()
