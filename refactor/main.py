"""Methods to evaluate pairs."""
from pathlib import Path
import inspect
from typing import Generator, Any
from concurrent.futures import ProcessPoolExecutor

import polars as pl
import pandas as pd
import numpy as np
import numpy.typing as npt
from numba import float64, guvectorize

from modules.nwm import ModelConfiguration
from modules.pairs import scan_pairs, GROUP_SPECIFICATIONS
from modules.logger import get_logger
from modules.routelink import download_routelink

@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()")
def nash_sutcliffe_efficiency(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of Nash-Sutcliffe Model Efficiency.
        
    Parameters
    ----------
    y_true: NDArray[np.float64], required
        Ground truth (correct) target values, also called observations,
        measurements, or observed values.
    y_pred: NDArray[np.float64], required
        Estimated target values, also called simulations, forecasts, or modeled values.
    result: NDArray[np.float64], required
        Stores scalar result.
        
    Returns
    -------
    None
    """
    variance = np.sum((y_true - np.mean(y_true)) ** 2.0)
    if variance == 0:
        result[0] = np.nan
        return
    result[0] = 1.0 - np.sum((y_true - y_pred) ** 2.0) / variance

def bootstrap_metrics(
    data: pd.DataFrame,
    # minimum_sample_size: int = 30,
    # minimum_mean: float = 0.01,
    # minimum_variance: float = 0.000025
    ) -> dict[str, Any]:
    """
    Use stationary bootstrap to generate metrics with confidence intervals.
    """
    result = {
        "nwm_feature_id": data["nwm_feature_id"].iloc[0],
        "lead_time_hours_min": data["lead_time_hours_min"].min(),
        "predicted_cfs_min": data["predicted_cfs_min"].min(),
        "predicted_cfs_median": data["predicted_cfs_median"].median(),
        "predicted_cfs_max": data["predicted_cfs_max"].max(),
        "predicted_value_time_min": data["predicted_value_time_min"].min(),
        "predicted_value_time_max": data["predicted_value_time_max"].max(),
        "reference_time_min": data["reference_time_min"].min(),
        "reference_time_max": data["reference_time_max"].max(),
        "usgs_site_code": data["usgs_site_code"].iloc[0],
        "observed_cfs_min": data["observed_cfs_min"].min(),
        "observed_cfs_median": data["observed_cfs_median"].median(),
        "observed_cfs_max": data["observed_cfs_max"].max(),
        "observed_value_time_min": data["observed_value_time_min"].min(),
        "observed_value_time_max": data["observed_value_time_max"].max()
    }
    return result

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
        ) -> Generator[list[pd.DataFrame]]:
    """
    Iteratively, load and group forecast pairs into lead time pools. Returns a
    list of DataFrame for each nwm_feature_id and lead_time_hours_min combination.

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
    Generator[list[pandas.DataFrame]]
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

        # Generate groups
        yield [df.to_pandas() for _, df in data.group_by(["nwm_feature_id", "lead_time_hours_min"])]

def main(
        label: str = "FY2024Q1",
        start_time: pd.Timestamp = pd.Timestamp("2023-10-01"),
        end_time: pd.Timestamp = pd.Timestamp("2023-12-31"),
        processes: int = 18,
        sites_per_chunk: int = 250
) -> None:
    """Main."""
    # Start process pool
    with ProcessPoolExecutor(max_workers=processes) as parallel_computer:
        # Process each configuration
        for config, specs in GROUP_SPECIFICATIONS.items():
            # Process in chunks
            for groups in prediction_pool_generator(
                root=Path("/ised/nwm_explorer_data"),
                configuration=config,
                start_time=start_time,
                end_time=end_time,
                lead_time_interval=specs.window_interval,
                sites_per_chunk=sites_per_chunk
            ):
                # Chunk size
                chunksize = len(groups) // processes + 1

                # Compute
                results = pd.DataFrame.from_records(
                    parallel_computer.map(
                        bootstrap_metrics, groups, chunksize=chunksize
                    )
                )
                print(results)
                break
            break

if __name__ == "__main__":
    main()
