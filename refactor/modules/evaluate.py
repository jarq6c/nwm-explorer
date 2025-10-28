"""Methods to evaluate pairs."""
from pathlib import Path
import inspect
from typing import Generator, Any, Callable, Literal
from concurrent.futures import ProcessPoolExecutor
from enum import StrEnum
import itertools
import functools

import polars as pl
import pandas as pd
import numpy as np
import numpy.typing as npt
from numba import float64, guvectorize
from arch.bootstrap import StationaryBootstrap, optimal_block_length

from .nwm import ModelConfiguration
from .pairs import scan_pairs, GROUP_SPECIFICATIONS
from .logger import get_logger
from .routelink import download_routelink

class Metric(StrEnum):
    """Symbols for common metrics."""
    NASH_SUTCLIFFE_EFFICIENCY = "nash_sutcliffe_efficiency"
    RELATIVE_MEAN_BIAS = "relative_mean_bias"
    PEARSON_CORRELATION_COEFFICIENT = "pearson_correlation_coefficient"
    RELATIVE_MEAN = "relative_mean"
    RELATIVE_MEDIAN = "relative_median"
    RELATIVE_MINIMUM = "relative_minimum"
    RELATIVE_MAXIMUM = "relative_maximum"
    RELATIVE_STANDARD_DEVIATION = "relative_standard_deviation"
    KLING_GUPTA_EFFICIENCY = "kling_gupta_efficiency"

MetricFunction = Callable[[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
    ], None]
"""Type hint for Numba functions that generate metrics."""

LRU_CACHE_SIZE: int = 10
"""Maximum size of functools.lru_cache."""

SUBDIRECTORY: str = "evaluations"
"""Subdirectory that indicates root of evaluation parquet store."""

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

@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()")
def relative_mean_bias(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of signed mean relative bias.
    Also called mean relative error or fractional bias.
        
    Parameters
    ----------
    y_true: NDArray[np.float64], required
        Ground truth (correct) target values, also called observations,
        measurements, or observed values.
    y_pred: NDArray[np.float64], required
        Estimated target values, also called simulations or modeled values.
    result: NDArray[np.float64], required
        Stores scalar result.
        
    Returns
    -------
    None
    """
    total = np.sum(y_true)
    if total == 0.0:
        result[0] = np.nan
        return
    result[0] = np.sum(y_pred - y_true) / np.sum(y_true)

@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()")
def pearson_correlation_coefficient(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of the Pearson correlation
    coefficient.
        
    Parameters
    ----------
    y_true: NDArray[np.float64], required
        Ground truth (correct) target values, also called observations,
        measurements, or observed values.
    y_pred: NDArray[np.float64], required
        Estimated target values, also called simulations or modeled values.
    result: NDArray[np.float64], required
        Stores scalar result.
        
    Returns
    -------
    None
    """
    y_true_dev = y_true - np.mean(y_true)
    y_pred_dev = y_pred - np.mean(y_pred)
    num = np.sum(y_true_dev * y_pred_dev)
    den = (
        np.sqrt(np.sum(y_true_dev ** 2)) *
        np.sqrt(np.sum(y_pred_dev ** 2))
        )
    if den == 0:
        result[0] = np.nan
        return
    result[0] = num / den

@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()")
def relative_standard_deviation(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of relative standard deviation,
    required to compute Kling-Gupta Model Efficiency.
        
    Parameters
    ----------
    y_true: NDArray[np.float64], required
        Ground truth (correct) target values, also called observations,
        measurements, or observed values.
    y_pred: NDArray[np.float64], required
        Estimated target values, also called simulations or modeled values.
    result: NDArray[np.float64], required
        Stores scalar result.
        
    Returns
    -------
    None
    """
    std_dev = np.std(y_true)
    if std_dev == 0:
        result[0] = np.nan
        return
    result[0] = np.std(y_pred) / std_dev

@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()")
def relative_mean(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of relative mean,
    required to compute Kling-Gupta Model Efficiency.
        
    Parameters
    ----------
    y_true: NDArray[np.float64], required
        Ground truth (correct) target values, also called observations,
        measurements, or observed values.
    y_pred: NDArray[np.float64], required
        Estimated target values, also called simulations or modeled values.
    result: NDArray[np.float64], required
        Stores scalar result.
        
    Returns
    -------
    None
    """
    mean = np.mean(y_true)
    if mean == 0:
        result[0] = np.nan
        return
    result[0] = np.mean(y_pred) / mean

@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()")
def relative_median(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of relative mean,
    required to compute Kling-Gupta Model Efficiency.
        
    Parameters
    ----------
    y_true: NDArray[np.float64], required
        Ground truth (correct) target values, also called observations,
        measurements, or observed values.
    y_pred: NDArray[np.float64], required
        Estimated target values, also called simulations or modeled values.
    result: NDArray[np.float64], required
        Stores scalar result.
        
    Returns
    -------
    None
    """
    median = np.median(y_true)
    if median == 0:
        result[0] = np.nan
        return
    result[0] = np.median(y_pred) / median

@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()")
def relative_minimum(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of relative minimum.
        
    Parameters
    ----------
    y_true: NDArray[np.float64], required
        Ground truth (correct) target values, also called observations,
        measurements, or observed values.
    y_pred: NDArray[np.float64], required
        Estimated target values, also called simulations or modeled values.
    result: NDArray[np.float64], required
        Stores scalar result.
        
    Returns
    -------
    None
    """
    minimum = np.min(y_true)
    if minimum == 0:
        result[0] = np.nan
        return
    result[0] = np.min(y_pred) / minimum

@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()")
def relative_maximum(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of relative maximum.
        
    Parameters
    ----------
    y_true: NDArray[np.float64], required
        Ground truth (correct) target values, also called observations,
        measurements, or observed values.
    y_pred: NDArray[np.float64], required
        Estimated target values, also called simulations or modeled values.
    result: NDArray[np.float64], required
        Stores scalar result.
        
    Returns
    -------
    None
    """
    maximum = np.max(y_true)
    if maximum == 0:
        result[0] = np.nan
        return
    result[0] = np.max(y_pred) / maximum

@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()")
def kling_gupta_efficiency(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of Kling-Gupta Model Efficiency.
        
    Parameters
    ----------
    y_true: NDArray[np.float64], required
        Ground truth (correct) target values, also called observations,
        measurements, or observed values.
    y_pred: NDArray[np.float64], required
        Estimated target values, also called simulations or modeled values.
    result: NDArray[np.float64], required
        Stores scalar result.
        
    Returns
    -------
    None
    """
    correlation = np.empty(shape=1, dtype=np.float64)
    pearson_correlation_coefficient(y_true, y_pred, correlation)
    rel_var = np.empty(shape=1, dtype=np.float64)
    relative_standard_deviation(y_true, y_pred, rel_var)
    rel_mean = np.empty(shape=1, dtype=np.float64)
    relative_mean(y_true, y_pred, rel_mean)
    result[0] = (1.0 - np.sqrt(
        ((correlation[0] - 1.0)) ** 2.0 +
        ((rel_var[0] - 1.0)) ** 2.0 +
        ((rel_mean[0] - 1.0)) ** 2.0
        ))

METRIC_FUNCTIONS: dict[Metric, MetricFunction] = {
    Metric.NASH_SUTCLIFFE_EFFICIENCY: nash_sutcliffe_efficiency,
    Metric.RELATIVE_MEAN_BIAS: relative_mean_bias,
    Metric.PEARSON_CORRELATION_COEFFICIENT: pearson_correlation_coefficient,
    Metric.RELATIVE_MEAN: relative_mean,
    Metric.RELATIVE_MEDIAN: relative_median,
    Metric.RELATIVE_MINIMUM: relative_minimum,
    Metric.RELATIVE_MAXIMUM: relative_maximum,
    Metric.RELATIVE_STANDARD_DEVIATION: relative_standard_deviation,
    Metric.KLING_GUPTA_EFFICIENCY: kling_gupta_efficiency
}
"""Mapping from metrics to functions used to compute them."""

def bootstrap_metrics(
    data: pd.DataFrame,
    minimum_sample_size: int = 30,
    minimum_mean: float = 0.01,
    minimum_variance: float = 0.000025
    ) -> dict[str, Any]:
    """
    Use stationary bootstrap to generate metrics with confidence intervals.
    Returns a dictionary. Assumed use is as record for pandas.DataFrame.from_records.

    Parameters
    ----------
    data: pandas.DataFrame, required
        DataFrame of pairs for a specific feature and lead time.
    minimum_sample_size: int, optional, default 30
        Minimum number of samples required to compute confidence intervals.
    minimum_mean: float, optional, default 0.01
        Smallest mean value of observed time series required to compute
        confidence intervals.
    minimum_variance: float, optional, default 0.000025
        Smallest variance of observed time series required to compute
        confidence intervals.
    
    Returns
    -------
    dict[str, Any]
    """
    # Start building results
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
        "observed_value_time_max": data["observed_value_time_max"].max(),
        "sample_size": data["nwm_feature_id"].count()
    }

    # Compute metrics
    for rank in ["min", "median", "max"]:
        # Extract numpy arrays for Numba functions
        y_true = data[f"observed_cfs_{rank}"].to_numpy(dtype=np.float64)
        y_pred = data[f"predicted_cfs_{rank}"].to_numpy(dtype=np.float64)

        # Compute each metric
        for label, func in METRIC_FUNCTIONS.items():
            # Point estimate
            # NOTE Numba has magic that implicitly instantiates point, but
            #  it makes the linter mad.
            point = np.empty(shape=1, dtype=np.float64)
            func(y_true, y_pred, point)
            result[f"{label}_{rank}_point"] = point[0]

            # Sample size too small reliably to compute confidence interval
            if result["sample_size"] < minimum_sample_size:
                result[f"{label}_{rank}_lower"] = np.nan
                result[f"{label}_{rank}_upper"] = np.nan
                continue

            # Values too small to reliably compute confidence interval
            if np.mean(y_true) < minimum_mean:
                result[f"{label}_{rank}_lower"] = np.nan
                result[f"{label}_{rank}_upper"] = np.nan
                continue

            # Variance too small to reliably compute confidence interval
            if np.var(y_true) < minimum_variance:
                result[f"{label}_{rank}_lower"] = np.nan
                result[f"{label}_{rank}_upper"] = np.nan
                continue

            # Optimal block size
            # NOTE Normalizing the values seems to produce more consistent
            #  block sizes. Here we let the "true" values determine the block size.
            max_value = np.max(y_true) * 1.01
            normalized = y_true / max_value
            block_size = optimal_block_length(normalized)["stationary"][0]
            if np.isnan(block_size):
                block_size = 1
            else:
                block_size = max(1, int(block_size))

            # Resample the array index to apply to both y_true and y_pred
            index = np.arange(y_true.size)
            bs = StationaryBootstrap(
                block_size,
                index,
                seed=2025
                )

            # Generate posterior distribution
            posterior = []
            estimate = np.empty(shape=1, dtype=np.float64)
            for samples in bs.bootstrap(1000):
                # Generate an index
                idx = samples[0][0]

                # Apply the new index and compute the metric
                func(y_true[idx], y_pred[idx], estimate)
                posterior.append(estimate[0])

            # Compute confidence interval
            ci = np.quantile(posterior, [0.025, 0.975])
            result[f"{label}_{rank}_lower"] = ci[0]
            result[f"{label}_{rank}_upper"] = ci[1]
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
    sites_per_chunk: int, optional, default 1
        Maximum number of sites returned per iteration.

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
            logger.debug("Empty pool, trying again")
            continue

        # Generate groups
        yield [df.to_pandas() for _, df in data.group_by(["nwm_feature_id", "lead_time_hours_min"])]

def evaluate(
        label: str,
        root: Path,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        processes: int = 1,
        sites_per_chunk: int = 1
) -> None:
    """
    Iteratively, load and group forecast pairs into lead time pools. Returns a
    list of DataFrame for each nwm_feature_id and lead_time_hours_min combination.

    Parameters
    ----------
    label: str, required
        Machine-friendly label used to generate parquet store.
    root: pathlib.Path
        Root data directory.
    start_time: pandas.Timestamp
        First reference time.
    end_time: pandas.Timestamp
        Last reference time.
    process: int, optional, default 1
        Number of parallel processes to use for computation.
    sites_per_chunk: int, optional, default 1
        Maximum number of sites to load into memory at once.

    Returns
    -------
    Generator[list[pandas.DataFrame]]
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Logging
    logger.info("Running evaluation %s", label)
    logger.info("Range: %s to %s", start_time, end_time)

    # Start process pool
    logger.info("Starting %d processes", processes)
    with ProcessPoolExecutor(max_workers=processes) as parallel_computer:
        # Process each configuration
        for config, specs in GROUP_SPECIFICATIONS.items():
            # Prepare output file
            ofile = root / f"{SUBDIRECTORY}/label={label}/configuration={config}/E0.parquet"
            if ofile.exists():
                logger.info("Found %s", ofile)
                continue
            ofile.parent.mkdir(exist_ok=True, parents=True)

            # Process in chunks
            logger.info("Evaluating %s", config)
            logger.info("Grouping into chunks of %d sites", sites_per_chunk)
            batch_counter = itertools.count(1)
            dataframes = []
            for groups in prediction_pool_generator(
                root=root,
                configuration=config,
                start_time=start_time,
                end_time=end_time,
                lead_time_interval=specs.window_interval,
                sites_per_chunk=sites_per_chunk
            ):
                # Chunk size
                logger.info("Batch: %d", next(batch_counter))
                chunksize = len(groups) // processes + 1
                logger.info("Evaluating %d groups", len(groups))
                logger.info("Running %d groups per process", chunksize)

                # Compute and collect results
                logger.info("Computing metrics")
                dataframes.append(
                    pd.DataFrame.from_records(
                        parallel_computer.map(
                            bootstrap_metrics, groups, chunksize=chunksize
                        )
                    )
                )

            # Concatenate into a single dataframe
            logger.info("Concatenating groups")
            results = pd.concat(dataframes, ignore_index=True)

            # Save results
            logger.info("Saving %s", ofile)
            pl.DataFrame(results).write_parquet(ofile)

def scan_evaluations_no_cache(root: Path) -> pl.LazyFrame:
    """
    Return polars.LazyFrame of Evaluation results.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.

    Returns
    -------
    polars.LazyFrame
    """
    return pl.scan_parquet(
        root / f"{SUBDIRECTORY}/",
        hive_schema={
            "label": pl.String,
            "configuration": pl.Enum(ModelConfiguration)
        }
    )

@functools.lru_cache(LRU_CACHE_SIZE)
def scan_evaluations_cache(root: Path) -> pl.LazyFrame:
    """
    Return polars.LazyFrame of Evaluation results. Cache LazyFrame.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.

    Returns
    -------
    polars.LazyFrame
    """
    return scan_evaluations_no_cache(root)

def scan_evaluations(root: Path, cache: bool = False) -> pl.LazyFrame:
    """
    Return polars.LazyFrame of Evaluation results. Optionally, cache LazyFrame.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    cache: bool, optional, default False
        If True, caches LazyFrame.

    Returns
    -------
    polars.LazyFrame
    """
    if cache:
        return scan_evaluations_cache(root)
    return scan_evaluations_no_cache(root)

def load_metrics_no_cache(
        root: Path,
        label: str,
        configuration: ModelConfiguration,
        metric: Metric,
        lead_time_hours_min: int = 0,
        rank: Literal["min", "median", "max"] = "median"
) -> pl.DataFrame:
    """
    Returns DataFrame of metrics.

    Parameters
    ----------
    root: pathlib.Path
        Root of data directory.
    label: str
        Label of evaluation.
    configuration: ModelConfiguration
        National Water Model configuration.
    metric: Metric
        Evaluation metric.
    lead_time_hours_min: int, optional, default 0
        Minimum lead time in hours.
    rank: Literal["min", "median", "max"], optional, default "median"
        Aggregated function applied to predictions and observations. Varies by
        configuration. Analysis & Assimilation and Medium Range configurations
        will typically return daily streamflow aggregated to minimum, median, or
        maximum streamflow. Short Range CONUS, Hawaii, and Puerto Rico return
        6-hourly aggregation. Short Range Alaska returns 5-hourly aggregations.
    
    Returns
    -------
    polars.DataFrame
    """
    return scan_evaluations(
        root
    ).filter(
        pl.col("label") == label,
        pl.col("configuration") == configuration,
        pl.col("lead_time_hours_min") == lead_time_hours_min
    ).select(
        [
            "nwm_feature_id",
            f"{metric}_{rank}_lower",
            f"{metric}_{rank}_point",
            f"{metric}_{rank}_upper"
        ]
    ).collect()

@functools.lru_cache(LRU_CACHE_SIZE)
def load_metrics_cache(
        root: Path,
        label: str,
        configuration: ModelConfiguration,
        metric: Metric,
        lead_time_hours_min: int = 0,
        rank: Literal["min", "median", "max"] = "median"
) -> pl.DataFrame:
    """
    Returns DataFrame of metrics. Cache result.

    Parameters
    ----------
    root: pathlib.Path
        Root of data directory.
    label: str
        Label of evaluation.
    configuration: ModelConfiguration
        National Water Model configuration.
    metric: Metric
        Evaluation metric.
    lead_time_hours_min: int, optional, default 0
        Minimum lead time in hours.
    rank: Literal["min", "median", "max"], optional, default "median"
        Aggregated function applied to predictions and observations. Varies by
        configuration. Analysis & Assimilation and Medium Range configurations
        will typically return daily streamflow aggregated to minimum, median, or
        maximum streamflow. Short Range CONUS, Hawaii, and Puerto Rico return
        6-hourly aggregation. Short Range Alaska returns 5-hourly aggregations.
    
    Returns
    -------
    polars.DataFrame
    """
    return scan_evaluations(
        root, cache=True
    ).filter(
        pl.col("label") == label,
        pl.col("configuration") == configuration,
        pl.col("lead_time_hours_min") == lead_time_hours_min
    ).select(
        [
            "nwm_feature_id",
            f"{metric}_{rank}_lower",
            f"{metric}_{rank}_point",
            f"{metric}_{rank}_upper"
        ]
    ).collect()

def load_metrics(
        root: Path,
        label: str,
        configuration: ModelConfiguration,
        metric: Metric,
        lead_time_hours_min: int = 0,
        rank: Literal["min", "median", "max"] = "median",
        cache: bool = False
) -> pl.DataFrame:
    """
    Returns DataFrame of metrics. Cache result.

    Parameters
    ----------
    root: pathlib.Path
        Root of data directory.
    label: str
        Label of evaluation.
    configuration: ModelConfiguration
        National Water Model configuration.
    metric: Metric
        Evaluation metric.
    lead_time_hours_min: int, optional, default 0
        Minimum lead time in hours.
    rank: Literal["min", "median", "max"], optional, default "median"
        Aggregated function applied to predictions and observations. Varies by
        configuration. Analysis & Assimilation and Medium Range configurations
        will typically return daily streamflow aggregated to minimum, median, or
        maximum streamflow. Short Range CONUS, Hawaii, and Puerto Rico return
        6-hourly aggregation. Short Range Alaska returns 5-hourly aggregations.
    cache: bool, optional, default False
        If true, cache result.
    
    Returns
    -------
    polars.DataFrame
    """
    if cache:
        return load_metrics_cache(
            root=root,
            label=label,
            configuration=configuration,
            metric=metric,
            lead_time_hours_min=lead_time_hours_min,
            rank=rank
        )
    return load_metrics_no_cache(
        root=root,
        label=label,
        configuration=configuration,
        metric=metric,
        lead_time_hours_min=lead_time_hours_min,
        rank=rank
    )
