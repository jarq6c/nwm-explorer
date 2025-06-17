"""Various methods used to compute model evaluation metrics."""
from typing import Callable
import polars as pl
import numpy as np
import numpy.typing as npt
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from numba import float64, guvectorize

@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()")
def nash_sutcliffe_efficiency(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba and polars compatible implementation of signed mean relative bias.
    Also called, mean relative error or fractional bias.
        
    Parameters
    ----------
    y_true: NDArray[np.float64], required
        Ground truth (correct) target values, also called observations, measurements, or observed values.
    y_pred: NDArray[np.float64], required
        Estimated target values, also called simulations or modeled values.
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
    Numba and polars compatible implementation of signed mean relative bias.
    Also called, mean relative error or fractional bias.
        
    Parameters
    ----------
    y_true: NDArray[np.float64], required
        Ground truth (correct) target values, also called observations, measurements, or observed values.
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
        return np.nan
    result[0] = np.sum(y_pred - y_true) / np.sum(y_true)

@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()")
def pearson_correlation_coefficient(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba and polars compatible implementation of the Pearson correlation
    coefficient.
        
    Parameters
    ----------
    y_true: NDArray[np.float64], required
        Ground truth (correct) target values, also called observations, measurements, or observed values.
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
def relative_variability(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba and polars compatible implementation of relative variability,
    required to compute Kling-Gupta Model Efficiency.
        
    Parameters
    ----------
    y_true: NDArray[np.float64], required
        Ground truth (correct) target values, also called observations, measurements, or observed values.
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
    Numba and polars compatible implementation of relative mean,
    required to compute Kling-Gupta Model Efficiency.
        
    Parameters
    ----------
    y_true: NDArray[np.float64], required
        Ground truth (correct) target values, also called observations, measurements, or observed values.
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
    result[0] = np.std(y_pred) / mean

@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()")
def kling_gupta_efficiency(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """Returns an expression used to add a 'kling_gupta_efficiency' column to a dataframe."""
    correlation = np.array([0.0], dtype=np.float64)
    pearson_correlation_coefficient(y_true, y_pred, correlation)
    rel_var = np.array([0.0], dtype=np.float64)
    relative_variability(y_true, y_pred, rel_var)
    rel_mean = np.array([0.0], dtype=np.float64)
    relative_mean(y_true, y_pred, rel_mean)
    result[0] = (1.0 - np.sqrt(
        ((correlation[0] - 1.0)) ** 2.0 + 
        ((rel_var[0] - 1.0)) ** 2.0 + 
        ((rel_mean[0] - 1.0)) ** 2.0
        ))

METRIC_FUNCTIONS: dict[str, Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = {
    "nse": nash_sutcliffe_efficiency,
    "rmb": relative_mean_bias,
    "pearson": pearson_correlation_coefficient,
    "rel_mean": relative_mean,
    "rel_var": relative_variability,
    "kge": kling_gupta_efficiency
}
"""Metrics to compute."""

SUFFIXES: list[str] = ["", "_lower", "_upper"]
"""Metric field suffixes."""

METRIC_FIELDS: list[str] = [k+s for k in METRIC_FUNCTIONS.keys() for s in SUFFIXES]
"""Metric field names."""

def compute_metrics(
    y_true: pl.Series,
    y_pred: pl.Series,
    minimum_sample_size: int = 30,
    minimum_mean: float = 0.01,
    minimum_variance: float = 0.000025
    ) -> npt.NDArray[np.float64]:
    results = []
    for _, func in METRIC_FUNCTIONS.items():
        # TODO check for nan?

        # Point estimates
        point_estimate = func(
            y_true.to_numpy(), 
            y_pred.to_numpy()
            )

        # Low sample size
        if len(y_true) < minimum_sample_size:
            results.append(np.asarray([point_estimate, np.nan, np.nan]))
            continue
        
        # Mostly zero values
        if y_true.mean() < minimum_mean:
            results.append(np.asarray([point_estimate, np.nan, np.nan]))
            continue
        
        # Nearly constant values
        if y_true.var() < minimum_variance:
            results.append(np.asarray([point_estimate, np.nan, np.nan]))
            continue
        
        # Optimal block size
        max_value = y_true.max() * 1.01
        normalized = y_true.to_numpy() / max_value
        block_size = optimal_block_length(normalized)["stationary"][0]
        if np.isnan(block_size):
            block_size = 1
        else:
            block_size = max(1, int(block_size))
        
        # Bootstrap confidence interval for each metric
        obs = y_true.to_numpy()
        pred = y_pred.to_numpy()
        indices = np.arange(obs.size)
        bs = StationaryBootstrap(
            block_size,
            indices,
            seed=2025
            )
        posterior = []
        for samples in bs.bootstrap(1000):
            # Certain functions may need additional error checks
            idx = samples[0][0]
            posterior.append(func(obs[idx], pred[idx]))
        ci = np.quantile(posterior, [0.025, 0.975])
        results.append(np.concatenate(([point_estimate], ci)))
    return np.concatenate(results)
