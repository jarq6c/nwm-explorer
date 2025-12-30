"""Methods to evaluate pairs."""
from pathlib import Path
import inspect
from typing import Generator, Any, Literal
from concurrent.futures import ProcessPoolExecutor
import itertools
import functools

import polars as pl
import pandas as pd
import numpy as np
import numpy.typing as npt
import numba as nb
from arch.bootstrap import StationaryBootstrap, optimal_block_length

from nwm_explorer.nwm import ModelConfiguration
from nwm_explorer.pairs import scan_pairs, GROUP_SPECIFICATIONS
from nwm_explorer.logger import get_logger
from nwm_explorer.routelink import download_routelink
from nwm_explorer.constants import (Metric, SUBDIRECTORIES, LRU_CACHE_SIZES,
    MetricFunction, NO_THRESHOLD_LABEL, CategoricalMetric, CategoricalMetricFunction)
from nwm_explorer.hypothesis import HYPOTHESIS_TESTS

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], "(n),(n)->()")
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

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], "(n),(n)->()")
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

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], "(n),(n)->()")
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

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], "(n),(n)->()")
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

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], "(n),(n)->()")
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

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], "(n),(n)->()")
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

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], "(n),(n)->()")
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

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], "(n),(n)->()")
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

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], "(n),(n)->()")
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

@nb.guvectorize([(nb.bool[:], nb.bool[:], nb.int64[:])], "(n),(n)->()")
def compute_true_positives(
    y_true: npt.NDArray[np.bool],
    y_pred: npt.NDArray[np.bool],
    result: npt.NDArray[np.int64]
    ) -> None:
    """
    Numba compatible implementation of contigency table cross-tabulation for
    true positives.
        
    Parameters
    ----------
    y_true: NDArray[np.bool], required
        Ground truth (correct) target values, also called observations,
        measurements, or observed values.
    y_pred: NDArray[np.bool], required
        Estimated target values, also called simulations, forecasts, or modeled values.
    result: NDArray[np.int64], required
        Stores tabulated value.
        
    Returns
    -------
    None
    """
    # True positives
    result[0] = np.sum(np.where(y_true & y_pred, 1, 0), dtype=np.int64)

@nb.guvectorize([(nb.bool[:], nb.bool[:], nb.int64[:])], "(n),(n)->()")
def compute_false_positives(
    y_true: npt.NDArray[np.bool],
    y_pred: npt.NDArray[np.bool],
    result: npt.NDArray[np.int64]
    ) -> None:
    """
    Numba compatible implementation of contigency table cross-tabulation for
    false positives.
        
    Parameters
    ----------
    y_true: NDArray[np.bool], required
        Ground truth (correct) target values, also called observations,
        measurements, or observed values.
    y_pred: NDArray[np.bool], required
        Estimated target values, also called simulations, forecasts, or modeled values.
    result: NDArray[np.int64], required
        Stores tabulated value.
        
    Returns
    -------
    None
    """
    # False positives
    result[0] = np.sum(np.where(~y_true & y_pred, 1, 0), dtype=np.int64)

@nb.guvectorize([(nb.bool[:], nb.bool[:], nb.int64[:])], "(n),(n)->()")
def compute_false_negatives(
    y_true: npt.NDArray[np.bool],
    y_pred: npt.NDArray[np.bool],
    result: npt.NDArray[np.int64]
    ) -> None:
    """
    Numba compatible implementation of contigency table cross-tabulation for
    false negatives.
        
    Parameters
    ----------
    y_true: NDArray[np.bool], required
        Ground truth (correct) target values, also called observations,
        measurements, or observed values.
    y_pred: NDArray[np.bool], required
        Estimated target values, also called simulations, forecasts, or modeled values.
    result: NDArray[np.int64], required
        Stores tabulated value.
        
    Returns
    -------
    None
    """
    # False negatives
    result[0] = np.sum(np.where(y_true & ~y_pred, 1, 0), dtype=np.int64)

@nb.guvectorize([(nb.bool[:], nb.bool[:], nb.int64[:])], "(n),(n)->()")
def compute_true_negatives(
    y_true: npt.NDArray[np.bool],
    y_pred: npt.NDArray[np.bool],
    result: npt.NDArray[np.int64]
    ) -> None:
    """
    Numba compatible implementation of contigency table cross-tabulation for
    true negatives.
        
    Parameters
    ----------
    y_true: NDArray[np.bool], required
        Ground truth (correct) target values, also called observations,
        measurements, or observed values.
    y_pred: NDArray[np.bool], required
        Estimated target values, also called simulations, forecasts, or modeled values.
    result: NDArray[np.int64], required
        Stores tabulated value.
        
    Returns
    -------
    None
    """
    # True negatives
    result[0] = np.sum(np.where(~y_true & ~y_pred, 1, 0), dtype=np.int64)

@nb.guvectorize(
    [(nb.bool[:], nb.bool[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.int64[:])],
    "(n),(n)->(),(),(),()"
)
def compute_contingency_table(
    y_true: npt.NDArray[np.bool],
    y_pred: npt.NDArray[np.bool],
    true_positives: npt.NDArray[np.int64],
    false_positives: npt.NDArray[np.int64],
    false_negatives: npt.NDArray[np.int64],
    true_negatives: npt.NDArray[np.int64]
    ) -> None:
    """
    Numba compatible implementation of contigency table cross-tabulation.
        
    Parameters
    ----------
    y_true: NDArray[np.bool], required
        Ground truth (correct) target values, also called observations,
        measurements, or observed values.
    y_pred: NDArray[np.bool], required
        Estimated target values, also called simulations, forecasts, or modeled values.
    true_positives: NDArray[np.int64], required
        Stores tabulated true positives.
    false_positives: NDArray[np.int64], required
        Stores tabulated false positives.
    false_negatives: NDArray[np.int64], required
        Stores tabulated false negatives.
    true_negatives: NDArray[np.int64], required
        Stores tabulated true negatives.
        
    Returns
    -------
    None
    """
    compute_true_positives(y_true, y_pred, true_positives)
    compute_false_positives(y_true, y_pred, false_positives)
    compute_false_negatives(y_true, y_pred, false_negatives)
    compute_true_negatives(y_true, y_pred, true_negatives)

@nb.guvectorize([
    (nb.int64[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.float64[:])
    ],
    "(n),(n),(n),(n)->(n)")
def probability_of_detection(
    true_positives: npt.NDArray[np.int64],
    false_positives: npt.NDArray[np.int64],
    false_negatives: npt.NDArray[np.int64],
    true_negatives: npt.NDArray[np.int64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of probability of detection.
        
    Parameters
    ----------
    true_positives: NDArray[np.int64], required
        True positives.
    false_positives: NDArray[np.int64], required
        False positives.
    false_negatives: NDArray[np.int64], required
        False negatives.
    true_negatives: NDArray[np.int64], required
        True negatives.
    result: NDArray[np.int64], required
        Resulting values.
        
    Returns
    -------
    None
    """
    for i in range(true_positives.shape[0]):
        denominator = true_positives[i] + false_negatives[i]
        if denominator <= 0.0:
            result[i] = np.nan
        else:
            result[i] = true_positives[i] / denominator

@nb.guvectorize([
    (nb.int64[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.float64[:])
    ],
    "(n),(n),(n),(n)->(n)")
def probability_of_false_detection(
    true_positives: npt.NDArray[np.int64],
    false_positives: npt.NDArray[np.int64],
    false_negatives: npt.NDArray[np.int64],
    true_negatives: npt.NDArray[np.int64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of probability of false detection.
        
    Parameters
    ----------
    true_positives: NDArray[np.int64], required
        True positives.
    false_positives: NDArray[np.int64], required
        False positives.
    false_negatives: NDArray[np.int64], required
        False negatives.
    true_negatives: NDArray[np.int64], required
        True negatives.
    result: NDArray[np.int64], required
        Resulting values.
        
    Returns
    -------
    None
    """
    for i in range(false_positives.shape[0]):
        denominator = false_positives[i] + true_negatives[i]
        if denominator <= 0.0:
            result[i] = np.nan
        else:
            result[i] = false_positives[i] / denominator

@nb.guvectorize([
    (nb.int64[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.float64[:])
    ],
    "(n),(n),(n),(n)->(n)")
def probability_of_false_alarm(
    true_positives: npt.NDArray[np.int64],
    false_positives: npt.NDArray[np.int64],
    false_negatives: npt.NDArray[np.int64],
    true_negatives: npt.NDArray[np.int64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of probability of false alarm.
        
    Parameters
    ----------
    true_positives: NDArray[np.int64], required
        True positives.
    false_positives: NDArray[np.int64], required
        False positives.
    false_negatives: NDArray[np.int64], required
        False negatives.
    true_negatives: NDArray[np.int64], required
        True negatives.
    result: NDArray[np.int64], required
        Resulting values.
        
    Returns
    -------
    None
    """
    for i in range(false_positives.shape[0]):
        denominator = false_positives[i] + true_positives[i]
        if denominator <= 0.0:
            result[i] = np.nan
        else:
            result[i] = false_positives[i] / denominator

@nb.guvectorize([
    (nb.int64[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.float64[:])
    ],
    "(n),(n),(n),(n)->(n)")
def threat_score(
    true_positives: npt.NDArray[np.int64],
    false_positives: npt.NDArray[np.int64],
    false_negatives: npt.NDArray[np.int64],
    true_negatives: npt.NDArray[np.int64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of threat score.
        
    Parameters
    ----------
    true_positives: NDArray[np.int64], required
        True positives.
    false_positives: NDArray[np.int64], required
        False positives.
    false_negatives: NDArray[np.int64], required
        False negatives.
    true_negatives: NDArray[np.int64], required
        True negatives.
    result: NDArray[np.int64], required
        Resulting values.
        
    Returns
    -------
    None
    """
    for i in range(true_positives.shape[0]):
        denominator = true_positives[i] + false_positives[i] + false_negatives[i]
        if denominator <= 0.0:
            result[i] = np.nan
        else:
            result[i] = true_positives[i] / denominator

@nb.guvectorize([
    (nb.int64[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.float64[:])
    ],
    "(n),(n),(n),(n)->(n)")
def frequency_bias(
    true_positives: npt.NDArray[np.int64],
    false_positives: npt.NDArray[np.int64],
    false_negatives: npt.NDArray[np.int64],
    true_negatives: npt.NDArray[np.int64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of frequency bias.
        
    Parameters
    ----------
    true_positives: NDArray[np.int64], required
        True positives.
    false_positives: NDArray[np.int64], required
        False positives.
    false_negatives: NDArray[np.int64], required
        False negatives.
    true_negatives: NDArray[np.int64], required
        True negatives.
    result: NDArray[np.int64], required
        Resulting values.
        
    Returns
    -------
    None
    """
    for i in range(true_positives.shape[0]):
        denominator = true_positives[i] + false_negatives[i]
        if denominator <= 0.0:
            result[i] = np.nan
        else:
            result[i] = (true_positives[i] + false_positives[i]) / denominator

@nb.guvectorize([
    (nb.int64[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.float64[:])
    ],
    "(n),(n),(n),(n)->(n)")
def percent_correct(
    true_positives: npt.NDArray[np.int64],
    false_positives: npt.NDArray[np.int64],
    false_negatives: npt.NDArray[np.int64],
    true_negatives: npt.NDArray[np.int64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of percent correct.
        
    Parameters
    ----------
    true_positives: NDArray[np.int64], required
        True positives.
    false_positives: NDArray[np.int64], required
        False positives.
    false_negatives: NDArray[np.int64], required
        False negatives.
    true_negatives: NDArray[np.int64], required
        True negatives.
    result: NDArray[np.int64], required
        Resulting values.
        
    Returns
    -------
    None
    """
    for i in range(true_positives.shape[0]):
        denominator = (
            true_positives[i] +
            false_positives[i] +
            false_negatives[i] +
            true_negatives[i]
        )
        if denominator <= 0.0:
            result[i] = np.nan
        else:
            result[i] = (true_positives[i] + true_negatives[i]) / denominator

@nb.guvectorize([
    (nb.int64[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.float64[:])
    ],
    "(n),(n),(n),(n)->(n)")
def base_chance(
    true_positives: npt.NDArray[np.int64],
    false_positives: npt.NDArray[np.int64],
    false_negatives: npt.NDArray[np.int64],
    true_negatives: npt.NDArray[np.int64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of base chance.
        
    Parameters
    ----------
    true_positives: NDArray[np.int64], required
        True positives.
    false_positives: NDArray[np.int64], required
        False positives.
    false_negatives: NDArray[np.int64], required
        False negatives.
    true_negatives: NDArray[np.int64], required
        True negatives.
    result: NDArray[np.int64], required
        Resulting values.
        
    Returns
    -------
    None
    """
    for i in range(true_positives.shape[0]):
        denominator = (
            true_positives[i] +
            false_positives[i] +
            false_negatives[i] +
            true_negatives[i]
        )
        if denominator <= 0.0:
            result[i] = np.nan
        else:
            result[i] = (
                (true_positives[i] + false_positives[i]) *
                (true_positives[i] + false_negatives[i]) /
                denominator
            )

@nb.guvectorize([
    (nb.int64[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.float64[:])
    ],
    "(n),(n),(n),(n)->(n)")
def equitable_threat_score(
    true_positives: npt.NDArray[np.int64],
    false_positives: npt.NDArray[np.int64],
    false_negatives: npt.NDArray[np.int64],
    true_negatives: npt.NDArray[np.int64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba compatible implementation of equitable threat score.
        
    Parameters
    ----------
    true_positives: NDArray[np.int64], required
        True positives.
    false_positives: NDArray[np.int64], required
        False positives.
    false_negatives: NDArray[np.int64], required
        False negatives.
    true_negatives: NDArray[np.int64], required
        True negatives.
    result: NDArray[np.int64], required
        Resulting values.
        
    Returns
    -------
    None
    """
    a_r = np.zeros(shape=true_positives.shape, dtype=np.float64)
    base_chance(
        true_positives,
        false_positives,
        false_negatives,
        true_negatives,
        a_r
    )
    for i in range(true_positives.shape[0]):
        if np.isnan(a_r[i]):
            result[i] = np.nan
            continue
        denominator = true_positives[i] + false_positives[i] + false_negatives[i] - a_r[i]
        if denominator <= 0.0:
            result[i] = np.nan
        else:
            result[i] = (true_positives[i] - a_r[i]) / denominator

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

CATEOGRICAL_METRIC_FUNCTIONS: dict[CategoricalMetric, CategoricalMetricFunction] = {
    CategoricalMetric.PROBABILITY_OF_DETECTION: probability_of_detection,
    CategoricalMetric.PROBABILITY_OF_FALSE_DETECTION: probability_of_false_detection,
    CategoricalMetric.PROBABILITY_OF_FALSE_ALARM: probability_of_false_alarm,
    CategoricalMetric.FREQUENCY_BIAS: frequency_bias,
    CategoricalMetric.THREAT_SCORE: threat_score,
    CategoricalMetric.EQUITABLE_THREAT_SCORE: equitable_threat_score,
    CategoricalMetric.PERCENT_CORRECT: percent_correct,
}
"""Mapping from categorical metrics to functions used to compute them."""

def bootstrap_metrics(
    data: pd.DataFrame,
    minimum_sample_size: int = 30,
    minimum_mean: float = 0.01,
    minimum_variance: float = 0.000025,
    bootstrap_iterations: int = 1000
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
    bootstrap_iterations: int, optional, default 1000
        Number of posterior bootstrap samples to generate.
    
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
        "threshold": data["threshold"].iloc[0],
        "threshold_value": data["threshold_value"].iloc[0]
    }

    # Allocate memory
    posterior = np.zeros(bootstrap_iterations, dtype=np.float64)
    estimate = np.zeros(shape=1, dtype=np.float64)
    tp = np.zeros(shape=1, dtype=np.int64)
    fp = np.zeros(shape=1, dtype=np.int64)
    fn = np.zeros(shape=1, dtype=np.int64)
    tn = np.zeros(shape=1, dtype=np.int64)

    # Memory for contingency table posteriors
    tps = np.zeros(shape=bootstrap_iterations, dtype=np.int64)
    fps = np.zeros(shape=bootstrap_iterations, dtype=np.int64)
    fns = np.zeros(shape=bootstrap_iterations, dtype=np.int64)
    tns = np.zeros(shape=bootstrap_iterations, dtype=np.int64)

    # Compute continuous metrics
    for rank in ["min", "median", "max"]:
        # Extract numpy arrays for Numba functions
        y_true = data[f"observed_cfs_{rank}"].to_numpy(dtype=np.float64)
        y_pred = data[f"predicted_cfs_{rank}"].to_numpy(dtype=np.float64)

        # Condition continuous metrics
        if result["threshold"] != NO_THRESHOLD_LABEL:
            y_true = y_true[y_pred >= result["threshold_value"]]
            y_pred = y_pred[y_pred >= result["threshold_value"]]

        # Initialize bootstrap samples
        idxs = []

        # Sample size
        result["sample_size"] = len(y_pred)

        # Process each continuous metric
        for label, func in METRIC_FUNCTIONS.items():
            # Sample size too small to compute metric
            if result["sample_size"] == 0:
                result[f"{label}_{rank}_point"] = np.nan
                result[f"{label}_{rank}_lower"] = np.nan
                result[f"{label}_{rank}_upper"] = np.nan
                continue

            # Point estimate
            # NOTE Numba has magic that implicitly instantiates point, but
            #  it makes the linter mad.
            func(y_true, y_pred, estimate)
            result[f"{label}_{rank}_point"] = estimate[0]

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

            # Optimal block size for bootstrap
            # NOTE Normalizing the values seems to produce more consistent
            #  block sizes. Here we let the "true" values determine the block size.
            if len(idxs) == 0:
                max_value = np.max(y_true) * 1.01
                normalized = y_true / max(0.01, max_value)
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

                # Bootstrap sample time series
                for samples in bs.bootstrap(bootstrap_iterations):
                    # Generate an index
                    idxs.append(samples[0][0])

            # Apply the new index and compute the metric
            for iteration, idx in enumerate(idxs):
                func(y_true[idx], y_pred[idx], estimate)
                posterior[iteration] = estimate[0]

            # Compute confidence interval
            ci = np.quantile(posterior, [0.025, 0.975])
            result[f"{label}_{rank}_lower"] = ci[0]
            result[f"{label}_{rank}_upper"] = ci[1]

        # Re-initialize bootstrap samples
        idxs = []

        # Generate dichotomous variables
        y_true = data[f"observed_cfs_{rank}"].to_numpy(dtype=np.float64)
        y_true_bool = y_true >= result["threshold_value"]
        y_pred = data[f"predicted_cfs_{rank}"].to_numpy(dtype=np.float64)
        y_pred_bool = y_pred >= result["threshold_value"]

        # Process each categorical metric
        for label, func in CATEOGRICAL_METRIC_FUNCTIONS.items():
            # Sample size too small to compute metric
            if result["threshold"] == NO_THRESHOLD_LABEL:
                result[f"{label}_{rank}_point"] = np.nan
                result[f"{label}_{rank}_lower"] = np.nan
                result[f"{label}_{rank}_upper"] = np.nan
                continue

            # Point estimate
            # NOTE Numba has magic that implicitly instantiates point, but
            #  it makes the linter mad.
            compute_contingency_table(y_true_bool, y_pred_bool, tp, fp, fn, tn)
            func(tp, fp, fn, tn, estimate)
            result[f"{label}_{rank}_point"] = estimate[0]

            # Sample size too small reliably to compute confidence interval
            if result["sample_size"] < minimum_sample_size:
                result[f"{label}_{rank}_lower"] = np.nan
                result[f"{label}_{rank}_upper"] = np.nan
                continue

            # Optimal block size for bootstrap
            # NOTE Normalizing the values seems to produce more consistent
            #  block sizes. Here we let the "true" values determine the block size.
            if len(idxs) == 0:
                max_value = np.max(y_true) * 1.01
                normalized = y_true / max(0.01, max_value)
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

                # Bootstrap sample time series
                for samples in bs.bootstrap(bootstrap_iterations):
                    # Generate an index
                    idxs.append(samples[0][0])

                # Apply the new index and compute the metric
                for iteration, idx in enumerate(idxs):
                    # Contingency table
                    compute_contingency_table(
                        y_true_bool[idx],
                        y_pred_bool[idx],
                        tp, fp, fn, tn
                    )

                    # Store results
                    tps[iteration] = tp[0]
                    fps[iteration] = fp[0]
                    fns[iteration] = fn[0]
                    tns[iteration] = tn[0]

            # Compute confidence interval
            func(tps, fps, fns, tns, posterior)
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
        features: npt.NDArray[np.int64],
        threshold_map: pl.DataFrame | None = None,
        threshold_column: str | None = None
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

    # Check for data
    if len(dataframes) == 0:
        return pl.DataFrame()

    # Compute lead time and pool pairs
    logger.info("Pooling pairs")
    data: pl.DataFrame = pl.concat(
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

    # Add thresholds
    if threshold_column == NO_THRESHOLD_LABEL:
        logger.info("Adding stub thresholds")
        data = data.with_columns(
            pl.Series(name="threshold", values=[NO_THRESHOLD_LABEL]*len(data["nwm_feature_id"])),
            pl.Series(name="threshold_value", values=[np.nan]*len(data["nwm_feature_id"]))
        )
    else:
        logger.info("Mapping thresholds")
        data = data.with_columns(
            pl.Series(name="threshold", values=[threshold_column]*len(data["nwm_feature_id"])),
            threshold_value=pl.col("nwm_feature_id").replace_strict(
                old=threshold_map["nwm_feature_id"].implode(),
                new=threshold_map[threshold_column].implode()
            )
        )
    return data

def prediction_pool_generator(
        root: Path,
        configuration: ModelConfiguration,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        lead_time_interval: int,
        sites_per_chunk: int = 1,
        threshold_map: pl.DataFrame | None = None,
        threshold_column: str | None = None
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
            features=features,
            threshold_map=threshold_map,
            threshold_column=threshold_column
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
        sites_per_chunk: int = 1,
        threshold_file: Path | None = None,
        threshold_columns: list[str] | None = None
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
    processes: int, optional, default 1
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

    # Load thresholds
    thresholds = None
    tholds = [NO_THRESHOLD_LABEL]
    if threshold_file is None:
        logger.info("No thresholds applied")
    else:
        logger.info("Loading %s", threshold_file)
        thresholds = pl.scan_parquet(threshold_file).select(
            ["nwm_feature_id"]+threshold_columns
        ).collect()
        tholds = tholds+threshold_columns

    # Start process pool
    logger.info("Starting %d processes", processes)
    subdirectory = SUBDIRECTORIES["evaluations"]
    with ProcessPoolExecutor(max_workers=processes) as parallel_computer:
        # Process each configuration
        for config, specs in GROUP_SPECIFICATIONS.items():
            for thold in tholds:
                # Prepare output file
                ofile = root / (
                    f"{subdirectory}/label={label}/threshold={thold}/" +
                    f"configuration={config}/E0.parquet"
                )
                if ofile.exists():
                    logger.info("Found %s", ofile)
                    continue
                ofile.parent.mkdir(exist_ok=True, parents=True)

                # Process in chunks
                logger.info("Building %s", ofile)
                logger.info("Grouping into chunks of %d sites", sites_per_chunk)
                batch_counter = itertools.count(1)
                dataframes = []
                for groups in prediction_pool_generator(
                    root=root,
                    configuration=config,
                    start_time=start_time,
                    end_time=end_time,
                    lead_time_interval=specs.window_interval,
                    sites_per_chunk=sites_per_chunk,
                    threshold_map=thresholds,
                    threshold_column=thold
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

                # Check for data
                if len(dataframes) == 0:
                    logger.info("No groups, skipping %s", ofile)
                    continue

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
    subdirectory = SUBDIRECTORIES["evaluations"]
    return pl.scan_parquet(
        root / f"{subdirectory}/",
        hive_schema={
            "label": pl.String,
            "threshold": pl.String,
            "configuration": pl.Enum(ModelConfiguration)
        }
    )

@functools.lru_cache(LRU_CACHE_SIZES["evaluations"])
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
        rank: Literal["min", "median", "max"] = "median",
        additional_columns: tuple[str] | None = None,
        condition: str | None = None,
        threshold: str = NO_THRESHOLD_LABEL
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
    additional_columns: tuple[str], optional, default ("nwm_feature_id",)
        Additional columns (often metadata) to return with metric values.
    condition: str, optional
        Conditional test to apply to filter out metrics.
    threshold: str, optional
        Streamflow threshold label applied.
    
    Returns
    -------
    polars.DataFrame
    """
    # Check additional columns
    if additional_columns is None:
        additional_columns = ["nwm_feature_id"]
    else:
        additional_columns = list(additional_columns)

    # Retrieve
    data = scan_evaluations(
        root
    ).filter(
        pl.col("label") == label,
        pl.col("threshold") == threshold,
        pl.col("configuration") == configuration,
        pl.col("lead_time_hours_min") == lead_time_hours_min
    ).select(
        [
            f"{metric}_{rank}_lower",
            f"{metric}_{rank}_point",
            f"{metric}_{rank}_upper"
        ] + additional_columns
    ).collect()

    # Apply hypothesis test
    func = HYPOTHESIS_TESTS.get(condition)
    if func is not None:
        return data.filter(
            func(
                data[f"{metric}_{rank}_lower"].to_numpy(),
                data[f"{metric}_{rank}_upper"].to_numpy()
            )
        )
    return data

@functools.lru_cache(LRU_CACHE_SIZES["evaluations"])
def load_metrics_cache(
        root: Path,
        label: str,
        configuration: ModelConfiguration,
        metric: Metric,
        lead_time_hours_min: int = 0,
        rank: Literal["min", "median", "max"] = "median",
        additional_columns: tuple[str] | None = None,
        condition: str | None = None,
        threshold: str = NO_THRESHOLD_LABEL
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
    additional_columns: tuple[str], optional, default ("nwm_feature_id",)
        Additional columns (often metadata) to return with metric values.
    condition: str, optional
        Conditional test to apply to filter out metrics.
    threshold: str, optional
        Streamflow threshold label applied.
    
    Returns
    -------
    polars.DataFrame
    """
    # Check additional columns
    if additional_columns is None:
        additional_columns = ["nwm_feature_id"]
    else:
        additional_columns = list(additional_columns)

    # Retrieve
    data = scan_evaluations(
        root, cache=True
    ).filter(
        pl.col("label") == label,
        pl.col("threshold") == threshold,
        pl.col("configuration") == configuration,
        pl.col("lead_time_hours_min") == lead_time_hours_min
    ).select(
        [
            f"{metric}_{rank}_lower",
            f"{metric}_{rank}_point",
            f"{metric}_{rank}_upper"
        ] + additional_columns
    ).collect()

    # Apply hypothesis test
    func = HYPOTHESIS_TESTS.get(condition)
    if func is not None:
        return data.filter(
            func(
                data[f"{metric}_{rank}_lower"].to_numpy(),
                data[f"{metric}_{rank}_upper"].to_numpy()
            )
        )
    return data

def load_metrics(
        root: Path,
        label: str,
        configuration: ModelConfiguration,
        metric: Metric,
        lead_time_hours_min: int = 0,
        rank: Literal["min", "median", "max"] = "median",
        additional_columns: tuple[str] | None = None,
        condition: str | None = None,
        threshold: str = NO_THRESHOLD_LABEL,
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
    additional_columns: tuple[str], optional, default ("nwm_feature_id",)
        Additional columns (often metadata) to return with metric values.
    condition: str, optional
        Conditional test to apply to filter out metrics.
    threshold: str, optional
        Streamflow threshold label applied.
    cache: bool, optional, default False
        If true, cache result.
    
    Returns
    -------
    polars.DataFrame
    """
    # Retrieve
    if cache:
        return load_metrics_cache(
            root=root,
            label=label,
            configuration=configuration,
            metric=metric,
            lead_time_hours_min=lead_time_hours_min,
            rank=rank,
            additional_columns=additional_columns,
            condition=condition,
            threshold=threshold
        )
    return load_metrics_no_cache(
        root=root,
        label=label,
        configuration=configuration,
        metric=metric,
        lead_time_hours_min=lead_time_hours_min,
        rank=rank,
        additional_columns=additional_columns,
        condition=condition,
        threshold=threshold
    )

def load_site_metrics(
        root: Path,
        label: str,
        configuration: ModelConfiguration,
        metric: Metric,
        nwm_feature_id: int,
        rank: Literal["min", "median", "max"] = "median",
        threshold: str = NO_THRESHOLD_LABEL,
        cache: bool = False
) -> pl.DataFrame:
    """
    Returns DataFrame of metrics for a single site.

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
    nwm_feature_id: int
        National Water Model channel feature identifier.
    rank: Literal["min", "median", "max"], optional, default "median"
        Aggregated function applied to predictions and observations. Varies by
        configuration. Analysis & Assimilation and Medium Range configurations
        will typically return daily streamflow aggregated to minimum, median, or
        maximum streamflow. Short Range CONUS, Hawaii, and Puerto Rico return
        6-hourly aggregation. Short Range Alaska returns 5-hourly aggregations.
    threshold: str, optional
        Streamflow threshold label applied.
    cache: bool, optional, default False
        If true, cache underlying polars.LazyFrame.
    
    Returns
    -------
    polars.DataFrame
    """
    # Retrieve
    return scan_evaluations(root, cache).filter(
        pl.col("label") == label,
        pl.col("threshold") == threshold,
        pl.col("configuration") == configuration,
        pl.col("nwm_feature_id") == nwm_feature_id
    ).select(
        [
            "lead_time_hours_min",
            f"{metric}_{rank}_lower",
            f"{metric}_{rank}_point",
            f"{metric}_{rank}_upper"
        ]
    ).collect().sort("lead_time_hours_min")
