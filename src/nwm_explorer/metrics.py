"""Various methods used to compute model evaluation metrics."""
from typing import Iterable
from datetime import timedelta
import numpy as np
import numpy.typing as npt
from numba import float64, guvectorize
import polars as pl

@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()")
def nash_sutcliffe_efficiency(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    result: npt.NDArray[np.float64]
    ) -> None:
    """
    Numba and polars compatible implementation of the Nash-Sutcliffe Model
    Efficiency score.
        
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
        
    References
    ----------
    Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through 
        conceptual models part I—A discussion of principles. Journal of 
        hydrology, 10(3), 282-290.
    Nossent, J., & Bauwens, W. (2012, April). Application of a normalized 
        Nash-Sutcliffe efficiency to improve the accuracy of the Sobol'sensitivity 
        analysis of a hydrological model. In EGU General Assembly Conference 
        Abstracts (p. 237).
    
    """
    variance = np.sum((y_true - np.mean(y_true)) ** 2.0)
    if variance == 0:
        result[0] = np.nan
        return
    result[0] = 1.0 - np.sum((y_true - y_pred) ** 2.0) / variance

@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()")
def mean_relative_bias(
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
    non_zero = np.where(y_true != 0, True, False)
    if not non_zero.all():
        result[0] = np.nan
        return
    result[0] = np.mean((y_pred[non_zero] - y_true[non_zero]) / y_true[non_zero])

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

def resample(
        data: pl.LazyFrame,
        sort_by: Iterable[pl.Expr] | None = None,
        index_column: str | pl.Expr = "value_time",
        group_by: str | Iterable[pl.Expr] = "usgs_site_code",
        sampling: Iterable[pl.Expr] | None = None,
        sampling_frequency: str | timedelta = "1d"
    ) -> pl.LazyFrame:
    """
    Resample input to new time frequency. Returns daily maximum flow by
    default.

    Parameters
    ----------
    data: pl.LazyFrame
        Input dataframe.
    sort_by: Iterable[pl.Expr], optional
        Data must be sorted before resampling. Sorts by 'usgs_site_code' and
        'value_time' by default.
    index_column: str | pl.Expr, optional
        Column group time on. Default is 'value_time'.
    group_by: str | Iterable[pl.Expr], optional
        Additional column to group on. Default is 'usgs_site_code'.
    sampling: Iterable[pl.Expr], optional
        Functions to use for each column. Default is to take the maximum of
        each group for 'predicted' and 'observed' columns.
    sampling_frequency: str | timedelta, optional
        Desired resultant frequency. Default is daily.

    Returns
    -------
    pl.LazyFrame
    """
    if sort_by is None:
        sort_by = ("usgs_site_code", "value_time")
    if sampling is None:
        sampling = (
            pl.col("predicted").max(),
            pl.col("observed").max()
            )
    return data.sort(sort_by).group_by_dynamic(
        index_column,
        every=sampling_frequency,
        group_by=group_by
    ).agg(*sampling)

def kling_gupta_efficiency() -> pl.Expr:
    """Returns an expression used to add a 'kling_gupta_efficiency' column to a dataframe."""
    return (1.0 - np.sqrt(
        ((pl.col("pearson_correlation_coefficient") - 1.0)) ** 2.0 + 
        ((pl.col("relative_variability") - 1.0)) ** 2.0 + 
        ((pl.col("relative_mean") - 1.0)) ** 2.0
        )).alias("kling_gupta_efficiency")
