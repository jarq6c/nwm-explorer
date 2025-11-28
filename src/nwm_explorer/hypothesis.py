"""
Methods to support applying alternative hypotheses to evaluation metric
95% confidence intervals.
"""
from typing import Protocol

import numpy as np
import numpy.typing as npt

class HypothesisTest(Protocol):
    """
    Protocol definition for hypothesis test functions.
    
    Parameters
    ----------
    lower: array-like, required
        Array of lower values of 95% confidence intervals to test. Must be same
        length as upper. Values are assumed to correpond to upper.
    upper: array-like, required
        Array of upper values of 95% confidence intervals to test. Must be same
        length as lower. Values are assumed to correpond to lower.
    
    Returns
    -------
    Boolean array.
    """
    def __call__(
            self,
            lower: npt.NDArray[np.float32],
            upper: npt.NDArray[np.float32]
            ) -> npt.NDArray[np.bool]:
        ...

HYPOTHESIS_TESTS: dict[str, HypothesisTest | None] = {
    "None": None,
    "metric is not 0.0": lambda lower, upper: (lower > 0.0) | (upper < 0.0 ),
    "metric is not 1.0": lambda lower, upper: (lower > 1.0) | (upper < 1.0 ),
    "metric greater than 0.0": lambda lower, _: (lower > 0.0),
    "metric less than 0.0": lambda _, upper: (upper < 0.0 ),
    "metric greater than 1.0": lambda lower, _: (lower > 1.0),
    "metric less than 1.0": lambda _, upper: (upper < 1.0 )
}
"""Mapping from strings to hypothesis test functions."""
