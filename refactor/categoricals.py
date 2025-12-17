"""Implement vectorized versions of categorical metrics."""
import numpy as np
import numpy.typing as npt
import numba as nb

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
    # TODO Finishing updating metric computations
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
    for i in range(true_positives.shape[0]):
        denominator = true_positives[i] + false_negatives[i]
        if denominator <= 0.0:
            result[i] = np.nan
        else:
            result[i] = true_positives[i] / denominator

def main():
    rng = np.random.default_rng(seed=2025)
    x = rng.choice([False, True], 100)
    y = rng.choice([False, True], 100)

    # contingency table
    tp = np.empty(shape=1, dtype=np.int64)
    fp = np.empty(shape=1, dtype=np.int64)
    fn = np.empty(shape=1, dtype=np.int64)
    tn = np.empty(shape=1, dtype=np.int64)
    compute_contingency_table(x, y, tp, fp, fn, tn)

    pod = np.empty(shape=1, dtype=np.float64)
    probability_of_detection(tp, fp, fn, tn, pod)
    pofd = np.empty(shape=1, dtype=np.float64)
    probability_of_false_detection(tp, fp, fn, tn, pofd)
    pofa = np.empty(shape=1, dtype=np.float64)
    probability_of_false_alarm(tp, fp, fn, tn, pofa)
    csi = np.empty(shape=1, dtype=np.float64)
    threat_score(tp, fp, fn, tn, csi)
    fbi = np.empty(shape=1, dtype=np.float64)
    frequency_bias(tp, fp, fn, tn, fbi)
    pc = np.empty(shape=1, dtype=np.float64)
    percent_correct(tp, fp, fn, tn, pc)
    base = np.empty(shape=1, dtype=np.float64)
    base_chance(tp, fp, fn, tn, base)
    ets = np.empty(shape=1, dtype=np.float64)
    equitable_threat_score(tp, fp, fn, tn, ets)

    print(f"POD: {pod[0]:.3f}")
    print(f"POFD: {pofd[0]:.3f}")
    print(f"POFA: {pofa[0]:.3f}")
    print(f"CSI: {csi[0]:.3f}")
    print(f"FBI: {fbi[0]:.3f}")
    print(f"PC: {pc[0]:.3f}")
    print(f"Base: {base[0]:.3f}")
    print(f"ETS: {ets[0]:.3f}")

if __name__ == "__main__":
    main()
