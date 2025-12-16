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

    a = tp[0]
    b = fp[0]
    c = fn[0]
    d = tn[0]

    pod = a / (a+c)
    pofd = b / (b+d)
    pofa = b / (b+a)
    csi = a / (a+b+c)
    fbi = (a+b) / (a+c)
    pc = (a+d) / (a+b+c+d)
    base = ((a+b) * (a+c)) // (a+b+c+d)
    ets = (a-base) / (a+b+c-base)

    print(f"POD: {pod:.3f}")
    print(f"POFD: {pofd:.3f}")
    print(f"POFA: {pofa:.3f}")
    print(f"CSI: {csi:.3f}")
    print(f"FBI: {fbi:.3f}")
    print(f"PC: {pc:.3f}")
    print(f"Base: {base:.3f}")
    print(f"ETS: {ets:.3f}")

if __name__ == "__main__":
    main()
