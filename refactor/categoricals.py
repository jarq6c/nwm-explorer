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

def main():
    rng = np.random.default_rng(seed=2025)
    bootstrap_samples = 1000
    time_series_length = 100
    choices = [False, True]
    qs = [0.025, 0.975]

    xs, ys = [], []
    for _ in range(bootstrap_samples):
        # Simulate bootstrap
        xs.append(rng.choice(choices, time_series_length))
        ys.append(rng.choice(choices, time_series_length))

    from time import perf_counter
    print("Computing...")
    start = perf_counter()

    tps = np.zeros(shape=bootstrap_samples, dtype=np.int64)
    fps = np.zeros(shape=bootstrap_samples, dtype=np.int64)
    fns = np.zeros(shape=bootstrap_samples, dtype=np.int64)
    tns = np.zeros(shape=bootstrap_samples, dtype=np.int64)

    tp = np.zeros(shape=1, dtype=np.int64)
    fp = np.zeros(shape=1, dtype=np.int64)
    fn = np.zeros(shape=1, dtype=np.int64)
    tn = np.zeros(shape=1, dtype=np.int64)

    for idx, (x, y) in enumerate(zip(xs, ys)):
        # contingency table
        compute_contingency_table(x, y, tp, fp, fn, tn)

        # Store result
        tps[idx] = tp[0]
        fps[idx] = fp[0]
        fns[idx] = fn[0]
        tns[idx] = tn[0]

    pod = np.zeros(shape=bootstrap_samples, dtype=np.float64)
    probability_of_detection(tps, fps, fns, tns, pod)
    pofd = np.zeros(shape=bootstrap_samples, dtype=np.float64)
    probability_of_false_detection(tps, fps, fns, tns, pofd)
    pofa = np.zeros(shape=bootstrap_samples, dtype=np.float64)
    probability_of_false_alarm(tps, fps, fns, tns, pofa)
    csi = np.zeros(shape=bootstrap_samples, dtype=np.float64)
    threat_score(tps, fps, fns, tns, csi)
    fbi = np.zeros(shape=bootstrap_samples, dtype=np.float64)
    frequency_bias(tps, fps, fns, tns, fbi)
    pc = np.zeros(shape=bootstrap_samples, dtype=np.float64)
    percent_correct(tps, fps, fns, tns, pc)
    base = np.zeros(shape=bootstrap_samples, dtype=np.float64)
    base_chance(tps, fps, fns, tns, base)
    ets = np.zeros(shape=bootstrap_samples, dtype=np.float64)
    equitable_threat_score(tps, fps, fns, tns, ets)

    pods = np.quantile(pod, qs)
    pofds = np.quantile(pofd, qs)
    pofas = np.quantile(pofa, qs)
    csis = np.quantile(csi, qs)
    fbis = np.quantile(fbi, qs)
    pcs = np.quantile(pc, qs)
    bases = np.quantile(base, qs)
    etss = np.quantile(ets, qs)

    duration = perf_counter() - start
    print(f"Time: {duration:.4f} s")

    print(f"POD: {pods[0]:.3f} - {pods[1]:.3f}")
    print(f"POFD: {pofds[0]:.3f} - {pofds[1]:.3f}")
    print(f"POFA: {pofas[0]:.3f} - {pofas[1]:.3f}")
    print(f"CSI: {csis[0]:.3f} - {csis[1]:.3f}")
    print(f"FBI: {fbis[0]:.3f} - {fbis[1]:.3f}")
    print(f"PC: {pcs[0]:.3f} - {pcs[1]:.3f}")
    print(f"Base: {bases[0]:.3f} - {bases[1]:.3f}")
    print(f"ETS: {etss[0]:.3f} - {etss[1]:.3f}")

if __name__ == "__main__":
    main()
