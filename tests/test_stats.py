"""Tests for `lfox.stats`, written before the implementation.

`jackknife(estimator, *columns, bin_size=1, **constants)` takes an ordinary
function of samples.  Positional columns (leading axis = Monte Carlo time) are
resampled together by deleting one block of `bin_size` raw samples at a time;
keyword constants pass through.  It returns (full-sample estimate, jackknife
error), both float64.

`tau_int(series)` returns a `TauInt` report with Sokal's automatic window;
`resolved` is False when the window never closed.
"""

import numpy as np
import pytest

from lfox.stats import TauInt, jackknife, tau_int


def ar1(rho, n, seed=0):
    # Stationary AR(1) series: Gamma(t)/Gamma(0) = rho^|t|, so
    # tau_int = (1 + rho) / (2 (1 - rho)) exactly.
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = noise[0] / np.sqrt(1 - rho**2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + noise[i]
    return x


def delete_one_block(estimator, columns, bin_size):
    # The definition, written out: re-evaluate on raw samples with one block
    # removed, then the usual jackknife variance.
    n_bins = len(columns[0]) // bin_size
    columns = [c[: n_bins * bin_size] for c in columns]
    thetas = []
    for i in range(n_bins):
        keep = np.r_[0 : i * bin_size, (i + 1) * bin_size : n_bins * bin_size]
        thetas.append(estimator(*(c[keep] for c in columns)))
    thetas = np.array(thetas)
    return np.sqrt((n_bins - 1) / n_bins * np.sum((thetas - thetas.mean(0)) ** 2, 0))


# --- jackknife ----------------------------------------------------------------


def test_jackknife_of_mean_is_the_binned_error():
    # For a linear estimator the jackknife error is exactly the standard error
    # of the block means -- so no separate `binned_error` is needed.
    x = np.random.default_rng(1).standard_normal(1200)
    est, err = jackknife(np.mean, x, bin_size=10)

    blocks = x.reshape(120, 10).mean(axis=1)
    assert est == pytest.approx(x.mean(), rel=1e-12)
    assert err == pytest.approx(blocks.std(ddof=1) / np.sqrt(120), rel=1e-12)


def test_jackknife_estimate_is_the_full_sample_estimator():
    # The central value is the estimator on all the data, not the
    # bias-corrected mean of pseudo-values.
    x = np.random.default_rng(2).standard_normal(500) + 0.3
    est, _ = jackknife(lambda x: np.mean(x) ** 2, x)

    assert est == pytest.approx(np.mean(x) ** 2, rel=1e-12)


def test_jackknife_drops_a_partial_final_block():
    x = np.random.default_rng(3).standard_normal(1005)
    est, err = jackknife(np.mean, x, bin_size=10)
    ref_est, ref_err = jackknife(np.mean, x[:1000], bin_size=10)

    assert est == ref_est
    assert err == ref_err


def test_jackknife_deletes_raw_samples_not_bin_means():
    # A nonlinear moment must be re-evaluated on raw samples: <x^4> is not the
    # fourth power of bin means.  Compare against the definition directly.
    def kurtosis(x):
        return np.mean(x**4) / np.mean(x**2) ** 2

    x = np.random.default_rng(4).standard_normal(600)
    est, err = jackknife(kurtosis, x, bin_size=5)

    assert est == pytest.approx(kurtosis(x), rel=1e-12)
    assert err == pytest.approx(delete_one_block(kurtosis, [x], 5), rel=1e-10)


def test_jackknife_resamples_columns_together_and_passes_constants():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(400) + 2.0
    y = x + 0.1 * rng.standard_normal(400)  # strongly correlated with x
    w = np.arange(400.0)
    seen = []

    def ratio(x, y, *, scale, w):
        seen.append(len(w))
        return scale * np.mean(x) / np.mean(y)

    est, err = jackknife(ratio, x, y, scale=3.0, w=w, bin_size=4)

    assert est == pytest.approx(3.0 * x.mean() / y.mean(), rel=1e-12)
    # Deleting the same block from both columns is what lets the correlated
    # fluctuations cancel; resampling them independently would not.
    reference = delete_one_block(
        lambda x, y: 3.0 * np.mean(x) / np.mean(y), [x, y], 4
    )
    assert err == pytest.approx(reference, rel=1e-10)
    # Constants are never resampled.
    assert set(seen) == {400}


def test_jackknife_rejects_misaligned_columns():
    with pytest.raises(ValueError, match="aligned|length"):
        jackknife(lambda x, y: np.mean(x * y), np.ones(10), np.ones(11))


def test_jackknife_needs_at_least_two_bins():
    with pytest.raises(ValueError, match="2 bins"):
        jackknife(np.mean, np.ones(10), bin_size=6)


def test_jackknife_vector_valued_estimator():
    # A column of correlators C(t) has shape (n, T); an estimator returning
    # shape (T,) gets an estimate and error of shape (T,), each component
    # identical to jackknifing that component alone.
    C = np.random.default_rng(6).standard_normal((300, 5))
    est, err = jackknife(lambda C: C.mean(axis=0), C, bin_size=3)

    assert est.shape == err.shape == (5,)
    for t in range(5):
        e_t, d_t = jackknife(np.mean, C[:, t], bin_size=3)
        assert est[t] == pytest.approx(e_t, rel=1e-12)
        assert err[t] == pytest.approx(d_t, rel=1e-12)


def test_jackknife_cost_does_not_grow_with_column_length():
    # KNOWN FAILURE, pending design.  At the default bin_size=1 every sample is
    # its own bin, so N samples cost N evaluations on N samples: quadratic,
    # 0.6 s at N = 32k and ~10 min at N = 1e6.  The number of deletions should
    # be bounded independent of N -- e.g. a cap on the number of bins -- but
    # what the default should be is still to be decided.  The budget below is
    # a placeholder; the estimator aborts past it so this fails fast.
    calls = []

    def counted_mean(x):
        calls.append(1)
        if len(calls) > 1001:
            raise RuntimeError("jackknife exceeded its evaluation budget")
        return np.mean(x)

    jackknife(counted_mean, np.zeros(100_000))


def test_jackknife_works_in_float64():
    x = np.random.default_rng(7).standard_normal(200).astype(np.float32)
    est, err = jackknife(np.mean, x)

    assert np.asarray(est).dtype == np.float64
    assert np.asarray(err).dtype == np.float64
    assert est == pytest.approx(np.mean(x.astype(np.float64)), rel=1e-14)


# --- tau_int ------------------------------------------------------------------


def test_tau_int_of_independent_samples_is_one_half():
    x = np.random.default_rng(8).standard_normal(50_000)
    report = tau_int(x)

    assert isinstance(report, TauInt)
    assert report.resolved
    assert report.tau == pytest.approx(0.5, abs=0.05)


def test_tau_int_of_ar1_series():
    rho = 0.8
    report = tau_int(ar1(rho, 100_000))
    exact = (1 + rho) / (2 * (1 - rho))  # 4.5

    assert report.resolved
    assert abs(report.tau - exact) < 4 * report.error
    assert report.gamma[0] == 1.0
    assert report.gamma[1] == pytest.approx(rho, abs=0.02)


def test_tau_int_error_is_madras_sokal():
    report = tau_int(ar1(0.5, 20_000, seed=1))
    expected = report.tau * np.sqrt(2 * (2 * report.window + 1) / 20_000)

    assert report.error == pytest.approx(expected, rel=1e-12)


def test_tau_int_of_constant_series_is_finite():
    # A frozen observable has nothing to correlate: report tau = 1/2, not nan.
    report = tau_int(np.full(1000, 3.0))

    assert report.tau == 0.5
    assert np.all(np.isfinite(report.gamma))


def test_tau_int_flags_an_unresolved_window():
    # tau ~ 1000 on a chain of 500: the window cannot close, and the result is
    # only a lower bound.
    report = tau_int(ar1(0.999, 500, seed=2))

    assert not report.resolved


def test_tau_int_repr_is_value_with_error():
    # Prints as tau with its error in parenthetical notation (two significant
    # digits of error), plus the window -- not the raw tuple with gamma.
    gamma = np.ones(10)
    assert repr(TauInt(4.5231, 0.1834, 23, True, gamma)) == "TauInt(tau=4.52(18), window=23)"
    assert repr(TauInt(812.4, 73.2, 250, False, gamma)) == (
        "TauInt(tau>=812(73), window=250, unresolved)"
    )
    assert repr(TauInt(0.5, 0.0, 0, True, gamma)) == "TauInt(tau=0.5(0), window=0)"
    assert str(TauInt(4.5231, 0.1834, 23, True, gamma)) == "TauInt(tau=4.52(18), window=23)"
