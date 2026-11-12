"""Statistics on Monte Carlo columns: error estimates and autocorrelation.

A *column* is an array whose leading axis is Monte Carlo time: one entry per
measured configuration, as returned by `Chain.column`.  Trailing axes, if any,
are the shape of the measurement itself (a correlator C(t) gives a column of
shape (n, T)).

Everything here is host-side NumPy at float64.  The jackknife multiplies any
rounding error in an estimator by the number of bins, so working precision is
not good enough.

    U, dU = jackknife(binder, m, bin_size=50)
    xi, dxi = jackknife(xi_2nd, m, Gp, volume=V, L=L, bin_size=50)
    report = tau_int(m)
"""

from typing import NamedTuple

import numpy as np

# Sokal's window factor: a bias/variance trade-off, not a correctness knob.
DEFAULT_WINDOW_FACTOR = 5.0


def jackknife(estimator, *columns, bin_size=1, **constants):
    # (estimate, error) for `estimator(*columns, **constants)`.
    #
    # `estimator` is an ordinary function of samples -- the same function that
    # gives the plain estimate -- e.g. `lambda m: np.mean(m**4)`.  Positional
    # `columns` are resampled together, so they must be aligned; keyword
    # `constants` are passed through untouched.
    #
    # Each column is cut into consecutive blocks of `bin_size` (a leftover
    # partial block is dropped), and the estimator is re-evaluated on the raw
    # samples with one block deleted at a time.  Deleting blocks of raw
    # samples, rather than binning first, is what keeps nonlinear moments
    # right: the mean of m**4 is not the fourth power of bin means.
    #
    # The estimate is the estimator on the full (truncated) sample; the error
    # is the jackknife standard error.  Both are float64, with the shape of
    # the estimator's output.
    columns = [np.asarray(c, dtype=np.float64) for c in columns]
    n = len(columns[0])
    if any(len(c) != n for c in columns):
        raise ValueError(
            f"columns must be aligned, but have lengths {[len(c) for c in columns]}."
        )

    n_bins = n // bin_size
    if n_bins < 2:
        raise ValueError(
            f"jackknife needs at least 2 bins; {n} samples at bin_size={bin_size} "
            f"give {n_bins}."
        )
    columns = [c[: n_bins * bin_size] for c in columns]

    full = estimator(*columns, **constants)
    block = np.arange(n_bins * bin_size) // bin_size
    deleted = np.array(
        [
            estimator(*(c[block != i] for c in columns), **constants)
            for i in range(n_bins)
        ],
        dtype=np.float64,
    )
    # (n-1)/n sum (theta_i - theta_bar)^2
    err = np.sqrt((n_bins - 1) * np.var(deleted, axis=0))

    return np.asarray(full, dtype=np.float64)[()], err[()]


class TauInt(NamedTuple):
    # Integrated autocorrelation time of one column.  If `resolved` is False,
    # the window never closed and `tau` is only a lower bound.
    tau: float
    error: float
    window: int
    resolved: bool
    gamma: np.ndarray  # normalized autocorrelation, gamma[0] = 1

    def __repr__(self):
        # TauInt(tau=4.52(18), window=23); `tau>=` and `unresolved` mark a
        # lower bound.
        bound = "=" if self.resolved else ">="
        flag = "" if self.resolved else ", unresolved"
        return f"TauInt(tau{bound}{_paren(self.tau, self.error)}, window={self.window}{flag})"


def _paren(value, error):
    # value(error) with two significant digits of error: 4.52(18), 812(73).
    if not error > 0:
        return f"{value:g}(0)"
    decimals = max(1 - int(np.floor(np.log10(error))), 0)
    return f"{value:.{decimals}f}({round(error * 10**decimals)})"


def tau_int(series, window_factor=DEFAULT_WINDOW_FACTOR, max_lag=None):
    # tau_int = 1/2 + sum_{t>=1} Gamma(t)/Gamma(0), summed to Sokal's window:
    # the smallest W with W >= window_factor * tau_int(W).  Error is
    # Madras-Sokal, tau * sqrt(2 (2W + 1) / N).  `max_lag` defaults to N/2.
    # A constant series has nothing to correlate and reports tau = 1/2 exactly.
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    max_lag = n // 2 if max_lag is None else max_lag

    # Autocovariance at every lag at once, zero-padded past 2n so that the
    # circular correlation cannot fold the tail back onto the head.
    size = 1 << int(np.ceil(np.log2(2 * n)))
    ft = np.fft.rfft(x - x.mean(), n=size)
    lags = np.arange(max_lag + 1)
    cov = np.fft.irfft(ft * np.conj(ft), n=size)[lags] / (n - lags)

    if cov[0] <= 0:
        gamma = np.zeros(max_lag + 1)
        gamma[0] = 1.0
        return TauInt(0.5, 0.0, 0, True, gamma)
    gamma = cov / cov[0]

    # running[W - 1] is tau_int summed to window W.
    running = 0.5 + np.cumsum(gamma[1:])
    windows = lags[1:]
    closed = np.flatnonzero(windows >= window_factor * running)
    resolved = closed.size > 0
    window = int(windows[closed[0]]) if resolved else max_lag

    # First-order correction for the bias from subtracting the sample mean.
    tau = max(running[window - 1] * (1 + (2 * window + 1) / n), 0.5)
    error = tau * np.sqrt(2 * (2 * window + 1) / n)

    return TauInt(float(tau), float(error), window, resolved, gamma)
