import numpy as np
from typing import Callable, Tuple
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal.signaltools import correlate, correlation_lags
from scipy.interpolate import UnivariateSpline


SignalFunction = Callable[[np.ndarray, float, float, float, float], np.ndarray]


class UnfittableDataError(Exception):
    """The fit can't be performed over the input data."""

    pass


def average_vpp(y: np.ndarray) -> Tuple[float, float]:
    mean_max_vpp = y[y > np.percentile(y, 95)].mean()
    mean_min_vpp = y[y < np.percentile(y, 5)].mean()
    std_max_vpp = y[y > np.percentile(y, 95)].std()
    std_min_vpp = y[y < np.percentile(y, 5)].std()
    mean_vpp = mean_max_vpp - mean_min_vpp
    std_vpp = std_max_vpp + std_min_vpp

    return mean_vpp, std_vpp


def estimate_fft(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    yf = rfft(y)
    xf = rfftfreq(y.size, float(np.mean(x[1:] - x[0:-1])))
    peaks, _ = find_peaks(np.abs(yf), height=np.mean(np.abs(yf)))

    if len(peaks) > 0:
        freq = xf[peaks[0]]
        angle = np.angle(yf[peaks[0]])
        return freq, angle
    else:
        return np.nan, np.nan


def estimate_roots(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    spl = UnivariateSpline(x, y - np.mean(y), s=1 / len(y))
    zeros = spl.roots()

    if len(zeros) > 1:
        freq = float(1 / np.mean(zeros[1:] - zeros[0:-1]) / 2)
        angle = 2 * np.pi * zeros[0] * freq
        angle = (angle - np.pi) % (2 * np.pi) - np.pi
        return freq, angle
    else:
        return np.nan, np.nan


def estimate_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    xc = correlation_lags(y.size, y.size) * np.mean(x[1:] - x[0:-1])
    yc = correlate(y, y)
    peaks, _ = find_peaks(yc)
    if len(peaks) > 1:
        freq = float(np.mean(1 / (xc[peaks[1:]] - xc[peaks[0:-1]])))
        x_shift = float(xc[np.argmax(yc)])
        angle = 2 * np.pi * x_shift * freq
        angle = (angle - np.pi) % (2 * np.pi) - np.pi
        return freq, angle
    else:
        return np.nan, np.nan


def estimate_phase_correlation(x: np.ndarray, y: np.ndarray, freq: float) -> float:
    mean_vpp, std_vpp = average_vpp(y)
    y_without_phase = sine(x, mean_vpp / 2, freq, 0.0, float(y.mean()))
    xc = correlation_lags(x.size, x.size) * (x[1] - x[0])
    yc = correlate(y, y_without_phase)
    x_shift = xc[np.argmax(yc)]
    angle = 2 * np.pi * x_shift * freq
    return angle


def sine(
    x: np.ndarray, a: float, freq: float, phase: float, offset: float
) -> np.ndarray:
    return a * np.sin(x * 2 * np.pi * freq + phase) + offset


def fit_signal(
    f: SignalFunction, x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    mean_vpp, std_vpp = average_vpp(y)
    estimated_params = np.asarray(
        [estimate_fft(x, y), estimate_roots(x, y), estimate_correlation(x, y)]
    )

    _params, _std, _rmse = [], [], []
    for freq, angle in estimated_params:
        if freq is np.nan or angle is np.nan:
            _params.append(np.nan)
            _std.append(np.nan)
            _rmse.append(np.nan)
        else:
            p0 = [mean_vpp / 2, freq, angle, y.mean()]
            p_opt, p_cov = curve_fit(f, x, y, p0=p0)
            std = np.sqrt(np.diag(p_cov))
            rmse = np.sqrt(np.mean((f(x, *p_opt) - y) ** 2))
            _params.append(p_opt)
            _std.append(std)
            _rmse.append(rmse)

    if np.all(np.isnan(_rmse)):
        raise UnfittableDataError("This data can't be fitted by this method")

    min_index = np.nanargmin(_rmse)
    if _params[min_index][0] < 0:
        _params[min_index][0] = np.abs(_params[min_index][0])
        _params[min_index][2] += np.pi

    _params[min_index][2] = (_params[min_index][2] - np.pi) % (2 * np.pi) - np.pi

    return _params[min_index], _std[min_index], _rmse[min_index]
