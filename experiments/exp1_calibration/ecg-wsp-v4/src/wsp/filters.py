"""Hand-designed FIR kernels. Each function is a formula, not a fit."""
import numpy as np


def hamming(n):
    k = np.arange(n)
    return 0.54 - 0.46 * np.cos(2 * np.pi * k / (n - 1))


def windowed_sinc_bandpass(fs, f_lo, f_hi, n_taps, f_ref=None):
    """Classic windowed-sinc bandpass: ideal bandpass impulse response
    (difference of two sincs) multiplied by a Hamming window.

    Gain is normalised to exactly 1.0 at f_ref (default: band centre), so the
    filter's output is in the input's units.
    """
    assert n_taps % 2 == 1
    m = (n_taps - 1) / 2
    n = np.arange(n_taps) - m
    # ideal lowpass at f_hi minus ideal lowpass at f_lo (np.sinc is sin(pi x)/(pi x))
    h = (2 * f_hi / fs) * np.sinc(2 * f_hi * n / fs) - (2 * f_lo / fs) * np.sinc(2 * f_lo * n / fs)
    h *= hamming(n_taps)
    f_ref = (f_lo + f_hi) / 2 if f_ref is None else f_ref
    gain = np.abs(np.sum(h * np.exp(-2j * np.pi * f_ref * n / fs)))
    return h / gain


def frequency_response(h, fs, freqs):
    n = np.arange(len(h)) - (len(h) - 1) / 2
    return np.array([np.abs(np.sum(h * np.exp(-2j * np.pi * f * n / fs))) for f in freqs])
