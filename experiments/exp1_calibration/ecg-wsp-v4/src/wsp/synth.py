"""Synthetic ECG strips for sanity checks. Toy-level morphology: Gaussian
bumps for P, Q, R, S, T. Used only to catch implementation bugs, never to
set weights."""
import numpy as np


def bump(t, center, width, amp):
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)


def strip(fs=360, duration=10.0, rr=0.8, rr_jitter=0.0, p_amp=0.15, qrs_width=0.09,
          r_amp=1.0, t_amp=0.3, wander_amp=0.2, noise_sd=0.02, seed=0, start=0.3):
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    beats = []
    tb = start
    while tb < duration:
        beats.append(tb)
        # P wave 160 ms before R, 80 ms wide
        x += bump(t, tb - 0.16, 0.025, p_amp)
        # QRS: Q, R, S; sigma scales with requested width (narrow 0.09 s, wide 0.16 s)
        sig = qrs_width / 6
        x += bump(t, tb - 1.5 * sig, sig * 0.6, -0.1 * r_amp)
        x += bump(t, tb, sig, r_amp)
        x += bump(t, tb + 1.5 * sig, sig * 0.6, -0.25 * r_amp)
        # T wave 300 ms after R, ~160 ms wide
        x += bump(t, tb + 0.30, 0.05, t_amp)
        tb += rr + rng.normal(0, rr_jitter)
    x += wander_amp * np.sin(2 * np.pi * 0.3 * t + 1.0)
    x += rng.normal(0, noise_sd, n)
    return x, np.array(beats)
