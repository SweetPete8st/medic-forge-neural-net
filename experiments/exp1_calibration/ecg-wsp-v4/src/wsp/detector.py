"""Stage 1: QRS (beat) detector, written in weight space.

Design: Pan & Tompkins (1985) compiled into a fixed conv net. Every constant
below is stated with its reason. Nothing here was tuned against data; the
ledger records any later change.

Signal path (fs = 360 Hz unless stated):

    x        raw single lead, mV
    bp       5-15 Hz bandpass           Conv1d k=361      QRS energy band, kills baseline wander and most T/P energy
    d        5-point derivative         Conv1d k=5        slope emphasises the steep QRS
    |d|      full-wave rectification    Conv1d 1->2, ReLU, Conv1d 2->1
    m        moving-window integration  Conv1d k=55       ~150 ms window: one QRS becomes one hump
    L        local signal level         MaxPool k=721     max of m over +/-1.0 s: the current beat size
    F        floor level                MaxPool k=3601    max of m over +/-5.0 s: a longer-memory size
    c        m - 0.30 L - 0.05 F - 0.15 Conv1d 3->1 k=1   Pan-Tompkins threshold, 0.25-0.33 of signal peak,
                                                       plus an absolute floor (v1.2) of ~0.05 mV QRS
    s        clip(200 c, 0, 1)          two layers        'above threshold' indicator
    M        max of m over +/-200 ms    MaxPool k=145     refractory window: no two beats closer than 200 ms
    P        ReLU(1 + 1e6 (m - M))      Conv1d 2->1 k=1   'is the local maximum' indicator (exactly 1 at the argmax)
    beat0    ReLU(s + P - 1), binarised                   both conditions true
    prev     max of beat0 over the 50 ms before t         CausalMaxPool k=18 (v1.1)
    beat     ReLU(beat0 - prev)                            drop a second fire within 50 ms of a first

v1.1 (double-fire suppression): the local-max gate fires at every sample whose
value is within 1e-6 of its 400 ms window maximum. At a rounded hump top two
samples can tie to that precision, giving two fires 3 ms apart for one beat
(seen once in 1227 beats on 03700181). The causal max-pool over the preceding
50 ms keeps the first fire and removes the second; no physiological beat
follows another within 50 ms, so nothing real is lost. Logic only, no threshold.

Deviations from Pan-Tompkins, with reasons:
  * Rectification (|d|) instead of squaring (d^2). A ReLU network cannot
    square a signal in a scale-free way; |d| is scale-free. Costs some
    QRS-vs-T discrimination.
  * Fixed-fraction threshold off a local max instead of the running adaptive
    signal/noise estimates. A feed-forward net has no running state; the
    max-pool over 2 s is the stateless stand-in.
  * No search-back for missed beats, no T-wave slope rule. Those are
    sequential logic. If they turn out to matter, that is a finding.
"""
import numpy as np

from .filters import windowed_sinc_bandpass
from .layers import Conv1d, ReLU, MaxPool1d, CausalMaxPool1d, Concat, Net, clip01, and_gate


def odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


DESIGN = dict(
    fs=360,
    bp_lo=5.0,            # Hz. Pan-Tompkins passband lower edge
    bp_hi=15.0,           # Hz. upper edge
    bp_taps_sec=1.0,      # 1 s kernel -> Hamming transition ~3.3 Hz, clean enough edges
    mwi_sec=0.150,        # 150 ms: longest normal QRS plus margin (Pan-Tompkins used 150 ms)
    mwi_gain=10.0,        # scales m so a 1 mV QRS gives m ~ 1-5; purely a units choice
    level_sec=1.0,        # +/-1 s local level: any sample is within 1 s of a beat if HR >= 30
    floor_sec=5.0,        # +/-5 s floor level
    alpha=0.30,           # threshold fraction of local level (P&T effective ~0.25-0.33)
    beta=0.05,            # small floor fraction so flat-line/noise stretches cannot trigger
    m_floor=0.15,         # v1.2 absolute floor in MWI units (~0.05 mV QRS): pure noise has no QRS
    K_thr=200.0,          # threshold-gate steepness: soft region 1/200 of m units
    refractory_sec=0.200, # physiological refractory: no two QRS within 200 ms
    K_peak=1e6,           # local-max gate steepness: only the exact argmax fires
    K_bin=100.0,          # binariser steepness
    dedup_sec=0.050,      # v1.1: a second fire within 50 ms of a first is the same beat
)


def build_detector(fs=None, design=None):
    D = dict(DESIGN)
    if design:
        D.update(design)
    if fs is not None:
        D["fs"] = fs
    fs = D["fs"]

    # --- bandpass
    n_bp = odd(D["bp_taps_sec"] * fs)
    h_bp = windowed_sinc_bandpass(fs, D["bp_lo"], D["bp_hi"], n_bp)
    bandpass = Conv1d(h_bp[None, None, :], name="bandpass")

    # --- derivative: y[t] = -x[t-2] - 2x[t-1] + 2x[t+1] + x[t+2]  (P&T, unscaled)
    deriv = Conv1d(np.array([[[-1.0, -2.0, 0.0, 2.0, 1.0]]]), name="deriv")

    # --- rectify: |d| = ReLU(d) + ReLU(-d)
    split = Conv1d(np.array([[[1.0]], [[-1.0]]]), name="rect.split")
    merge = Conv1d(np.array([[[1.0], [1.0]]]), name="rect.merge")

    # --- moving window integration
    n_mwi = odd(D["mwi_sec"] * fs)
    mwi = Conv1d(np.full((1, 1, n_mwi), D["mwi_gain"] / n_mwi), name="mwi")

    # --- levels
    level = MaxPool1d(odd(2 * D["level_sec"] * fs), name="level")
    floor = MaxPool1d(odd(2 * D["floor_sec"] * fs), name="floor")

    # --- threshold gate: c = m - alpha L - beta F ; s = clip(K c, 0, 1)
    thr = Conv1d(np.array([[[1.0], [-D["alpha"]], [-D["beta"]]]]), np.array([-D["m_floor"]]), name="thr")
    s_up, s_down = clip01(D["K_thr"], name="s")

    # --- local max gate: P = ReLU(1 + K (m - M))
    refr = MaxPool1d(odd(2 * D["refractory_sec"] * fs), name="refractory")
    peak = Conv1d(np.array([[[D["K_peak"]], [-D["K_peak"]]]]), bias=np.array([1.0]),
                  name="peak")

    # --- AND and binarise
    both = and_gate(2, name="beat")
    b_up, b_down = clip01(D["K_bin"], bias=-0.5 * D["K_bin"], name="bin")

    # --- v1.1 double-fire suppression
    prev = CausalMaxPool1d(int(round(D["dedup_sec"] * fs)), name="prev")
    dedup = Conv1d(np.array([[[1.0], [-1.0]]]), name="dedup")

    ops = [
        (bandpass, ["x"], "bp"),
        (deriv, ["bp"], "d"),
        (split, ["d"], "d2"), (ReLU(), ["d2"], "d2r"), (merge, ["d2r"], "absd"),
        (mwi, ["absd"], "m"),
        (level, ["m"], "L"),
        (floor, ["m"], "F"),
        (Concat(), ["m", "L", "F"], "mLF"), (thr, ["mLF"], "c"),
        (s_up, ["c"], "s2"), (ReLU(), ["s2"], "s2r"), (s_down, ["s2r"], "s"),
        (refr, ["m"], "M"),
        (Concat(), ["m", "M"], "mM"), (peak, ["mM"], "p0"), (ReLU(), ["p0"], "P"),
        (Concat(), ["s", "P"], "sP"), (both, ["sP"], "b0"), (ReLU(), ["b0"], "beat_soft"),
        (b_up, ["beat_soft"], "b2"), (ReLU(), ["b2"], "b2r"), (b_down, ["b2r"], "beat0"),
        (prev, ["beat0"], "prev"),
        (Concat(), ["beat0", "prev"], "bp2"), (dedup, ["bp2"], "dd0"), (ReLU(), ["dd0"], "beat"),
    ]
    net = Net(ops, inputs=("x",), output="beat")
    net.design = D
    net.bandpass_kernel = h_bp
    return net


def beat_samples(beat_train, thresh=0.5):
    """Convert the per-sample beat indicator to a list of sample indices."""
    return np.flatnonzero(beat_train[0] > thresh)
