"""Stage 2: rate, regularity and QRS-width heads, written in weight space.

These sit on top of the Stage 1 beat train. Everything is standard layers:
Conv1d (kernel 1 = a linear layer), ReLU, MaxPool1d, global sum / max pooling,
a max over channels, and one fixed positional ramp. The window is fixed length
(10 s) so the global reductions produce one number per head.

Three constructions do the work that a feed-forward net cannot do natively:

1. Integer division by a count (span / (N-1), width_sum / N, best / (N-1)).
   N is an integer, so a one-hot of N is exact:
       u_k = ReLU(N - k), v_k = ReLU(k - N), onehot_k = ReLU(1 - u_k - v_k)
   A quantity q is then gated into the k-th slot, z_k = ReLU(q - BIG(1 - onehot_k)),
   and the division is a fixed linear readout: q / N = sum_k z_k / k.

2. First and last beat time (for the mean RR interval), using the ramp r[t]:
       last  = max_t ReLU(r[t]     - BIG(1 - beat[t]))      = t_last
       first = max_t ReLU(T - r[t] - BIG(1 - beat[t]))      = T - t_first
   so span = last + first - T. No beats: span = -T, later clipped to 0.

3. A reciprocal, 60 / RR, as a one-hidden-layer ReLU interpolant with knots
   every 20 ms from 0.2 s to 0.6 s, every 50 ms to 1.2 s, every 100 ms to
   2.5 s. Worst-case interpolation error is under 1 bpm (checked analytically:
   f'' h^2 / 8 with f = 60/x). Below 0.2 s it is clamped at 300 bpm; the
   output is then gated to 0 when fewer than two beats were found.

Regularity: a comb bank. For each candidate RR period P (0.25 s to 2.0 s,
2.5% steps, 85 channels), succ_P[t] = 1 if a beat lies in [t+P-w, t+P+w] with
w = 10% of P (the clinical "same RR" tolerance). cnt_P = number of beats whose
successor is at ~P. best = max over P. score = best / (N-1). A regular rhythm
scores 1.0 at its own period; atrial fibrillation spreads successors across
periods and scores roughly 0.3-0.5; a sinus strip with two PVCs loses about
four successions out of eleven. Flag "regular" if score >= 0.75: halfway
between the AF ceiling and the sinus floor, chosen before any data was seen.

QRS width: a wide-band branch (1-30 Hz, to keep the morphology of a broad QRS
that the 5-15 Hz detector band would thin out), derivative, rectify, then
"active" = samples whose slope exceeds 20% of the beat's own peak slope
(peak taken over +/-120 ms). The count of active samples within +/-120 ms of
the beat, in ms, is the width measure. It is the duration of steep slope, so
it runs under the clinical QRS duration; from the shape of a QRS (two
near-triangular lobes with a brief low-slope gap) the steep portion is
roughly 60-75% of the complex. 120 ms clinical -> about 85 ms of steep slope.
Wide flag at 85 ms, set before seeing data.
"""
import numpy as np

from .detector import build_detector, odd
from .filters import windowed_sinc_bandpass
from .layers import (Conv1d, ReLU, MaxPool1d, Concat, GlobalSum, GlobalMax,
                     ChannelMax, Ramp, Net, clip01)

BIG = 1000.0


def onehot_block(prefix, n_max):
    """Returns (layer_uv, layer_onehot) mapping N (1,1) -> onehot (n_max+1, 1)."""
    K = n_max + 1
    w = np.zeros((2 * K, 1, 1))
    b = np.zeros(2 * K)
    for k in range(K):
        w[2 * k, 0, 0] = 1.0;  b[2 * k] = -k          # u_k = ReLU(N - k)
        w[2 * k + 1, 0, 0] = -1.0; b[2 * k + 1] = k   # v_k = ReLU(k - N)
    uv = Conv1d(w, b, name=f"{prefix}.uv")
    w2 = np.zeros((K, 2 * K, 1))
    for k in range(K):
        w2[k, 2 * k, 0] = -1.0
        w2[k, 2 * k + 1, 0] = -1.0
    oh = Conv1d(w2, np.ones(K), name=f"{prefix}.onehot")
    return uv, oh


def gated_divide(prefix, n_max, offset=0, big=BIG):
    """Layers mapping concat[q, onehot] -> q / (N - offset).
    z_k = ReLU(q - big(1 - onehot_k)) for k > offset; out = sum_k z_k/(k-offset).
    `big` must exceed the largest value q can take, or the gate leaks."""
    K = n_max + 1
    ks = [k for k in range(K) if k - offset >= 1]
    w = np.zeros((len(ks), 1 + K, 1))
    b = np.full(len(ks), -big)
    for i, k in enumerate(ks):
        w[i, 0, 0] = 1.0
        w[i, 1 + k, 0] = big
    gate = Conv1d(w, b, name=f"{prefix}.gate")
    w2 = np.array([[[1.0 / (k - offset)] for k in ks]])
    read = Conv1d(w2, name=f"{prefix}.read")
    return gate, read


def reciprocal_layers(prefix, scale=60.0):
    """60 / x as a ReLU interpolant. Returns (hidden, readout), x in (1,1)."""
    knots = np.concatenate([np.arange(0.20, 0.60, 0.02), np.arange(0.60, 1.20, 0.05),
                            np.arange(1.20, 2.51, 0.10)])
    f = scale / knots
    slopes = np.diff(f) / np.diff(knots)
    coef = np.concatenate([[slopes[0]], np.diff(slopes)])  # one coefficient per knot except the last
    n = len(knots) - 1
    hidden = Conv1d(np.ones((n, 1, 1)), -knots[:n], name=f"{prefix}.knots")
    readout = Conv1d(coef[None, :, None].copy(), np.array([f[0]]), name=f"{prefix}.read")
    return hidden, readout, knots


DESIGN2 = dict(
    window_sec=10.0,
    n_max=60,             # most beats a 10 s strip can hold (HR 360)
    reg_p_min=0.25,       # s, HR 240
    reg_p_max=2.0,        # s, HR 30
    reg_step=1.025,       # 2.5% geometric spacing
    reg_tol=0.10,         # +/-10% of the period counts as the same RR
    reg_thresh=0.75,      # score at or above this = regular
    K_gate=1000.0,
    wb_lo=1.0, wb_hi=30.0, wb_taps_sec=2.0,  # wide-band branch for QRS morphology
    act_frac=0.20,        # steep = slope above 20% of the beat's peak slope
    act_half_sec=0.120,   # look +/-120 ms around the beat for steep slope
    wide_ms=85.0,         # steep-slope duration at or above this = wide complex
)


def build_stage2(fs=360, design=None, detector_design=None):
    D = dict(DESIGN2)
    if design:
        D.update(design)
    det = build_detector(fs=fs, design=detector_design)
    T_sec = D["window_sec"]
    ops = list(det.ops)

    # ---------------- count and 6-second-method rate
    ops += [(GlobalSum(), ["beat"], "N"),
            (Conv1d(np.array([[[60.0 / T_sec]]]), name="rate_count"), ["N"], "rate_count")]

    # ---------------- one-hot of N (shared)
    uv, oh = onehot_block("oh", D["n_max"])
    ops += [(uv, ["N"], "uv"), (ReLU(), ["uv"], "uvr"), (oh, ["uvr"], "oh0"), (ReLU(), ["oh0"], "onehot")]

    # ---------------- span = t_last - t_first
    ops += [(Ramp(fs), ["beat"], "ramp"),
            (Concat(), ["ramp", "beat"], "rb"),
            (Conv1d(np.array([[[1.0], [BIG]]]), np.array([-BIG]), name="glast"), ["rb"], "gl0"),
            (ReLU(), ["gl0"], "gl"), (GlobalMax(), ["gl"], "t_last"),
            (Conv1d(np.array([[[-1.0], [BIG]]]), np.array([T_sec - BIG]), name="gfirst"), ["rb"], "gf0"),
            (ReLU(), ["gf0"], "gf"), (GlobalMax(), ["gf"], "T_minus_t_first"),
            (Concat(), ["t_last", "T_minus_t_first"], "lf"),
            (Conv1d(np.array([[[1.0], [1.0]]]), np.array([-T_sec]), name="span"), ["lf"], "span0"),
            (ReLU(), ["span0"], "span")]

    # ---------------- mean RR = span / (N - 1), rate_rr = 60 / meanRR
    gate, read = gated_divide("rr", D["n_max"], offset=1)
    hid, rout, knots = reciprocal_layers("recip")
    ops += [(Concat(), ["span", "onehot"], "span_oh"), (gate, ["span_oh"], "rrz0"),
            (ReLU(), ["rrz0"], "rrz"), (read, ["rrz"], "mean_rr"),
            (hid, ["mean_rr"], "rk0"), (ReLU(), ["rk0"], "rk"), (rout, ["rk"], "rate_rr_raw")]
    # valid = clip(N - 1, 0, 1); rate_rr = ReLU(rate_rr_raw - BIG(1 - valid))
    v_up, v_down = clip01(1.0, bias=-1.0, name="valid")
    ops += [(v_up, ["N"], "v2"), (ReLU(), ["v2"], "v2r"), (v_down, ["v2r"], "valid"),
            (Concat(), ["rate_rr_raw", "valid"], "rv"),
            (Conv1d(np.array([[[1.0], [BIG]]]), np.array([-BIG]), name="rate_rr_gate"), ["rv"], "rate_rr0"),
            (ReLU(), ["rate_rr0"], "rate_rr")]

    # ---------------- regularity comb bank
    periods = []
    p = D["reg_p_min"]
    while p <= D["reg_p_max"] + 1e-9:
        periods.append(p)
        p *= D["reg_step"]
    periods = np.array(periods)
    C = len(periods)
    pmax = int(round(periods[-1] * (1 + D["reg_tol"]) * fs)) + 1
    k_succ = 2 * pmax + 1
    w = np.zeros((C, 1, k_succ))
    for c, P in enumerate(periods):
        ps = int(round(P * fs))
        tol = int(round(D["reg_tol"] * P * fs))
        w[c, 0, pmax + ps - tol: pmax + ps + tol + 1] = 1.0   # picks beat[t + P +/- w]
    succ = Conv1d(w, name="succ")
    # per-channel clip to [0,1]: two beats in one tolerance window still count once
    s_up_c = Conv1d(np.concatenate([np.eye(C)[:, :, None] * D["K_gate"]] * 2, axis=0),
                    np.concatenate([np.zeros(C), -np.ones(C)]), name="succ.clip.up")
    s_down_c = Conv1d(np.concatenate([np.eye(C)[:, :, None], -np.eye(C)[:, :, None]], axis=1),
                      name="succ.clip.down")
    # both_c = ReLU(beat + succ_c - 1): this beat has a successor at ~P
    w_and = np.zeros((C, 1 + C, 1))
    w_and[:, 0, 0] = 1.0
    for c in range(C):
        w_and[c, 1 + c, 0] = 1.0
    both = Conv1d(w_and, -np.ones(C), name="succ.and")
    gate_r, read_r = gated_divide("reg", D["n_max"], offset=1)
    r_up, r_down = clip01(D["K_gate"], bias=-D["K_gate"] * D["reg_thresh"], name="regular")
    ops += [(succ, ["beat"], "succ0"),
            (s_up_c, ["succ0"], "succ2"), (ReLU(), ["succ2"], "succ2r"), (s_down_c, ["succ2r"], "succ"),
            (Concat(), ["beat", "succ"], "bs"), (both, ["bs"], "both0"), (ReLU(), ["both0"], "both"),
            (GlobalSum(), ["both"], "cnt"), (ChannelMax(), ["cnt"], "best"),
            (Concat(), ["best", "onehot"], "best_oh"), (gate_r, ["best_oh"], "rz0"),
            (ReLU(), ["rz0"], "rz"), (read_r, ["rz"], "reg_score"),
            (r_up, ["reg_score"], "rg2"), (ReLU(), ["rg2"], "rg2r"), (r_down, ["rg2r"], "regular")]

    # ---------------- QRS width (wide-band branch)
    n_wb = odd(D["wb_taps_sec"] * fs)
    h_wb = windowed_sinc_bandpass(fs, D["wb_lo"], D["wb_hi"], n_wb)
    wb = Conv1d(h_wb[None, None, :], name="wideband")
    deriv2 = Conv1d(np.array([[[-1.0, -2.0, 0.0, 2.0, 1.0]]]), name="deriv2")
    split2 = Conv1d(np.array([[[1.0]], [[-1.0]]]), name="rect2.split")
    merge2 = Conv1d(np.array([[[1.0], [1.0]]]), name="rect2.merge")
    k_act = odd(2 * D["act_half_sec"] * fs)
    ld = MaxPool1d(k_act, name="slope_level")
    act_pre = Conv1d(np.array([[[1.0], [-D["act_frac"]]]]), name="act")
    a_up, a_down = clip01(D["K_gate"], name="act.clip")
    wsum = Conv1d(np.full((1, 1, k_act), 1000.0 / fs), name="width_ms")   # ms of steep slope near t
    wgate = Conv1d(np.array([[[1.0], [BIG]]]), np.array([-BIG]), name="width_at_beat")
    gate_w, read_w = gated_divide("wmean", D["n_max"], offset=0, big=1e5)  # width_sum can reach 60 x 240 ms
    wd_up, wd_down = clip01(D["K_gate"], bias=-D["K_gate"] * D["wide_ms"], name="wide")
    wx_up, wx_down = clip01(D["K_gate"], bias=-D["K_gate"] * D["wide_ms"], name="any_wide")
    ops += [(wb, ["x"], "wb"), (deriv2, ["wb"], "d2"),
            (split2, ["d2"], "d2s"), (ReLU(), ["d2s"], "d2sr"), (merge2, ["d2sr"], "absd2"),
            (ld, ["absd2"], "Ld"),
            (Concat(), ["absd2", "Ld"], "dLd"), (act_pre, ["dLd"], "act0"),
            (a_up, ["act0"], "act2"), (ReLU(), ["act2"], "act2r"), (a_down, ["act2r"], "act"),
            (wsum, ["act"], "width_ms"),
            (Concat(), ["width_ms", "beat"], "wbeat"), (wgate, ["wbeat"], "wab0"), (ReLU(), ["wab0"], "width_at_beat"),
            (GlobalMax(), ["width_at_beat"], "width_max"),
            (GlobalSum(), ["width_at_beat"], "width_sum"),
            (Concat(), ["width_sum", "onehot"], "wsum_oh"), (gate_w, ["wsum_oh"], "wz0"),
            (ReLU(), ["wz0"], "wz"), (read_w, ["wz"], "width_mean"),
            (wd_up, ["width_mean"], "wd2"), (ReLU(), ["wd2"], "wd2r"), (wd_down, ["wd2r"], "wide"),
            (wx_up, ["width_max"], "wx2"), (ReLU(), ["wx2"], "wx2r"), (wx_down, ["wx2r"], "any_wide")]

    net = Net(ops, inputs=("x",), output="rate_rr")
    net.design = D
    net.detector_design = det.design
    net.periods = periods
    net.knots = knots
    net.outputs = ["N", "rate_count", "mean_rr", "rate_rr", "reg_score", "regular",
                   "width_mean", "width_max", "wide", "any_wide"]
    return net


def read_outputs(t, net):
    return {k: float(t[k][0, 0]) for k in net.outputs}
