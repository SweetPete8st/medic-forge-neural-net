"""Stage 3: P-wave detection head, written in weight space.

A P wave is a small smooth hump 80-120 ms wide that precedes the QRS by the
PR interval (120-200 ms from P onset to QRS onset, up to ~300 ms in
first-degree block). The head asks, for each detected beat: is there exactly
one such hump in the diastolic interval before it, and does that hump sit in
the PR window? That mirrors how a reader checks a strip: one P before every
QRS, on a flat baseline.

    resp      matched filter on the raw lead        Conv1d k=55   zero-mean Gaussian (sigma 20 ms); unit gain for an
                                                                   ideal P, zero response to offsets and linear ramps
    ismax     resp equals its max over +/-60 ms     MaxPool k=43 + steep ReLU gate (same construction as the beat detector)
    above     resp >= 0.05 mV                       the smallest hump that is visibly a P at standard gain
    bump      ismax AND above                       a candidate atrial hump, one sample per hump
    n_dia     bumps in [R-400, R-131] ms            lag-box conv on the bump train: the diastolic interval
    n_pr      bumps in [R-292, R-131] ms            the PR window inside it
    one       n_dia == 1                            ReLU(1 - |n_dia - 1|): exactly one hump in diastole
    in_pr     n_pr >= 1                             and it is where a P belongs
    prevclose previous beat within 700 ms           box over lags [200, 700] ms on the beat train
    assess    beat AND NOT prevclose                above ~86 bpm the previous T wave's peak enters the
                                                    diastolic window, so P is "not assessable", not "absent"
    pdet      assess AND one AND in_pr
    p_frac    sum(pdet) / sum(assess)               one-hot gated division
    p_present p_frac >= 0.6

Why the PR window ends at R-131 ms: the kernel spans +/-75 ms and the QRS
about +/-50 ms around R, so QRS energy leaks into resp until about R-125 ms.
P waves with PR under ~130 ms are missed by design.

Why "exactly one hump" instead of "hump above baseline": v3 used a contrast
test against a baseline window [R-450, R-300]; on synthetic strips at 95 bpm
the previous T wave landed in that window and killed every P. Counting humps
needs no baseline estimate. Atrial fibrillation fills the diastolic window
with 2-3 f-wave humps and fails "exactly one"; fine AF under 0.05 mV never
makes a hump and fails "in_pr".

Expected failure modes, stated before running on real data: atrial flutter
(F waves every 200 ms) will give exactly one hump about half the time; a
tall U wave or a notched/biphasic P can read as two humps; muscle noise can
add humps; sinus rhythm above 86 bpm is reported as not assessable.
"""
import numpy as np

from .heads import build_stage2, onehot_block, gated_divide
from .layers import Conv1d, ReLU, MaxPool1d, Concat, GlobalSum, Net, clip01, and_gate


def lag_box_conv(lag_lo, lag_hi, name):
    """y[t] = sum_{tau = lag_lo..lag_hi} x[t - tau]  (lags in samples)."""
    pad = lag_hi
    w = np.zeros((1, 1, 2 * pad + 1))
    w[0, 0, pad - lag_hi: pad - lag_lo + 1] = 1.0
    return Conv1d(w, name=name)


DESIGN3 = dict(
    p_sigma_sec=0.020,      # Gaussian sigma of the P template (P waves ~80-120 ms wide)
    p_support_sec=0.075,    # kernel half-support
    hump_half_sec=0.060,    # a hump is a local max of resp over +/-60 ms
    p_amp_mv=0.05,          # smallest hump that counts
    dia_lo_sec=0.131, dia_hi_sec=0.400,   # diastolic window before R
    pr_lo_sec=0.131, pr_hi_sec=0.292,     # PR window inside it
    prev_lo_sec=0.200, prev_hi_sec=0.700, # a previous beat this close makes P unassessable
    p_frac_thresh=0.60,
    K=1000.0,
    K_peak=1e6,
)


def build_stage3(fs=360, design=None, design2=None, detector_design=None):
    D = dict(DESIGN3)
    if design:
        D.update(design)
    s2 = build_stage2(fs=fs, design=design2, detector_design=detector_design)
    ops = list(s2.ops)
    n_max = s2.design["n_max"]
    BIG = 1000.0

    # matched filter
    sig = D["p_sigma_sec"] * fs
    half = int(round(D["p_support_sec"] * fs))
    n = np.arange(-half, half + 1)
    g = np.exp(-0.5 * (n / sig) ** 2)
    k = g - g.mean()
    k = k / np.sum(k * g)                      # unit gain for the ideal P
    pmf = Conv1d(k[None, None, :], name="p_matched")

    # hump = local max of resp that exceeds the amplitude floor
    k_h = 2 * int(round(D["hump_half_sec"] * fs)) + 1
    hpool = MaxPool1d(k_h, name="hump_pool")
    ismax = Conv1d(np.array([[[D["K_peak"]], [-D["K_peak"]]]]), np.array([1.0]), name="hump.ismax")
    above = Conv1d(np.array([[[1.0]]]), np.array([-D["p_amp_mv"]]), name="hump.above.pre")
    ab_up, ab_down = clip01(D["K"], name="hump.above")
    hand = and_gate(2, name="hump.and")
    hb_up, hb_down = clip01(100.0, bias=-50.0, name="hump.bin")   # binarise at 0.5

    # counts in windows
    dia = lag_box_conv(int(round(D["dia_lo_sec"] * fs)), int(round(D["dia_hi_sec"] * fs)), "n_dia")
    prw = lag_box_conv(int(round(D["pr_lo_sec"] * fs)), int(round(D["pr_hi_sec"] * fs)), "n_pr")
    # one = ReLU(1 - ReLU(n_dia - 1) - ReLU(1 - n_dia))
    one_uv = Conv1d(np.array([[[1.0]], [[-1.0]]]), np.array([-1.0, 1.0]), name="one.uv")
    one_rd = Conv1d(np.array([[[-1.0], [-1.0]]]), np.array([1.0]), name="one.read")
    ip_up, ip_down = clip01(D["K"], name="in_pr")

    # assessable beats
    prev = lag_box_conv(int(round(D["prev_lo_sec"] * fs)), int(round(D["prev_hi_sec"] * fs)), "prevclose")
    pv_up, pv_down = clip01(D["K"], name="prevclose.clip")
    assess = Conv1d(np.array([[[1.0], [-1.0]]]), name="assess")   # beat - prevclose, then ReLU

    # AND of three, counts, division
    pand = and_gate(3, name="pdet")
    uv, oh = onehot_block("pa", n_max)
    gate, read = gated_divide("pfrac", n_max, offset=0, big=BIG)
    pp_up, pp_down = clip01(D["K"], bias=-D["K"] * D["p_frac_thresh"], name="p_present")

    ops += [(pmf, ["x"], "p_resp"),
            (hpool, ["p_resp"], "p_hpool"),
            (Concat(), ["p_resp", "p_hpool"], "p_rh"), (ismax, ["p_rh"], "p_im0"), (ReLU(), ["p_im0"], "p_ismax"),
            (above, ["p_resp"], "p_ab0"), (ab_up, ["p_ab0"], "p_ab2"), (ReLU(), ["p_ab2"], "p_ab2r"), (ab_down, ["p_ab2r"], "p_above"),
            (Concat(), ["p_ismax", "p_above"], "p_ia"), (hand, ["p_ia"], "p_h0"), (ReLU(), ["p_h0"], "p_hsoft"),
            (hb_up, ["p_hsoft"], "p_hb2"), (ReLU(), ["p_hb2"], "p_hb2r"), (hb_down, ["p_hb2r"], "hump"),
            (dia, ["hump"], "n_dia"), (prw, ["hump"], "n_pr"),
            (one_uv, ["n_dia"], "one_uv"), (ReLU(), ["one_uv"], "one_uvr"), (one_rd, ["one_uvr"], "one0"), (ReLU(), ["one0"], "one"),
            (ip_up, ["n_pr"], "ip2"), (ReLU(), ["ip2"], "ip2r"), (ip_down, ["ip2r"], "in_pr"),
            (prev, ["beat"], "prev0"), (pv_up, ["prev0"], "pv2"), (ReLU(), ["pv2"], "pv2r"), (pv_down, ["pv2r"], "prevclose"),
            (Concat(), ["beat", "prevclose"], "bpc"), (assess, ["bpc"], "as0"), (ReLU(), ["as0"], "assess"),
            (Concat(), ["assess", "one", "in_pr"], "aoi"), (pand, ["aoi"], "pd0"), (ReLU(), ["pd0"], "pdet"),
            (GlobalSum(), ["assess"], "n_assess"), (GlobalSum(), ["pdet"], "n_p"),
            (uv, ["n_assess"], "pa_uv"), (ReLU(), ["pa_uv"], "pa_uvr"), (oh, ["pa_uvr"], "pa_oh0"), (ReLU(), ["pa_oh0"], "pa_onehot"),
            (Concat(), ["n_p", "pa_onehot"], "np_oh"), (gate, ["np_oh"], "pz0"), (ReLU(), ["pz0"], "pz"), (read, ["pz"], "p_frac"),
            (pp_up, ["p_frac"], "pp2"), (ReLU(), ["pp2"], "pp2r"), (pp_down, ["pp2r"], "p_present")]

    net = Net(ops, inputs=("x",), output="p_present")
    net.design = D
    net.design2 = s2.design
    net.detector_design = s2.detector_design
    net.periods = s2.periods
    net.knots = s2.knots
    net.outputs = s2.outputs + ["n_assess", "n_p", "p_frac", "p_present"]
    net.p_kernel = k
    return net
