"""Stage 4: rhythm label head, written in weight space.

The rhythm algorithm every paramedic student learns (rate, regular?, wide?,
P waves?) compiled into two ReLU layers over the Stage 2 and 3 outputs.

Indicators (soft thresholds; a 1 bpm ramp at each rate boundary):
    hasbeats  N >= 2
    fast      rate > 100          veryfast  rate > 150
    slow      rate < 60           vslow     rate < 50
    regI      regular flag        wideI     wide flag        anyw   any_wide flag
    pp        p_present flag      Pa        n_assess >= 3 (P was assessable)

Derived:  Pyes = pp AND Pa,  Pno = (NOT pp) AND Pa,  Pna = NOT Pa,  ectopy = anyw AND NOT wide

Labels (each one AND of literals; NOT x is written 1 - x, so every label is
a single ReLU unit over the indicator vector):
     0 no QRS detected                 NOT hasbeats
     1 ventricular tachycardia          wide, fast
     2 wide-QRS rhythm with P (BBB)     wide, NOT fast, Pyes
     3 idioventricular                  wide, NOT fast, vslow, NOT Pyes
     4 accelerated idioventricular      wide, NOT fast, NOT vslow, NOT Pyes
     5 atrial fibrillation              narrow, irregular, NOT Pyes
     6 irregular with P waves           narrow, irregular, Pyes        (sinus arrhythmia / sinus with ectopy)
     7 SVT                              narrow, regular, veryfast
     8 narrow-complex tachycardia       narrow, regular, fast, NOT veryfast  (sinus tach vs SVT; P not assessable at this rate)
     9 sinus rhythm                     narrow, regular, 60-100, Pyes
    10 junctional / accel. junctional   narrow, regular, 60-100, Pno
    11 regular narrow 60-100, P n/a     narrow, regular, 60-100, Pna
    12 sinus bradycardia                narrow, regular, slow, Pyes
    13 junctional bradycardia / escape  narrow, regular, slow, NOT Pyes
    14 modifier: ectopy present         anyw AND NOT wide

Not attempted: AV blocks (need P:QRS ratio), atrial flutter (needs an F-wave
comb), paced rhythm, ventricular fibrillation.
"""
import numpy as np

from .pwave import build_stage3
from .layers import Conv1d, ReLU, Concat, Net

LABELS = ["no QRS detected", "VT", "wide QRS with P (BBB)", "idioventricular", "accelerated idioventricular",
          "atrial fibrillation", "irregular with P", "SVT", "narrow-complex tachycardia 100-150",
          "sinus rhythm", "junctional", "regular narrow 60-100 (P n/a)", "sinus bradycardia",
          "junctional bradycardia", "ectopy present (modifier)"]


def build_stage4(fs=360, **kw):
    s3 = build_stage3(fs=fs, **kw)
    ops = list(s3.ops)
    feats = ["N", "rate_rr", "regular", "wide", "any_wide", "p_present", "n_assess"]
    F = len(feats)
    fi = {f: i for i, f in enumerate(feats)}

    # ---- indicator block: clip(K (sign * f - sign * theta), 0, 1) for each indicator
    # (feature, sign, theta, K)
    spec = [("N", +1, 1.0, 1.0),          # hasbeats: N>=2
            ("rate_rr", +1, 100.0, 1.0),  # fast
            ("rate_rr", +1, 150.0, 1.0),  # veryfast
            ("rate_rr", -1, 60.0, 1.0),   # slow
            ("rate_rr", -1, 50.0, 1.0),   # vslow
            ("regular", +1, 0.0, 1.0),    # regI (already 0/1)
            ("wide", +1, 0.0, 1.0),       # wideI
            ("any_wide", +1, 0.0, 1.0),   # anyw
            ("p_present", +1, 0.0, 1.0),  # pp
            ("n_assess", +1, 2.0, 1.0)]   # Pa: n_assess >= 3
    I = len(spec)
    wA = np.zeros((2 * I, F, 1)); bA = np.zeros(2 * I)
    for i, (f, s, th, K) in enumerate(spec):
        wA[2 * i, fi[f], 0] = K * s;     bA[2 * i] = -K * s * th
        wA[2 * i + 1, fi[f], 0] = K * s; bA[2 * i + 1] = -K * s * th - 1.0
    indA = Conv1d(wA, bA, name="ind.up")
    wB = np.zeros((I, 2 * I, 1))
    for i in range(I):
        wB[i, 2 * i, 0] = 1.0; wB[i, 2 * i + 1, 0] = -1.0
    indB = Conv1d(wB, name="ind.down")
    names = ["hasbeats", "fast", "veryfast", "slow", "vslow", "regI", "wideI", "anyw", "pp", "Pa"]
    ni = {n: i for i, n in enumerate(names)}

    # ---- derived block: pass-through of indicators plus Pyes, Pno, Pna, ectopy
    der = names + ["Pyes", "Pno", "Pna", "ectopy"]
    wD = np.zeros((len(der), I, 1)); bD = np.zeros(len(der))
    for i in range(I):
        wD[i, i, 0] = 1.0                                   # identity (ReLU of a [0,1] value)
    wD[ni_len := I, ni["pp"], 0] = 1.0; wD[I, ni["Pa"], 0] = 1.0; bD[I] = -1.0           # Pyes
    wD[I + 1, ni["pp"], 0] = -1.0; wD[I + 1, ni["Pa"], 0] = 1.0                           # Pno = ReLU(Pa - pp)
    wD[I + 2, ni["Pa"], 0] = -1.0; bD[I + 2] = 1.0                                        # Pna = 1 - Pa
    wD[I + 3, ni["anyw"], 0] = 1.0; wD[I + 3, ni["wideI"], 0] = -1.0                      # ectopy = ReLU(anyw - wide)
    derive = Conv1d(wD, bD, name="derive")
    di = {n: i for i, n in enumerate(der)}

    # ---- label block: one ReLU unit per label; literal lists with '~' for NOT
    rules = [["~hasbeats"],
             ["hasbeats", "wideI", "fast"],
             ["hasbeats", "wideI", "~fast", "Pyes"],
             ["hasbeats", "wideI", "~fast", "vslow", "~Pyes"],
             ["hasbeats", "wideI", "~fast", "~vslow", "~Pyes"],
             ["hasbeats", "~wideI", "~regI", "~Pyes"],
             ["hasbeats", "~wideI", "~regI", "Pyes"],
             ["hasbeats", "~wideI", "regI", "veryfast"],
             ["hasbeats", "~wideI", "regI", "fast", "~veryfast"],
             ["hasbeats", "~wideI", "regI", "~fast", "~slow", "Pyes"],
             ["hasbeats", "~wideI", "regI", "~fast", "~slow", "Pno"],
             ["hasbeats", "~wideI", "regI", "~fast", "~slow", "Pna"],
             ["hasbeats", "~wideI", "regI", "slow", "Pyes"],
             ["hasbeats", "~wideI", "regI", "slow", "~Pyes"],
             ["ectopy"]]
    assert len(rules) == len(LABELS)
    wL = np.zeros((len(rules), len(der), 1)); bL = np.zeros(len(rules))
    for r, lits in enumerate(rules):
        n_neg = 0
        for lit in lits:
            if lit.startswith("~"):
                wL[r, di[lit[1:]], 0] = -1.0; n_neg += 1
            else:
                wL[r, di[lit], 0] = 1.0
        bL[r] = n_neg - len(lits) + 1.0
    label = Conv1d(wL, bL, name="labels")

    ops += [(Concat(), feats, "feat"),
            (indA, ["feat"], "indA"), (ReLU(), ["indA"], "indAr"), (indB, ["indAr"], "ind"),
            (derive, ["ind"], "der0"), (ReLU(), ["der0"], "der"),
            (label, ["der"], "lab0"), (ReLU(), ["lab0"], "labels")]
    net = Net(ops, inputs=("x",), output="labels")
    for attr in ("design", "design2", "detector_design", "periods", "knots", "p_kernel"):
        setattr(net, attr, getattr(s3, attr))
    net.outputs = s3.outputs + ["labels"]
    net.label_names = LABELS
    return net


def read_labels(t):
    v = t["labels"][:, 0]
    top = int(np.argmax(v[:14]))
    return top, LABELS[top], bool(v[14] > 0.5), v
