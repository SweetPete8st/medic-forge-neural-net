"""Score the beat detector against PhysioNet reference annotations.

Usage:
    python src/eval_beats.py <record_path> [--channel 0] [--purpose "..."]

A matched beat is a detection within +/-150 ms of a reference beat annotation
(ANSI/AAMI EC57 window). Every run appends one forward-pass line to LEDGER.md.
"""
import argparse
import datetime
import os
import sys

import numpy as np
import wfdb
from scipy.signal import resample_poly
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp.detector import build_detector, beat_samples  # noqa: E402

BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(HERE, "LEDGER.md")


def load(record_path, channel=0, fs_target=360):
    rec = wfdb.rdrecord(record_path)
    x = rec.p_signal[:, channel].astype(np.float64)
    fs = rec.fs
    if fs != fs_target:
        fr = Fraction(fs_target, int(round(fs))).limit_denominator(1000)
        x = resample_poly(x, fr.numerator, fr.denominator)
    x = np.nan_to_num(x)
    return x, fs, rec.sig_name[channel]


def reference_beats(record_path, fs, fs_target=360, ann_ext="atr"):
    ann = wfdb.rdann(record_path, ann_ext)
    keep = [i for i, s in enumerate(ann.symbol) if s in BEAT_SYMBOLS]
    samples = ann.sample[keep].astype(np.float64) * (fs_target / fs)
    symbols = [ann.symbol[i] for i in keep]
    return np.round(samples).astype(int), symbols


def match(det, ref, fs=360, window_sec=0.150):
    """Greedy one-to-one matching within the window. Returns TP, FP, FN, offsets."""
    w = window_sec * fs
    det = np.sort(det)
    ref = np.sort(ref)
    used = np.zeros(len(det), bool)
    offsets = []
    tp = 0
    j = 0
    for r in ref:
        # advance j to first det candidate near r
        while j < len(det) and det[j] < r - w:
            j += 1
        best, bestd = -1, w + 1
        k = j
        while k < len(det) and det[k] <= r + w:
            if not used[k] and abs(det[k] - r) < bestd:
                best, bestd = k, abs(det[k] - r)
            k += 1
        if best >= 0:
            used[best] = True
            tp += 1
            offsets.append(det[best] - r)
    fp = int((~used).sum())
    fn = len(ref) - tp
    return tp, fp, fn, np.array(offsets)


def log_forward_pass(version, data, purpose):
    with open(LEDGER, "r") as f:
        text = f.read()
    n = text.count("\n| ", text.find("## Forward passes"), text.find("## Design changes")) - 1
    line = f"| {n + 1} | {datetime.date.today()} | {version} | {data} | {purpose} |\n"
    head, tail = text.split("## Design changes")
    head = head.rstrip("\n") + "\n" + line + "\n"
    with open(LEDGER, "w") as f:
        f.write(head + "## Design changes" + tail)
    return n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--ann", default="atr")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--purpose", default="evaluation")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--save", default=None, help="save intermediate tensors to .npz")
    args = ap.parse_args()

    x, fs, lead = load(args.record, args.channel)
    net = build_detector()
    out = net.forward(x[None, :], return_all=True)
    det = beat_samples(out["beat"])
    ref, syms = reference_beats(args.record, fs, ann_ext=args.ann)
    tp, fp, fn, off = match(det, ref)
    se = tp / max(tp + fn, 1)
    ppv = tp / max(tp + fp, 1)
    name = os.path.basename(args.record)
    print(f"record {name} lead {lead} fs {fs} -> 360; {len(x)/360:.0f} s")
    print(f"reference beats {len(ref)}  detected {len(det)}")
    print(f"TP {tp}  FP {fp}  FN {fn}")
    print(f"sensitivity {100*se:.2f}%   PPV {100*ppv:.2f}%")
    if len(off):
        print(f"offset det-ref: mean {off.mean()/360*1000:+.1f} ms, sd {off.std()/360*1000:.1f} ms")
    # where are the errors (by annotation type near FN)?
    if fn:
        det_sorted = np.sort(det)
        missed = []
        for r, s in zip(ref, syms):
            if len(det_sorted) == 0 or np.min(np.abs(det_sorted - r)) > 0.15 * 360:
                missed.append((r, s))
        from collections import Counter
        print("missed by type:", dict(Counter(s for _, s in missed)))
        print("first missed (sample, sec):", [(r, round(r / 360, 1)) for r, _ in missed[:10]])
    if fp:
        ref_sorted = np.sort(ref)
        extra = [d for d in det if np.min(np.abs(ref_sorted - d)) > 0.15 * 360]
        print("first false positives (sample, sec):", [(d, round(d / 360, 1)) for d in extra[:10]])
    if args.save:
        np.savez_compressed(args.save, raw=x, **{k: v for k, v in out.items()}, det=det, ref=ref)
    if not args.no_log:
        n = log_forward_pass(args.version, f"{name} ch{args.channel}", args.purpose)
        print(f"logged forward pass #{n}")


if __name__ == "__main__":
    main()
