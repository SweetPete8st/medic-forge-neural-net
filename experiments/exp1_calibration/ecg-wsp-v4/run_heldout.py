"""Held-out evaluation of the frozen network on MIT-BIH (mitdb) and the MIT-BIH
AF database (afdb). Run on a machine where download_mitdb.py has been run.

    python src/run_heldout.py                      # everything
    python src/run_heldout.py --records 203 208     # a few mitdb records
    python src/run_heldout.py --afdb-stride 20      # sample every 20th 10 s window of afdb (default)

Outputs (in results/):
    heldout_beats.csv      per record: beats, TP, FP, FN, sensitivity, PPV, mean offset
    heldout_windows.csv    per 10 s window: annotated rhythm, network label, every head output
    heldout_widths.csv     per annotated beat: beat type, steep-slope width (ms)
    heldout_summary.txt    the numbers that matter, printed at the end as well

The network is NOT modified by this script. Every record run is logged to
LEDGER.md as one forward pass.
"""
import argparse
import csv
import glob
import os
import sys
import traceback
from collections import Counter, defaultdict

import numpy as np
import wfdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_beats import load, reference_beats, match, log_forward_pass  # noqa: E402
from eval_stage3 import rhythm_per_window  # noqa: E402
from wsp.labels import build_stage4, read_labels  # noqa: E402
from wsp.heads import read_outputs  # noqa: E402

FS = 360
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results")

# What the MIT-BIH rhythm annotations mean, and which network labels would count as agreement.
RHYTHM_OK = {
    "(N": {"sinus rhythm", "sinus bradycardia", "narrow-complex tachycardia 100-150", "regular narrow 60-100 (P n/a)",
           "irregular with P"},
    "(SBR": {"sinus bradycardia", "junctional bradycardia"},
    "(AFIB": {"atrial fibrillation"},
    "(AFL": {"atrial fibrillation", "SVT", "narrow-complex tachycardia 100-150", "irregular with P"},
    "(SVTA": {"SVT", "narrow-complex tachycardia 100-150"},
    "(VT": {"VT"},
    "(IVR": {"idioventricular", "accelerated idioventricular"},
    "(NOD": {"junctional", "junctional bradycardia", "regular narrow 60-100 (P n/a)"},
    "(J": {"junctional", "junctional bradycardia", "regular narrow 60-100 (P n/a)"},
    "(B": {"irregular with P", "atrial fibrillation"},     # ventricular bigeminy: irregular, ectopy
    "(T": {"irregular with P", "atrial fibrillation"},     # trigeminy
    "(AB": {"irregular with P", "atrial fibrillation"},    # atrial bigeminy
}


def run_record(net, rec_path, ann_ext, afdb_stride, window_csv, width_csv, beat_rows, label_counts, rhythm_set):
    name = os.path.basename(rec_path)
    x, fs, lead = load(rec_path, 0)
    W = int(net.design2["window_sec"] * FS)
    n_win = len(x) // W
    stride = afdb_stride if ann_ext == "qrs" else 1
    # --- beats (full record, fully convolutional)
    full = net.forward(x[None, :], return_all=True)
    det = np.flatnonzero(full["beat"][0] > 0.5)
    try:
        ref, syms = reference_beats(rec_path, fs, ann_ext=ann_ext)
    except Exception:
        ref, syms = np.array([], int), []
    if len(ref):
        tp, fp, fn, off = match(det, ref)
        beat_rows.append(dict(record=name, lead=lead, reference=ann_ext, beats=len(ref), detected=len(det), TP=tp, FP=fp, FN=fn,
                              sensitivity=tp / max(tp + fn, 1), PPV=tp / max(tp + fp, 1),
                              offset_ms=float(off.mean() / FS * 1000) if len(off) else float("nan")))
        # per-beat width by type
        wab = full["width_at_beat"][0]
        for r_s, s in zip(ref, syms):
            j = np.searchsorted(det, r_s)
            c = det[max(0, j - 1): j + 1]
            if len(c) and np.min(np.abs(c - r_s)) <= 0.15 * FS:
                width_csv.writerow([name, s, f"{wab[c[np.argmin(np.abs(c - r_s))]]:.1f}"])
    # --- windows
    rhythms = rhythm_per_window(rec_path, fs, n_win, W)
    for i in range(0, n_win, stride):
        t = net.forward(x[None, i * W:(i + 1) * W], return_all=True)
        o = read_outputs(t, net)
        idx, lab, ect, _ = read_labels(t)
        rhythm = rhythms[i]
        row = dict(record=name, window=i, rhythm=rhythm, label=lab, ectopy=int(ect))
        row.update({k: f"{v:.4f}" for k, v in o.items() if k != "labels"})
        window_csv.writerow(row)
        label_counts[(rhythm, lab)] += 1
        rhythm_set[rhythm] += 1
    return n_win // stride


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="*", default=None, help="mitdb record names to run (default: all but 100)")
    ap.add_argument("--no-afdb", action="store_true")
    ap.add_argument("--afdb-stride", type=int, default=20)
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    net = build_stage4(fs=FS)
    mit = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(args.data, "mitdb", "*.hea")))
    mit = [r for r in mit if r != "100" and "-" not in r]
    if args.records:
        mit = args.records
    af = [] if args.no_afdb else sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(args.data, "afdb", "*.hea")))
    af = [r for r in af if os.path.exists(os.path.join(args.data, "afdb", r + ".dat"))]

    beat_rows = []
    label_counts = Counter()
    rhythm_set = Counter()
    with open(os.path.join(OUT, "heldout_windows.csv"), "w", newline="") as fw, \
            open(os.path.join(OUT, "heldout_widths.csv"), "w", newline="") as fwd:
        keys = ["record", "window", "rhythm", "label", "ectopy"] + [k for k in net.outputs if k != "labels"]
        window_csv = csv.DictWriter(fw, fieldnames=keys)
        window_csv.writeheader()
        width_csv = csv.writer(fwd)
        width_csv.writerow(["record", "beat_type", "width_ms"])
        jobs = [(os.path.join(args.data, "mitdb", r), "atr") for r in mit] + [(os.path.join(args.data, "afdb", r), "qrs") for r in af]
        for k, (rec_path, ext) in enumerate(jobs):
            try:
                n = run_record(net, rec_path, ext, args.afdb_stride, window_csv, width_csv, beat_rows, label_counts, rhythm_set)
                print(f"[{k+1}/{len(jobs)}] {os.path.basename(rec_path)}: {n} windows", flush=True)
                fw.flush(); fwd.flush()
                log_forward_pass("v4 (frozen)", f"{os.path.basename(rec_path)} held-out", "held-out evaluation on David's machine")
            except Exception:
                print(f"[{k+1}/{len(jobs)}] {os.path.basename(rec_path)}: FAILED")
                traceback.print_exc()

    with open(os.path.join(OUT, "heldout_beats.csv"), "w", newline="") as fb:
        w = csv.DictWriter(fb, fieldnames=list(beat_rows[0].keys()) if beat_rows else ["record"])
        w.writeheader()
        for r in beat_rows:
            w.writerow(r)

    lines = []
    mit_rows = [r for r in beat_rows if r["reference"] == "atr"]
    if mit_rows:
        tp = sum(r["TP"] for r in mit_rows); fp = sum(r["FP"] for r in mit_rows); fn = sum(r["FN"] for r in mit_rows)
        lines.append(f"MIT-BIH beat detection over {len(mit_rows)} held-out records: Se {100*tp/(tp+fn):.2f}%  PPV {100*tp/(tp+fp):.2f}%  "
                     f"(TP {tp} FP {fp} FN {fn})")
        worst = sorted(mit_rows, key=lambda r: min(r["sensitivity"], r["PPV"]))[:8]
        lines.append("  worst records: " + ", ".join(f"{r['record']} Se {100*r['sensitivity']:.1f}/PPV {100*r['PPV']:.1f}" for r in worst))
    # widths
    by_type = defaultdict(list)
    with open(os.path.join(OUT, "heldout_widths.csv")) as f:
        rd = csv.reader(f); next(rd)
        for rec, s, w in rd:
            by_type[s].append(float(w))
    lines.append("steep-slope width (ms) by beat type: " + "; ".join(
        f"{s}: n={len(v)} median {np.median(v):.0f} p10 {np.percentile(v,10):.0f} p90 {np.percentile(v,90):.0f} >=85ms {100*np.mean(np.array(v)>=85):.0f}%"
        for s, v in sorted(by_type.items(), key=lambda kv: -len(kv[1])) if len(v) >= 20))
    # confusion
    rhythms = [r for r, _ in rhythm_set.most_common()]
    labels = sorted({l for _, l in label_counts})
    lines.append("\nwindows by annotated rhythm (rows) and network label (columns):")
    lines.append("rhythm".ljust(8) + "".join(l[:14].rjust(16) for l in labels) + "   agree")
    for r in rhythms:
        row = [label_counts[(r, l)] for l in labels]
        ok = RHYTHM_OK.get(r)
        agree = sum(label_counts[(r, l)] for l in labels if ok and l in ok) / max(sum(row), 1)
        lines.append(r.ljust(8) + "".join(str(c).rjust(16) for c in row) + (f"   {100*agree:.0f}%" if ok else "   n/a"))
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(OUT, "heldout_summary.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
