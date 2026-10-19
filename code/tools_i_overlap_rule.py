"""I-column acoustic-overlap deletion rule: formalize the LB-verified signal.

LB-verified insight (I 63->59, +0.0029): I predictions whose context tail
shows no acoustic overlap (callhome-zho powerset P(>=2 spk)) are FP-dense;
genuine interruptions already show overlap build-up in the last seconds
(case 0199: last-2s overlap 0.98 was a rescued TP).

calibrate: tune (feature, ov_thr, cons_thr) on seed42 valid OOF windows.
apply:     report which i=1 cells of a submission the rule would delete.

Usage:
  python tools_i_overlap_rule.py calibrate \
    --oof outputs/oof/temporal_tail25_grid25.npz ... --out outputs/i_overlap_rule.json
  python tools_i_overlap_rule.py apply \
    --rule outputs/i_overlap_rule.json \
    --probs outputs/test_probs/ensemble_sota_s5.npy --ids outputs/test_probs/base_ids.txt \
    --base_csv ../../submissions/20260612_BC27_v2signal_del0122.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path("/workspace/xinye")
PB = ROOT / "code_src/pb"
OV_TRAIN = PB / "outputs/acoustic_overlap_train"
I_COL = 2           # column order c,na,i,bc,t
CTX = 375
CHUNK_S = 0.08
FEAT_NAMES = ["ov_mean", "ov_last5s", "ov_last2s_mean", "ov_last2s_max"]


def load_oof(paths):
    P, L, conv, end = None, None, None, None
    for p in paths:
        z = np.load(p, allow_pickle=True)
        if P is None:
            P = z["probs"].astype(np.float64)
            L, conv, end = z["labels"], z["conv_ids"].astype(str), z["end_idxs"]
        else:
            assert (z["conv_ids"].astype(str) == conv).all() and (z["end_idxs"] == end).all(), p
            P += z["probs"]
    return P / len(paths), L, conv, end


def window_feats(conv, end):
    """Aggregate frame-level overlap into the 4 test-aligned features per window."""
    cache = {}
    F = np.zeros((len(conv), 4), np.float32)
    for i, (c, e) in enumerate(zip(conv, end)):
        if c not in cache:
            ov = np.load(OV_TRAIN / f"{c}.npy")
            dur_s = len(np.load(ROOT / f"data/train/labels/{c}.npy")) * CHUNK_S
            cache[c] = (ov, len(ov) / max(dur_s, 1e-6))
        ov, fps = cache[c]
        e = int(e)
        a = int(max(0, (e - CTX) * CHUNK_S * fps))
        b = int(min(len(ov), e * CHUNK_S * fps))
        seg = ov[a:b]
        if len(seg) == 0:
            continue
        n5, n2 = max(1, int(5 * fps)), max(1, int(2 * fps))
        F[i] = [seg.mean(), seg[-n5:].mean(), seg[-n2:].mean(), seg[-n2:].max()]
    return F


def cmd_calibrate(args):
    P, L, conv, end = load_oof(args.oof)
    y, s = L[:, I_COL], P[:, I_COL]
    F = window_feats(conv, end)

    best_thr, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y, s >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, t
    pred = s >= best_thr

    best = {"feat": None, "ov_thr": None, "cons_thr": None, "f1": best_f1, "n_del": 0}
    for fi, fname in enumerate(FEAT_NAMES):
        x = F[:, fi]
        for ot in np.quantile(x[pred], np.linspace(0.02, 0.5, 13)):
            for ct in np.quantile(s[pred], np.linspace(0.1, 1.0, 10)):
                kill = pred & (x <= ot) & (s <= ct)
                if kill.sum() == 0:
                    continue
                f = f1_score(y, pred & ~kill, zero_division=0)
                if f > best["f1"]:
                    fp = int((kill & (y == 0)).sum())
                    best = {"feat": fname, "ov_thr": float(ot), "cons_thr": float(ct),
                            "f1": float(f), "n_del": int(kill.sum()), "del_fp": fp,
                            "del_precision": fp / int(kill.sum())}
    out = {"i_thr": float(best_thr), "base_f1": float(best_f1), "rule": best,
           "oof_sources": [str(p) for p in args.oof],
           "note": "delete I pred when ctx-tail overlap<=ov_thr and consensus<=cons_thr"}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def cmd_apply(args):
    rule = json.loads(Path(args.rule).read_text())
    r = rule["rule"]
    fi = FEAT_NAMES.index(r["feat"])
    ids = [l.strip() for l in open(args.ids)]
    P = np.load(args.probs)
    if P.ndim == 3:
        P = P.mean(0)
    cons = {sid.zfill(4): float(p) for sid, p in zip(ids, P[:, I_COL])}

    ov_probs = Path(args.acoustic_overlap_probs) if args.acoustic_overlap_probs \
        else PB / "outputs/test_probs/acoustic_overlap_test.npy"
    ov_ids = Path(args.acoustic_overlap_ids) if args.acoustic_overlap_ids \
        else PB / "outputs/test_probs/acoustic_overlap_test_ids.txt"
    tf = np.load(ov_probs)
    tids = [l.strip().zfill(4) for l in open(ov_ids)]
    feat = {sid: float(x) for sid, x in zip(tids, tf[:, fi])}

    rows = list(csv.DictReader(open(args.base_csv)))
    report = []
    hits = set()
    for row in rows:
        if row["i"] != "1":
            continue
        sid = row["segment_id"].zfill(4)
        f, c = feat.get(sid, float("nan")), cons.get(sid, float("nan"))
        hit = f <= r["ov_thr"] and c <= r["cons_thr"]
        if hit:
            hits.add(sid)
        report.append({"sid": sid, r["feat"]: round(f, 4), "consensus": round(c, 4), "rule_hit": hit})
    report.sort(key=lambda x: (-x["rule_hit"], x[r["feat"]]))
    n_hit = sum(x["rule_hit"] for x in report)
    print(f"rule ({r['feat']}<={r['ov_thr']:.4f}, cons<={r['cons_thr']:.4f}) "
          f"on {len(report)} I cells -> delete {n_hit}")
    for x in report:
        print(f"  {'DEL ' if x['rule_hit'] else 'keep'} {x['sid']} "
              f"{r['feat']}={x[r['feat']]:.4f} cons={x['consensus']:.3f}")

    if args.out:
        cols = ["c", "na", "i", "bc", "t"]
        with open(args.out, "w", newline="", encoding="utf-8") as f_out:
            w = csv.writer(f_out, lineterminator="\n")
            w.writerow(["segment_id"] + cols)
            for row in rows:
                sid = row["segment_id"].zfill(4)
                if sid in hits:
                    row["i"] = "0"
                w.writerow([row["segment_id"]] + [row[c] for c in cols])
        print(f"wrote {args.out} (I deleted: {len(hits)})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--oof", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_calibrate)
    a = sub.add_parser("apply")
    a.add_argument("--rule", required=True)
    a.add_argument("--probs", required=True)
    a.add_argument("--ids", required=True)
    a.add_argument("--base_csv", required=True)
    a.add_argument("--out", default=None,
                   help="output csv with I=1 cells deleted per rule (optional)")
    a.add_argument("--acoustic_overlap_probs", default=None,
                   help="override acoustic overlap test probs npy path")
    a.add_argument("--acoustic_overlap_ids", default=None,
                   help="override acoustic overlap test ids txt path")
    a.set_defaults(func=cmd_apply)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
