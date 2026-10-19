"""BC refractory-period deletion rule: formalize the LB-verified v2 signal.

LB-verified insight: cells whose 30s context already shows a high BC rate
(listener "just backchanneled") are FP-dense for BC prediction (75%+),
especially when model consensus is weak. This tool calibrates the rule
(rate_thr, cons_thr) on the seed42 valid OOF and applies it to a test
submission, replacing per-cell LB probing with a reproducible rule.

Usage:
  python tools_bc_refractory_rule.py calibrate \
    --oof outputs/oof/temporal_tail25_grid25.npz outputs/oof/stride5_e16_grid25.npz ... \
    --out outputs/bc_refractory_rule.json
  python tools_bc_refractory_rule.py apply \
    --rule outputs/bc_refractory_rule.json \
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
BC_COL = 3          # column order c,na,i,bc,t
BC_LABEL_ID = 2     # chunk label id for BC
CTX = 375


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


def ctx_bc_rate_valid(conv, end):
    cache = {}
    rates = np.zeros(len(conv))
    for i, (c, e) in enumerate(zip(conv, end)):
        if c not in cache:
            cache[c] = np.load(ROOT / f"data/train/labels/{c}.npy")
        e = int(e)
        seg = cache[c][max(0, e - CTX):e]
        rates[i] = (seg == BC_LABEL_ID).mean() if len(seg) else 0.0
    return rates


def cmd_calibrate(args):
    P, L, conv, end = load_oof(args.oof)
    y, s = L[:, BC_COL], P[:, BC_COL]
    rate = ctx_bc_rate_valid(conv, end)

    best_thr, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y, s >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, t
    pred = s >= best_thr

    grid_rate = np.arange(0.02, 0.30, 0.01)
    grid_cons = np.quantile(s[pred], np.linspace(0.1, 1.0, 19))
    base_f1 = best_f1
    best = {"rate_thr": None, "cons_thr": None, "f1": base_f1, "n_del": 0}
    for rt in grid_rate:
        for ct in grid_cons:
            kill = pred & (rate >= rt) & (s <= ct)
            if kill.sum() == 0:
                continue
            f = f1_score(y, pred & ~kill, zero_division=0)
            if f > best["f1"]:
                fp = int((kill & (y == 0)).sum())
                best = {"rate_thr": float(rt), "cons_thr": float(ct), "f1": float(f),
                        "n_del": int(kill.sum()), "del_fp": fp,
                        "del_precision": fp / int(kill.sum())}
    out = {
        "bc_thr": float(best_thr), "base_f1": float(base_f1), "rule": best,
        "oof_sources": [str(p) for p in args.oof],
        "note": "delete BC pred when ctx_bc_rate>=rate_thr and consensus<=cons_thr",
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def cmd_apply(args):
    rule = json.loads(Path(args.rule).read_text())
    rt, ct = rule["rule"]["rate_thr"], rule["rule"]["cons_thr"]
    ids = [l.strip() for l in open(args.ids)]
    P = np.load(args.probs)
    if P.ndim == 3:
        P = P.mean(0)
    s = {sid.zfill(4): float(p) for sid, p in zip(ids, P[:, BC_COL])}

    ctx_dir = Path(args.test_context_dir) if args.test_context_dir else (ROOT / "data/test/context")
    rows = list(csv.DictReader(open(args.base_csv)))
    report = []
    hits = set()
    for r in rows:
        if r["bc"] != "1":
            continue
        sid = r["segment_id"].zfill(4)
        ctx = np.load(ctx_dir / f"{sid}.npy")
        rate = float((ctx == BC_LABEL_ID).mean())
        cons = s.get(sid, float("nan"))
        hit = rate >= rt and cons <= ct
        if hit:
            hits.add(sid)
        report.append({"sid": sid, "ctx_bc_rate": round(rate, 4),
                       "consensus": round(cons, 4), "rule_hit": hit})
    report.sort(key=lambda x: (-x["rule_hit"], -x["ctx_bc_rate"]))
    n_hit = sum(r["rule_hit"] for r in report)
    print(f"rule (rate>={rt:.2f}, cons<={ct:.4f}) on {len(report)} BC cells -> delete {n_hit}")
    for r in report:
        print(f"  {'DEL ' if r['rule_hit'] else 'keep'} {r['sid']} rate={r['ctx_bc_rate']:.3f} cons={r['consensus']:.3f}")

    if args.out:
        cols = ["c", "na", "i", "bc", "t"]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["segment_id"] + cols)
            for row in rows:
                sid = row["segment_id"].zfill(4)
                if sid in hits:
                    row["bc"] = "0"
                w.writerow([row["segment_id"]] + [row[c] for c in cols])
        print(f"wrote {args.out} (BC deleted: {len(hits)})")


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
                   help="output csv with BC=1 cells deleted per rule (optional)")
    a.add_argument("--test_context_dir", default=None,
                   help="override test context dir (default: data/test/context under ROOT)")
    a.set_defaults(func=cmd_apply)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
