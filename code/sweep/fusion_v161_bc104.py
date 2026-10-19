"""融合 N 个成员 test probs，按 OOF 网格搜索的最优 per-class 线性权重，
ref-rate top-k 二值化产出 submission csv。

输入:
  --test_probs A.npy B.npy ... : 每个 [n_segments, 5] (cols: c,na,i,bc,t)
  --test_ids   A_ids.txt B_ids.txt ... : 同序行号 = npy 第 0 维 idx
  --oof A.npz B.npz ...        : 同源 OOF（probs/labels/conv_ids/end_idxs）
                                   提供则启用 per-class 权重网格搜索
  --weights "w1,w2,..."        : 不给 OOF 时可手动指定单纯形权重（C/NA/I/BC/T 共用）

输出:
  --out_probs : 融合 test probs [n_segments, 5]
  --out_csv   : ref-rate top-k 0/1 二值化 csv
  --out_ids   : 融合 test 的 segment_id（zfill 4），与 csv 第一列对齐
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

N = ["c", "na", "i", "bc", "t"]
REF = {"c": 0.987, "na": 0.780, "i": 0.150, "bc": 0.104, "t": 0.450}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_probs", nargs="+", required=True)
    p.add_argument("--test_ids", nargs="+", required=True,
                   help="同序 segment ids；所有成员 ids 必须一致")
    p.add_argument("--oof", nargs="+", default=None,
                   help="可选 OOF npz 列表，启用 per-class 权重网格搜索")
    p.add_argument("--weights", default=None,
                   help='手动指定 K 个等比例权重 "w1,w2,...,wK"（5 列共用；与 --oof 互斥）')
    p.add_argument("--weights_json", default=None,
                   help="JSON 文件 {'c':[w1..wK],'na':[..],'i':[..],'bc':[..],'t':[..]} 含 per-class 权重；"
                        "与 --oof/--weights 互斥；用于复赛固化 OOF 搜索结果跳过搜索")
    p.add_argument("--grid_step", type=float, default=0.1,
                   help="OOF 网格搜索的单纯形步长（默认 0.1，模型数<=5 时性能可接受）")
    p.add_argument("--objective", default="auc", choices=["auc", "ref_rate_f1"],
                   help="OOF 权重搜索目标：auc=ROC AUC（默认，稳）；"
                        "ref_rate_f1=直接优化 ref-rate top-k macroF1（与 LB 一致，单列收益高）")
    p.add_argument("--ref_override", default=None,
                   help='JSON 字符串覆盖 REF rate, e.g. {"t":0.48,"bc":0.04} (只覆盖给出的列)')
    p.add_argument("--out_probs", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--out_ids", required=True)
    return p.parse_args()


def load_oof_aligned(paths):
    data = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        keys = list(zip([str(x) for x in d["conv_ids"]], [int(x) for x in d["end_idxs"]]))
        kmap = {k: i for i, k in enumerate(keys)}
        data.append((d["probs"].astype(np.float64), d["labels"], kmap))
    common = sorted(set.intersection(*[set(km.keys()) for _, _, km in data]))
    if not common:
        raise SystemExit("ERROR: OOF 文件无公共 (conv_id,end_idx) 样本")
    probs_list, labels_ref = [], None
    for probs, labels, kmap in data:
        idx = [kmap[k] for k in common]
        probs_list.append(probs[idx])
        if labels_ref is None:
            labels_ref = labels[idx]
        else:
            assert np.array_equal(labels_ref, labels[idx]), "OOF 标签对齐失败"
    return probs_list, labels_ref


def search_per_class_weights(probs_list, labels, step=0.1, objective="auc"):
    """对每个 class 在 K 维单纯形上粗搜最优权重，返回 [5, K].

    objective:
      - "auc"          : 优化 ROC AUC（默认，稳定但不一定最优 ref-rate F1）
      - "ref_rate_f1"  : 直接优化 ref-rate top-k macroF1（与 LB 评分方式一致，单列收益更高）
    """
    K = len(probs_list)
    grid = [w for w in product(np.arange(0, 1.0 + step / 2, step), repeat=K)
            if abs(sum(w) - 1.0) < 1e-6]
    n = len(labels)
    W = np.zeros((5, K), np.float64)
    for j, name in enumerate(N):
        y = labels[:, j]
        if y.sum() == 0 or y.sum() == len(y):
            W[j] = np.full(K, 1.0 / K)
            continue
        if objective == "ref_rate_f1":
            kk = int(round(REF[name] * n))
            best_f, best_w = -1.0, None
            for w in grid:
                mix = sum(w[k] * probs_list[k][:, j] for k in range(K))
                pred = np.zeros(n, dtype=np.int32)
                pred[np.argsort(-mix)[:kk]] = 1
                f = f1_score(y, pred, zero_division=0)
                if f > best_f:
                    best_f, best_w = f, w
            W[j] = best_w
        else:  # auc
            best_auc, best_w = -1.0, None
            for w in grid:
                mix = sum(w[k] * probs_list[k][:, j] for k in range(K))
                au = roc_auc_score(y, mix)
                if au > best_auc:
                    best_auc, best_w = au, w
            W[j] = best_w
    return W


def load_test(probs_paths, ids_paths):
    probs, ids = [], None
    for p, ip in zip(probs_paths, ids_paths):
        x = np.load(p).astype(np.float64)
        i = [l.strip().zfill(4) for l in open(ip)]
        if ids is None:
            ids = i
        else:
            assert i == ids, f"test ids 不一致: {ip}"
        probs.append(x)
    return probs, ids


def refrate_topk(probs):
    """[N, 5] -> [N, 5] 0/1 ref-rate top-k 二值化."""
    n = len(probs)
    pred = np.zeros_like(probs, dtype=np.int32)
    for j, name in enumerate(N):
        k = int(round(REF[name] * n))
        pred[np.argsort(probs[:, j])[-k:], j] = 1
    return pred


def main():
    args = parse_args()
    assert len(args.test_probs) == len(args.test_ids), \
        "test_probs 与 test_ids 数量必须一致"
    K = len(args.test_probs)

    test_probs, test_ids = load_test(args.test_probs, args.test_ids)
    n_test = len(test_ids)
    print(f"[fusion] 加载 {K} 个成员 test probs, n={n_test}")

    if args.weights_json:
        import json
        raw = json.loads(Path(args.weights_json).read_text())
        W_dict = raw.get("weights", raw)   # 兼容含 metadata 包装 {"order":[..],"weights":{..}}
        W = np.zeros((5, K), np.float64)
        for j, name in enumerate(N):
            w = W_dict[name]
            assert len(w) == K and abs(sum(w) - 1.0) < 1e-3, \
                f"weights_json[{name}] 需 K={K} 项且和≈1.0，实得 {w}"
            W[j] = w
        print(f"[fusion] 从 {args.weights_json} 加载 per-class 权重：")
        for j, name in enumerate(N):
            wstr = "/".join(f"{w:.2f}" for w in W[j])
            print(f"  {name}: [{wstr}]")
    elif args.weights:
        w_user = [float(x) for x in args.weights.split(",")]
        assert len(w_user) == K and abs(sum(w_user) - 1.0) < 1e-4, \
            f"weights 长度需为 {K} 且和为 1.0"
        W = np.tile(np.asarray(w_user), (5, 1))
        print(f"[fusion] 用户指定权重（5 列共用）: {w_user}")
    elif args.oof:
        assert len(args.oof) == K, "oof 与 test_probs 数量必须一致"
        oof_probs, oof_labels = load_oof_aligned(args.oof)
        print(f"[fusion] OOF 对齐 {len(oof_labels)} 样本，开始 per-class 权重搜索"
              f"（目标={args.objective}）…")
        W = search_per_class_weights(oof_probs, oof_labels,
                                     step=args.grid_step, objective=args.objective)
        for j, name in enumerate(N):
            wstr = "/".join(f"{w:.1f}" for w in W[j])
            print(f"  {name}: [{wstr}]")
        # 报告 OOF 上的 tuned + ref-rate macro-F1
        blended = np.zeros_like(oof_probs[0])
        for j in range(5):
            blended[:, j] = sum(W[j, k] * oof_probs[k][:, j] for k in range(K))
        ref_pred = refrate_topk(blended)
        f1s = [f1_score(oof_labels[:, j], ref_pred[:, j], zero_division=0)
               for j in range(5)]
        print(f"[fusion] OOF ref-rate macroF1={np.mean(f1s):.4f}  "
              f"({', '.join(f'{n}={f:.4f}' for n, f in zip(N, f1s))})")
    else:
        W = np.full((5, K), 1.0 / K)
        print(f"[fusion] 等权融合（无 OOF/weights）")

    # 应用到 test
    fused = np.zeros((n_test, 5), np.float64)
    for j in range(5):
        fused[:, j] = sum(W[j, k] * test_probs[k][:, j] for k in range(K))

    out_probs = Path(args.out_probs); out_probs.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_probs, fused.astype(np.float32))
    print(f"[fusion] 写出融合 probs: {out_probs}")

    pred = refrate_topk(fused)
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["segment_id"] + N)
        for sid, row in zip(test_ids, pred):
            w.writerow([sid] + row.tolist())
    print(f"[fusion] 写出 csv: {out_csv} "
          f"(列计数 " + " / ".join(f"{N[j]}={pred[:, j].sum()}" for j in range(5)) + ")")

    out_ids = Path(args.out_ids); out_ids.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ids, "w") as f:
        f.write("\n".join(test_ids) + "\n")


if __name__ == "__main__":
    main()
