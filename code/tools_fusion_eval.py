"""融合评估: 在对齐的统一 valid 评测集上量化任意模型组合的增益.

输入: 多个 tools_oof_valid.py 产出的 .npz (probs/labels/conv_ids/end_idxs)
功能:
  1. 按 (conv_id,end_idx) 求多个 npz 的公共交集并严格对齐
  2. 报告每个模型 per-class AUC + tuned macro-F1
  3. 报告两两概率相关性 (低相关=融合潜力大)
  4. 网格搜索最优线性融合权重, 报告 blend 的 per-class AUC 与 macro-F1 增益
  5. ref-rate top-k 二值化下的 macro-F1 (模拟 LB 评分方式)

用法:
  python tools_fusion_eval.py --inputs outputs/oof/sota_valid.npz outputs/oof/temporal_valid.npz \
    --names SOTA TEMPORAL
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

N = ["c", "na", "i", "bc", "t"]
REF = {"c": 0.964, "na": 0.862, "i": 0.097, "bc": 0.031, "t": 0.450}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--names", nargs="+", default=None)
    p.add_argument("--ref_rate", action="store_true",
                   help="同时报告 ref-rate top-k 二值化的 macro-F1 (模拟 test 评分)")
    return p.parse_args()


def load_aligned(paths):
    """加载多个 npz, 按 (conv_id,end_idx) 求交集并对齐. 返回 (probs_list[K][N,5], labels[N,5])."""
    data = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        keys = list(zip([str(x) for x in d["conv_ids"]], [int(x) for x in d["end_idxs"]]))
        kmap = {k: i for i, k in enumerate(keys)}
        data.append((d["probs"], d["labels"], kmap))
    common = set(data[0][2].keys())
    for _, _, kmap in data[1:]:
        common &= set(kmap.keys())
    common = sorted(common)
    if not common:
        raise SystemExit("ERROR: 输入文件无公共 (conv_id,end_idx) 样本, 无法对齐")
    probs_list = []
    labels_ref = None
    for probs, labels, kmap in data:
        idx = [kmap[k] for k in common]
        probs_list.append(probs[idx])
        if labels_ref is None:
            labels_ref = labels[idx]
        else:
            assert np.array_equal(labels_ref, labels[idx]), "标签不一致, 对齐失败"
    return probs_list, labels_ref, len(common)


def tuned_macro_f1(probs, labels):
    f1s = []
    for j in range(5):
        y = labels[:, j]; p = probs[:, j]
        best = 0.0
        for t in np.arange(0.05, 0.95, 0.01):
            f = f1_score(y, p >= t, zero_division=0)
            if f > best:
                best = f
        f1s.append(best)
    return f1s


def refrate_macro_f1(probs, labels):
    f1s = []
    for j, n in enumerate(N):
        k = int(round(REF[n] * len(probs)))
        pred = np.zeros(len(probs), dtype=int)
        pred[np.argsort(probs[:, j])[-k:]] = 1
        f1s.append(f1_score(labels[:, j], pred, zero_division=0))
    return f1s


def main():
    args = parse_args()
    names = args.names or [Path(p).stem for p in args.inputs]
    probs_list, labels, n_common = load_aligned(args.inputs)
    K = len(probs_list)
    print(f"=== 对齐成功: {n_common} 个公共样本, {K} 个模型 ===\n")

    # 1. 每个模型的 per-class AUC + macro-F1
    print("--- 单模型 per-class AUC (tuned macro-F1) ---")
    aucs = {}
    for k in range(K):
        a = [roc_auc_score(labels[:, j], probs_list[k][:, j]) if labels[:, j].sum() > 0 else float("nan")
             for j in range(5)]
        aucs[k] = a
        mf = float(np.mean(tuned_macro_f1(probs_list[k], labels)))
        print(f"  {names[k]:>10}: " + " ".join(f"{N[j]}={a[j]:.4f}" for j in range(5)) + f"  | macroF1={mf:.4f}")
    print()

    # 2. 两两概率相关性 (per-class)
    if K >= 2:
        print("--- 两两概率相关性 (越低融合潜力越大) ---")
        for a in range(K):
            for b in range(a + 1, K):
                cors = [np.corrcoef(probs_list[a][:, j], probs_list[b][:, j])[0, 1] for j in range(5)]
                print(f"  {names[a]}×{names[b]}: " + " ".join(f"{N[j]}={cors[j]:.3f}" for j in range(5)))
        print()

    # 3. 最优线性融合 (per-class 独立搜权重) 的 AUC 增益
    if K >= 2:
        print("--- 最优 per-class 线性融合 AUC vs 最佳单模型 ---")
        base_auc = {j: max(aucs[k][j] for k in range(K)) for j in range(5)}
        blended = np.zeros_like(probs_list[0])
        for j in range(5):
            y = labels[:, j]
            if y.sum() == 0:
                blended[:, j] = np.mean([probs_list[k][:, j] for k in range(K)], axis=0)
                continue
            best = (-1.0, None)
            # 在 K 模型单纯形上粗搜 (步长 0.1)
            grid = [w for w in product(np.arange(0, 1.01, 0.1), repeat=K) if abs(sum(w) - 1.0) < 1e-6]
            for w in grid:
                mix = sum(w[k] * probs_list[k][:, j] for k in range(K))
                au = roc_auc_score(y, mix)
                if au > best[0]:
                    best = (au, w)
            blended[:, j] = sum(best[1][k] * probs_list[k][:, j] for k in range(K))
            gain = best[0] - base_auc[j]
            wstr = "/".join(f"{names[k][:4]}:{best[1][k]:.1f}" for k in range(K))
            print(f"  {N[j]}: bestSingle={base_auc[j]:.4f} blend={best[0]:.4f} gain={gain:+.4f}  [{wstr}]")
        print()
        mf_blend = float(np.mean(tuned_macro_f1(blended, labels)))
        mf_best_single = max(float(np.mean(tuned_macro_f1(probs_list[k], labels))) for k in range(K))
        print(f"  融合 tuned macroF1={mf_blend:.4f}  vs 最佳单模型={mf_best_single:.4f}  gain={mf_blend-mf_best_single:+.4f}")

    # 4. ref-rate macro-F1 (模拟 LB)
    if args.ref_rate:
        print("\n--- ref-rate top-k macro-F1 (模拟 test 评分方式) ---")
        for k in range(K):
            f1s = refrate_macro_f1(probs_list[k], labels)
            print(f"  {names[k]:>10}: " + " ".join(f"{N[j]}={f1s[j]:.4f}" for j in range(5)) + f"  | macro={np.mean(f1s):.4f}")


if __name__ == "__main__":
    main()