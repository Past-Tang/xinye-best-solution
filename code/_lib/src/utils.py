import json
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs(cfg: Dict) -> None:
    Path(cfg["paths"]["output_root"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["cache_root"]).mkdir(parents=True, exist_ok=True)


def set_env_paths(cfg: Dict) -> None:
    for k, v in cfg.get("env", {}).items():
        os.environ[k] = str(v)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    if not is_distributed():
        return 0, 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    # 验证/测试若只在 rank0 跑全量，其它 rank 会在 barrier 等待；默认 NCCL 约 600s 会超时
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=timedelta(hours=2),
    )
    return local_rank, world_size, rank


def cleanup_distributed() -> None:
    if is_distributed() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def compute_binary_metrics(labels, probs) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    else:
        metrics["roc_auc"] = 0.5
    return metrics


def compute_multilabel_metrics(labels, probs, label_names=None) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    if labels.ndim != 2 or probs.ndim != 2:
        raise ValueError(f"Expected 2D labels/probs, got {labels.shape} and {probs.shape}")
    if labels.shape != probs.shape:
        raise ValueError(f"Shape mismatch: labels {labels.shape} vs probs {probs.shape}")

    n_labels = labels.shape[1]
    if label_names is None:
        label_names = [f"label{i}" for i in range(n_labels)]
    if len(label_names) != n_labels:
        raise ValueError(f"label_names length {len(label_names)} != n_labels {n_labels}")

    out: Dict[str, float] = {}
    per_acc, per_f1, per_auc = [], [], []
    for i, name in enumerate(label_names):
        y = labels[:, i]
        p = probs[:, i]
        pred = (p >= 0.5).astype(int)
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, zero_division=0))
        if len(np.unique(y)) > 1:
            auc = float(roc_auc_score(y, p))
        else:
            auc = 0.5
        out[f"{name}_accuracy"] = acc
        out[f"{name}_f1"] = f1
        out[f"{name}_roc_auc"] = auc
        per_acc.append(acc)
        per_f1.append(f1)
        per_auc.append(auc)

    out["macro_accuracy"] = float(np.mean(per_acc))
    out["macro_f1"] = float(np.mean(per_f1))
    out["macro_roc_auc"] = float(np.mean(per_auc))
    # Alias for backward-compatible save_metric/print flow
    out["accuracy"] = out["macro_accuracy"]
    out["f1"] = out["macro_f1"]
    out["roc_auc"] = out["macro_roc_auc"]
    return out


@torch.no_grad()
def compute_gaussian_soft_f1_sequence(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 5,
    sigma: float = 2.0,
    avg_class_indices: tuple[int, ...] = (1, 2, 3),
    epsilon: float = 1e-8,
) -> Dict[str, float]:
    """
    高斯平滑时序 soft-f1（按类别做 TP/FP/FN 的 soft 版本）。

    probs: [B, C, T]，每个时间步每类的概率（例如 softmax 后）。
    targets: [B, T]，每个时间步的类别id（0..C-1）。
    """
    if probs.ndim != 3:
        raise ValueError(f"Expected probs shape [B,C,T], got {tuple(probs.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"Expected targets shape [B,T], got {tuple(targets.shape)}")
    b, c, t = probs.shape
    if c != num_classes:
        raise ValueError(f"probs C={c} != num_classes={num_classes}")
    if targets.shape[0] != b or targets.shape[1] != t:
        raise ValueError(f"targets shape {tuple(targets.shape)} not match probs {tuple(probs.shape)}")

    targets_onehot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 2, 1).float()  # [B,C,T]

    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = torch.arange(kernel_size, device=probs.device).float() - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.max()
    kernel = kernel.view(1, 1, -1)  # [1,1,K]

    padding = kernel_size // 2

    # 平滑 targets：对每个 (B,C) 位置做 conv1d
    targets_flat = targets_onehot.reshape(b * c, 1, t)  # [B*C,1,T]
    targets_smooth = F.conv1d(targets_flat, kernel, padding=padding).view(b, c, t)  # [B,C,T]

    # soft TP/FP/FN 形式的一种等价推导
    tp = (probs * targets_smooth).sum(dim=(0, 2))  # [C]
    sum_p = probs.sum(dim=(0, 2))  # [C]
    sum_t_true = targets_onehot.sum(dim=(0, 2))  # [C]

    f1 = (2 * tp + epsilon) / (sum_p + sum_t_true + epsilon)  # [C]
    avg_class_indices = tuple(avg_class_indices)
    score = f1[list(avg_class_indices)].mean()

    return {
        "soft_macro_f1": float(score.item()),
        "soft_f1_per_class_mean": float(f1.mean().item()),
    }


def save_json(path: Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
