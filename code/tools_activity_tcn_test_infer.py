"""TCN test inference using existing best.pt — non-destructive (separate from training).

Reuses ActivityTCN model class, ActivityTestDataset and feature builders from the
training script. Loads best.pt (which contains model state + mean/std/feature_names),
runs test_root inference and saves segment-sorted probs + ids in the standard format.

Output:
  outputs/test_probs/activity_tcn_test.npy        [1000, 5] float32 (c,na,i,bc,t)
  outputs/test_probs/activity_tcn_test_ids.txt    1000 segment ids

Usage:
  python tools_activity_tcn_test_infer.py \
    --checkpoint outputs_activity_tcn_grid25_gpu0/checkpoints/best.pt \
    --test_root /workspace/xinye/data/test \
    --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tools_activity_tcn_train import (
    ActivityTCN,
    ActivityTestDataset,
    build_feature_map,
    predict_test,
)
from tools_audio_state_quick_oof import OUT_NAMES, OUT_TO_CFG, load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_root", required=True)
    p.add_argument("--config", default="configs/sota_e12_ckpt_config.yaml")
    p.add_argument("--output_probs", default="outputs/test_probs/activity_tcn_test.npy")
    p.add_argument("--output_ids", default="outputs/test_probs/activity_tcn_test_ids.txt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = here / config_path
    test_root = Path(args.test_root)
    out_probs = Path(args.output_probs)
    if not out_probs.is_absolute():
        out_probs = here / out_probs
    out_ids = Path(args.output_ids)
    if not out_ids.is_absolute():
        out_ids = here / out_ids

    cfg = load_config(config_path)
    chunk_ms = int(cfg["chunk_ms"])
    context_chunks = int(cfg["context_chunks"])
    labels_cfg = cfg["labels"]
    raw_ids = [int(labels_cfg[OUT_TO_CFG[name]]) for name in OUT_NAMES]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = here / ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    feature_names = list(ckpt["feature_names"])
    train_args = ckpt.get("args", {})
    use_audio = not bool(train_args.get("no_audio", False))
    use_text = not bool(train_args.get("no_text", False))
    use_state = bool(train_args.get("use_state", False))
    hidden = int(train_args.get("hidden", 192))
    layers = int(train_args.get("layers", 6))
    dropout = float(train_args.get("dropout", 0.1))
    horizons_arg = train_args.get("horizons", "")
    horizons = tuple(int(h) for h in str(horizons_arg).split(",") if h.strip()) if horizons_arg else ()

    print(f"[ckpt] checkpoint={ckpt_path}")
    print(f"[ckpt] features={len(feature_names)} mean.shape={mean.shape} std.shape={std.shape}")
    print(f"[ckpt] arch hidden={hidden} layers={layers} horizons={horizons}")
    print(f"[ckpt] sources audio={use_audio} text={use_text} state={use_state}")

    model = ActivityTCN(
        in_dim=len(feature_names) + 1,  # +1 for rel_time appended by ActivityTestDataset
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        horizons=horizons,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    test_context_dir = test_root / "context"
    test_text_dir = test_root / "text"
    test_audio_dir = test_root / "audio"
    segment_ids = sorted(p.stem for p in test_context_dir.glob("*.npy"))
    print(f"[test] {len(segment_ids)} segments at {test_root}")

    test_feature_map, test_feature_names = build_feature_map(
        segment_ids,
        labels_dir=None,
        text_dir=test_text_dir,
        audio_dir=test_audio_dir,
        raw_ids=raw_ids,
        chunk_ms=chunk_ms,
        use_state=use_state,
        use_text=use_text,
        use_audio=use_audio,
        tag="test",
    )
    if test_feature_names != feature_names:
        # diagnose mismatch up to first 5 differences
        diffs = [(i, a, b) for i, (a, b) in enumerate(zip(test_feature_names, feature_names)) if a != b][:5]
        raise ValueError(f"test feature names mismatch: train_count={len(feature_names)} test_count={len(test_feature_names)}\n first diffs={diffs}")

    test_ds = ActivityTestDataset(segment_ids, test_feature_map, mean, std, context_chunks)
    loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    test_probs, test_ids = predict_test(model, loader, device)
    order = np.argsort(np.asarray(test_ids))
    test_probs = test_probs[order].astype(np.float32)
    test_ids_sorted = [test_ids[int(i)] for i in order]

    out_probs.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_probs, test_probs)
    out_ids.parent.mkdir(parents=True, exist_ok=True)
    out_ids.write_text("".join(f"{sid}\n" for sid in test_ids_sorted), encoding="utf-8")

    meta = {
        "checkpoint": str(ckpt_path),
        "test_root": str(test_root),
        "shape": list(test_probs.shape),
        "feature_count": len(feature_names),
        "feature_sources": {"audio": use_audio, "text": use_text, "state": use_state},
        "prob_mean": dict(zip(OUT_NAMES, test_probs.mean(axis=0).astype(float).tolist())),
        "best_metric": ckpt.get("best_metric", None),
    }
    out_probs.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] probs={out_probs} ids={out_ids} shape={test_probs.shape}")
    print(f"[stats] prob_mean={meta['prob_mean']}")


if __name__ == "__main__":
    main()