"""Dump test probs for a trained MultimodalLoRAChunkAuxModel checkpoint.

Mirrors tools/dump_test_probs_tuned.py but for this experiment's model factory.
Outputs:
  --output_probs : [N, 5] float32 npy in segment_id sorted order
  --output_ids   : N lines of segment_id
  --output_csv   : optional 0/1 CSV with header c,na,i,bc,t (LF), thresholds
                   passed via --thresholds "tc,tna,ti,tbc,tt".
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

BASE = Path(__file__).resolve().parent / "_lib"
HERE = Path(__file__).resolve().parent
for p in (str(BASE), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data import TurnTakingTestDataset, build_collate_fn  # noqa: E402
from src.utils import load_config, set_env_paths  # noqa: E402
from model.model import MultimodalLoRAChunkAuxModel  # noqa: E402
from model.temporal_model import MultimodalTemporalModel  # noqa: E402
from model.wavlm_temporal_model import MultimodalWavLMTemporalModel  # noqa: E402
from checkpoint_utils import load_task_checkpoint  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_root", required=True)
    p.add_argument("--output_probs", required=True)
    p.add_argument("--output_ids", required=True)
    p.add_argument("--output_csv", default=None)
    p.add_argument("--thresholds", default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_segments", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp_dtype", default="bf16")
    p.add_argument("--swap_channels", action="store_true",
                   help="TTA: swap L/R audio channels")
    return p.parse_args()


def parse_thresholds(value, n_labels):
    if value is None:
        return [0.5] * n_labels
    vals = [float(x) for x in value.split(",")]
    if len(vals) != n_labels:
        raise ValueError(f"thresholds length {len(vals)} != n_labels {n_labels}")
    return vals


def get_amp_dtype(name):
    n = str(name).lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_env_paths(cfg)
    multi_targets = list(cfg["labels"]["multi_targets"])
    label_cols = [x.lower() for x in multi_targets]
    thresholds = parse_thresholds(args.thresholds, len(label_cols))

    test_root = Path(args.test_root)
    ds = TurnTakingTestDataset(test_root=test_root, sample_rate=int(cfg["sample_rate"]))
    fa_test_dir = cfg.get("fa_feats", {}).get("test_dir", None)
    if fa_test_dir:
        ds.set_fa(fa_test_dir, int(cfg["chunk_ms"]))

    tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder"]["model_name"], use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    collate_fn = build_collate_fn(tokenizer, int(cfg["text_encoder"]["max_length"]))
    bs = int(args.batch_size or cfg["train"].get("eval_batch_size", 16))
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model_type = str(cfg.get("model", {}).get("type", "lora_chunkaux")).lower()
    if model_type in ("temporal", "temporal_aligned"):
        model = MultimodalTemporalModel(cfg).to(device)
    elif model_type in ("wavlm_temporal", "wavlm"):
        model = MultimodalWavLMTemporalModel(cfg).to(device)
    else:
        model = MultimodalLoRAChunkAuxModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Slim files may omit the base model, but never a task projection or head.
    load_task_checkpoint(model, ckpt)
    model.eval()

    amp_dtype = get_amp_dtype(args.amp_dtype)
    use_amp = (amp_dtype != torch.float32) and device.type == "cuda"

    all_probs = []
    all_ids = []
    done = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"dump_pb{'_swap' if args.swap_channels else ''}"):
            wav = batch["waveform"].to(device, non_blocking=True)
            if args.swap_channels and wav.dim() == 3 and wav.shape[1] == 2:
                wav = wav[:, [1, 0], :].contiguous()
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                extra = ({"fa_feats": batch["fa_feats"].to(device, non_blocking=True)}
                         if "fa_feats" in batch else {})
                main_logits, _, _, _ = model(
                    waveform=wav,
                    input_ids=batch["input_ids"].to(device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                    context_labels=batch["context_labels"].to(device, non_blocking=True),
                    return_aux=False,
                    **extra,
                )
            probs = torch.sigmoid(main_logits.float()).cpu().numpy()
            if probs.ndim == 1:
                probs = probs.reshape(-1, 1)
            ids = list(batch["segment_id"])
            if args.max_segments is not None and done + len(ids) > int(args.max_segments):
                keep = int(args.max_segments) - done
                probs = probs[:keep]
                ids = ids[:keep]
            all_probs.append(probs)
            all_ids.extend(ids)
            done += len(ids)
            if args.max_segments is not None and done >= int(args.max_segments):
                break

    probs = np.concatenate(all_probs, axis=0).astype(np.float32)
    order = np.argsort(np.asarray(all_ids))
    sorted_ids = [all_ids[i] for i in order]
    probs = probs[order]
    Path(args.output_probs).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_probs, probs)
    Path(args.output_ids).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_ids, "w", encoding="utf-8") as f:
        for sid in sorted_ids:
            f.write(f"{sid}\n")
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["segment_id"] + label_cols, lineterminator="\n")
            writer.writeheader()
            for i, sid in enumerate(sorted_ids):
                row = {"segment_id": sid}
                for j, col in enumerate(label_cols):
                    row[col] = int(float(probs[i, j]) >= thresholds[j])
                writer.writerow(row)
    print(f"wrote probs={args.output_probs} ids={args.output_ids} n={len(sorted_ids)} shape={probs.shape}")


if __name__ == "__main__":
    main()
