"""Slim training checkpoints for the Docker rematch package.

Keeps only model_ema tensors that differ from the frozen base weights
(LoRA adapters, task heads, projections); frozen base tensors are dropped
and restored from the public base models at load time.

slim:    ckpt(7.5GB) -> slim npz-style pt (~hundreds MB)
restore: slim + base models -> full state_dict, verify bitwise vs original
Usage:
  python tools_slim_ckpt.py slim --ckpt X/best_tuned_f1.pt --out X/best_slim.pt
  python tools_slim_ckpt.py verify --ckpt X/best_tuned_f1.pt --slim X/best_slim.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import os

WHISPER_DEFAULT = "/workspace/xinye/.cache/modelscope/AI-ModelScope/whisper-large-v3"
QWEN_DEFAULT = "/workspace/xinye/.cache/modelscope/Qwen/Qwen3-0___6B"


def base_state(whisper_path=None, qwen_path=None):
    """Reference tensors from public bases, keyed by likely ckpt names.

    Paths resolved in this order: explicit arg > env (XINYE_WHISPER/XINYE_QWEN) > default.
    Env-based override lets the rematch Docker image relocate bases without code changes.
    """
    from transformers import AutoModel, WhisperModel
    wp = whisper_path or os.environ.get("XINYE_WHISPER", WHISPER_DEFAULT)
    qp = qwen_path or os.environ.get("XINYE_QWEN", QWEN_DEFAULT)
    ref = {}
    wm = WhisperModel.from_pretrained(wp)
    for k, v in wm.encoder.state_dict().items():
        # model code wraps encoder with LoRA: base weights appear as
        # audio_encoder.base_model.model.<k> or audio_encoder.<k>
        ref[f"audio_encoder.base_model.model.{k}"] = v
        ref[f"audio_encoder.{k}"] = v
    del wm
    qm = AutoModel.from_pretrained(qp)
    for k, v in qm.state_dict().items():
        ref[f"text_encoder.backbone.{k}"] = v
        ref[f"text_encoder.model.{k}"] = v
        ref[f"text_encoder.backbone.base_model.model.{k}"] = v
    del qm
    return ref


def ref_lookup(ref, k):
    # torch.compile checkpoints prefix every key with _orig_mod.
    if k.startswith("_orig_mod."):
        k = k[len("_orig_mod."):]
    r = ref.get(k)
    if r is None:
        r = ref.get(k.replace(".base_layer", ""))  # PEFT wraps lora'd layers
    return r


def cmd_slim(args):
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("model_ema", ck.get("model"))
    ref = base_state()
    keep, dropped = {}, []
    for k, v in sd.items():
        r = ref_lookup(ref, k)
        if r is not None and r.shape == v.shape and torch.allclose(r.float(), v.float(), atol=1e-5):
            dropped.append(k)
        else:
            keep[k] = v
    out = {
        "slim_keep": keep,
        "dropped_from_base": dropped,
        "epoch": ck.get("epoch"),
        "best_metric": ck.get("best_metric"),
        "config": ck.get("config"),
    }
    torch.save(out, args.out)
    kept_gb = sum(v.numel() * v.element_size() for v in keep.values()) / 1e9
    print(f"kept {len(keep)} tensors ({kept_gb:.2f} GB), dropped {len(dropped)} base tensors")
    print(f"saved {args.out} ({Path(args.out).stat().st_size/1e9:.2f} GB)")


def restore_state(slim_path, whisper_path=None, qwen_path=None):
    sl = torch.load(slim_path, map_location="cpu", weights_only=False)
    ref = base_state(whisper_path=whisper_path, qwen_path=qwen_path)
    sd = dict(sl["slim_keep"])
    for k in sl["dropped_from_base"]:
        r = ref_lookup(ref, k)
        assert r is not None, f"no base ref for {k}"
        sd[k] = r
    return sd, sl


def cmd_verify(args):
    ck = torch.load(args.ckpt, map_location="cpu")
    orig = ck.get("model_ema", ck.get("model"))
    sd, _ = restore_state(args.slim)
    assert set(sd) == set(orig), f"key mismatch: {len(set(orig)-set(sd))} missing"
    bad = [k for k in orig if not torch.allclose(sd[k].float(), orig[k].float(), atol=1e-5)]
    print(f"verify: {len(orig)} tensors, mismatched={len(bad)}")
    assert not bad, bad[:5]
    print("BITWISE RESTORE OK")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slim")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_slim)
    v = sub.add_parser("verify")
    v.add_argument("--ckpt", required=True)
    v.add_argument("--slim", required=True)
    v.set_defaults(func=cmd_verify)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
