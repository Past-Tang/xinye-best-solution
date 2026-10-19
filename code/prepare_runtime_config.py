#!/usr/bin/env python3
"""Create a relocatable training/inference config from a submitted YAML file."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--cache-root", required=True)
    return parser.parse_args()


def local_model_path(model_root: Path, original: str, kind: str) -> str:
    name = str(original).lower()
    if kind == "text":
        candidate = model_root / "Qwen3-0.6B"
    elif "hubert" in name:
        candidate = model_root / "chinese-hubert-large"
    elif "wavlm" in name:
        candidate = model_root / "wavlm-large"
    else:
        candidate = model_root / "whisper-large-v3"
    if not candidate.is_dir():
        raise FileNotFoundError(f"Required base model directory not found: {candidate}")
    return str(candidate.resolve())


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    code_root = Path(args.code_root).resolve()
    data_root = Path(args.data_root).resolve()
    run_root = Path(args.run_root).resolve()
    model_root = Path(args.model_root).resolve()
    cache_root = Path(args.cache_root).resolve()

    with input_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    paths = cfg.setdefault("paths", {})
    paths.update(
        {
            "project_root": str(code_root),
            "train_audio_dir": str(data_root / "train" / "audio"),
            "train_text_dir": str(data_root / "train" / "text"),
            "train_labels_dir": str(data_root / "train" / "labels"),
            "output_root": str(run_root),
            "checkpoints_dir": str(run_root / "checkpoints"),
            "logs_dir": str(run_root / "logs"),
            "cache_root": str(cache_root),
        }
    )

    if "audio_encoder" in cfg:
        original = str(cfg["audio_encoder"].get("model_name", ""))
        cfg["audio_encoder"]["model_name"] = local_model_path(
            model_root, original, "audio"
        )
    if "text_encoder" in cfg:
        original = str(cfg["text_encoder"].get("model_name", ""))
        cfg["text_encoder"]["model_name"] = local_model_path(
            model_root, original, "text"
        )

    env = cfg.setdefault("env", {})
    env.update(
        {
            "HF_HOME": str(cache_root / "huggingface"),
            "TRANSFORMERS_CACHE": str(cache_root / "huggingface"),
            "TORCH_HOME": str(cache_root / "torch"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
    print(f"[config] {input_path.name} -> {output_path}")


if __name__ == "__main__":
    main()
