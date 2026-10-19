"""Train Whisper-LoRA + chunk-aux + Mix2 + Robust ASL.

Reuses baseline dataset / collate / split via sys.path injection.
Distinct from baseline: model factory, loss, dual param groups + curriculum LR,
DDP find_unused_parameters=False (LoRA always active; no conditional branches).
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

# ---- baseline imports (do NOT modify baseline) ----
BASE = Path(__file__).resolve().parent / "_lib"
HERE = Path(__file__).resolve().parent
for p in (str(BASE), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data import (  # noqa: E402
    TurnTakingTrainDataset,
    build_collate_fn,
    build_train_samples_multitask,
    list_conv_ids,
    split_conversation_ids,
)
from src.utils import (  # noqa: E402
    cleanup_distributed,
    compute_multilabel_metrics,
    ensure_dirs,
    is_distributed,
    load_config,
    save_json,
    set_env_paths,
    set_seed,
    setup_distributed,
)

# ---- this experiment ----
from model.model import MultimodalLoRAChunkAuxModel  # noqa: E402
from model.temporal_model import MultimodalTemporalModel  # noqa: E402
from model.wavlm_temporal_model import MultimodalWavLMTemporalModel  # noqa: E402
from model.lora_wrapper import count_trainable_params  # noqa: E402
from model.losses import (  # noqa: E402
    AuxWeightSchedule,
    RobustASL,
    TotalLoss,
    chunk_aux_loss,
)
from model.mix2 import Mix2Selector  # noqa: E402
from model.curriculum import (  # noqa: E402
    CurriculumConfig,
    make_curriculum_lambdas,
    split_param_groups,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--init_model", default=None,
                   help="Load only model weights (not optimizer state).")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max_steps_per_epoch", type=int, default=None)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_valid_samples", type=int, default=None)
    return p.parse_args()


def get_amp_dtype(name: str) -> torch.dtype:
    n = str(name).lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


@torch.no_grad()
def collect_probs(model, data_loader, device, amp_dtype, show_progress: bool = True):
    """Run main head only (no aux), gather sigmoid probs and labels per rank."""
    model.eval()
    all_probs, all_labels = [], []
    iterator = tqdm(data_loader, desc="eval", leave=False) if show_progress else data_loader
    for batch in iterator:
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            extra = ({"fa_feats": batch["fa_feats"].to(device, non_blocking=True)}
                     if "fa_feats" in batch else {})
            main_logits, _, _, _ = model(
                waveform=batch["waveform"].to(device, non_blocking=True),
                input_ids=batch["input_ids"].to(device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                context_labels=batch["context_labels"].to(device, non_blocking=True),
                return_aux=False,
                **extra,
            )
        probs = torch.sigmoid(main_logits.float()).cpu().numpy()
        labels = batch["label"].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)
    if not all_probs:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5), dtype=np.int64)
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def compute_tuned_metrics(labels, probs, label_names, t_min, t_max, t_step):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    fixed = compute_multilabel_metrics(labels, probs, label_names=label_names)
    grid = np.arange(float(t_min), float(t_max) + 1e-12, float(t_step))
    thresholds = []
    tuned_f1s = []
    rates = []
    for j, name in enumerate(label_names):
        y = labels[:, j]
        p = probs[:, j]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            sc = float(f1_score(y, p >= t, zero_division=0))
            if sc > best_f1:
                best_f1, best_t = sc, float(t)
        thresholds.append(best_t)
        tuned_f1s.append(best_f1)
        rates.append(float((p >= best_t).mean()))
        fixed[f"{name}_tuned_f1"] = best_f1
        fixed[f"{name}_tuned_threshold"] = best_t
        fixed[f"{name}_tuned_pred_pos_rate"] = rates[-1]
        fixed[f"{name}_true_pos_rate"] = float(y.mean())
    fixed["tuned_macro_f1"] = float(np.mean(tuned_f1s))
    fixed["tuned_thresholds"] = {n: float(t) for n, t in zip(label_names, thresholds)}
    fixed["tuned_pred_pos_rates"] = {n: float(r) for n, r in zip(label_names, rates)}
    fixed["selection_metric"] = "tuned_macro_f1"
    return fixed


def format_metrics(metrics, label_names):
    parts = []
    for name in label_names:
        parts.append(
            f"{name}:f1@0.5={metrics.get(f'{name}_f1', 0.0):.4f},"
            f"tuned={metrics.get(f'{name}_tuned_f1', 0.0):.4f},"
            f"thr={metrics.get(f'{name}_tuned_threshold', 0.5):.3f},"
            f"auc={metrics.get(f'{name}_roc_auc', 0.5):.4f}"
        )
    return " | ".join(parts)


def save_ckpt(path, epoch, best_metric, model, optimizer, scheduler, cfg, bad_epochs, ema_model=None):
    state = {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        # save full state_dict (LoRA-only would require PEFT save_pretrained;
        # full dict is ~2.6 GB but reload-compatible across any code change)
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": cfg,
        "bad_epochs": int(bad_epochs),
    }
    if ema_model is not None:
        state["model_ema"] = ema_model.state_dict()
    torch.save(state, path)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_env_paths(cfg)
    ensure_dirs(cfg)
    set_seed(int(cfg["seed"]))

    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.max_steps_per_epoch is not None:
        cfg["train"]["max_steps_per_epoch"] = int(args.max_steps_per_epoch)
    if args.max_train_samples is not None:
        cfg["max_train_samples"] = int(args.max_train_samples)
    if args.max_valid_samples is not None:
        cfg["max_valid_samples"] = int(args.max_valid_samples)

    local_rank, world_size, rank = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    writer = None

    paths = cfg["paths"]
    labels_dir = Path(paths["train_labels_dir"])
    train_audio_dir = Path(paths["train_audio_dir"])
    train_text_dir = Path(paths["train_text_dir"])
    checkpoints_dir = Path(paths["checkpoints_dir"])
    logs_dir = Path(paths["logs_dir"])
    intermediate_dir = Path(paths["output_root"]) / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # ---- split ----
    conv_ids = list_conv_ids(labels_dir)
    split_cfg = cfg.get("split", {})
    split_seed = int(split_cfg.get("seed", cfg["seed"]))
    tuned_eval_seed = int(cfg.get("train", {}).get("tuned_eval_subset_seed", cfg["seed"]))
    # extra-corpus convs (by id prefix) join train only; valid split stays
    # identical to the official-only split so OOF stays aligned.
    extra_prefix = cfg.get("extra_train", {}).get("id_prefix", None)
    if extra_prefix:
        extra_ids = [c for c in conv_ids if c.startswith(extra_prefix)]
        conv_ids = [c for c in conv_ids if not c.startswith(extra_prefix)]
    else:
        extra_ids = []
    split_ids = split_conversation_ids(conv_ids, float(split_cfg["valid_ratio"]), split_seed)
    train_ids, valid_ids = split_ids["train"] + extra_ids, split_ids["valid"]
    _mt = list(cfg.get("labels", {}).get("multi_targets", ["C", "NA", "I", "BC", "T"]))
    extra_loss_mask_cols = [_mt.index(x) for x in cfg.get("extra_train", {}).get("loss_mask_labels", [])
                            if x in _mt]
    if extra_ids and int(os.environ.get("RANK", "0")) == 0:
        print(f"[extra_train] +{len(extra_ids)} extra convs (prefix={extra_prefix}); "
              f"official train={len(split_ids['train'])} valid={len(valid_ids)}; "
              f"loss_mask_cols={extra_loss_mask_cols}", flush=True)

    multi_targets = list(cfg.get("labels", {}).get("multi_targets", ["C", "NA", "I", "BC", "T"]))
    metric_label_names = [x.lower() for x in multi_targets]

    train_samples = build_train_samples_multitask(
        labels_dir=labels_dir,
        conv_ids=train_ids,
        context_chunks=int(cfg["context_chunks"]),
        target_chunks=int(cfg["target_chunks"]),
        stride=int(cfg["stride"]),
        label_ids=cfg["labels"],
        target_labels=multi_targets,
        max_samples=cfg["max_train_samples"],
    )
    valid_samples = build_train_samples_multitask(
        labels_dir=labels_dir,
        conv_ids=valid_ids,
        context_chunks=int(cfg["context_chunks"]),
        target_chunks=int(cfg["target_chunks"]),
        stride=int(cfg["stride"]),
        label_ids=cfg["labels"],
        target_labels=multi_targets,
        max_samples=cfg["max_valid_samples"],
    )

    subset_size = cfg["train"].get("tuned_eval_subset", None)
    if subset_size is not None and len(valid_samples) > int(subset_size):
        rng = np.random.default_rng(tuned_eval_seed)
        idx = rng.choice(len(valid_samples), size=int(subset_size), replace=False)
        idx.sort()
        tuned_valid_samples = [valid_samples[i] for i in idx]
    else:
        tuned_valid_samples = valid_samples
    rank_tuned_valid_samples = (
        tuned_valid_samples[rank::world_size] if is_distributed() else tuned_valid_samples
    )

    if is_main:
        save_json(logs_dir / "split_ids.json", split_ids)
        save_json(logs_dir / "sample_count.json", {
            "train_samples": len(train_samples),
            "valid_samples": len(valid_samples),
            "tuned_valid_samples": len(tuned_valid_samples),
            "rank0_tuned_valid_samples": len(rank_tuned_valid_samples),
            "world_size": world_size,
        })
        writer = SummaryWriter(log_dir=str(logs_dir / "tb"))

    # ---- tokenizer + collate ----
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder"]["model_name"], use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    collate_fn = build_collate_fn(tokenizer, int(cfg["text_encoder"]["max_length"]))

    future_horizons = tuple(cfg.get("future_aux", {}).get("horizons", []))
    fa_dir = cfg.get("fa_feats", {}).get("train_dir", None)
    audio_aug_cfg = cfg.get("audio_augment", None)
    train_dataset = TurnTakingTrainDataset(
        samples=train_samples,
        train_audio_dir=train_audio_dir,
        train_text_dir=train_text_dir,
        train_labels_dir=labels_dir,
        context_chunks=int(cfg["context_chunks"]),
        target_chunks=int(cfg["target_chunks"]),
        chunk_ms=int(cfg["chunk_ms"]),
        sample_rate=int(cfg["sample_rate"]),
        future_horizons=future_horizons if future_horizons else None,
        fa_dir=fa_dir,
        audio_augment=audio_aug_cfg,
    )
    tuned_valid_dataset = TurnTakingTrainDataset(
        samples=rank_tuned_valid_samples,
        train_audio_dir=train_audio_dir,
        train_text_dir=train_text_dir,
        train_labels_dir=labels_dir,
        context_chunks=int(cfg["context_chunks"]),
        target_chunks=int(cfg["target_chunks"]),
        chunk_ms=int(cfg["chunk_ms"]),
        sample_rate=int(cfg["sample_rate"]),
        fa_dir=fa_dir,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    pin_memory = bool(cfg.get("pin_memory", True))
    persistent_workers = bool(cfg.get("persistent_workers", False)) and int(cfg["num_workers"]) > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    tuned_valid_loader = DataLoader(
        tuned_valid_dataset,
        batch_size=int(cfg["train"].get("eval_batch_size", 16)),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # ---- model ----
    model_type = str(cfg.get("model", {}).get("type", "lora_chunkaux")).lower()
    if model_type in ("temporal", "temporal_aligned"):
        model = MultimodalTemporalModel(cfg).to(device)
    elif model_type in ("wavlm_temporal", "wavlm"):
        model = MultimodalWavLMTemporalModel(cfg).to(device)
    else:
        model = MultimodalLoRAChunkAuxModel(cfg).to(device)
    if is_main:
        tr, tot = count_trainable_params(model)
        print(f"[model] trainable={tr/1e6:.2f}M / total={tot/1e6:.2f}M ({100*tr/tot:.3f}%)", flush=True)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Optional torch.compile (env-gated, default OFF to keep legacy behaviour).
    # Compiling before DDP wrap; fuses FFN matmul/GELU which is the GC+FFN bottleneck.
    # torch.compile is numerically near-equivalent (kernel fusion), so it does not
    # change the stride ablation attribution.
    if os.environ.get("COMPILE_MODEL", "0") == "1":
        _cmode = os.environ.get("COMPILE_MODE", "default")
        if is_main:
            print(f"[compile] torch.compile(model, mode={_cmode}) ...", flush=True)
        model = torch.compile(model, mode=_cmode)

    if is_distributed():
        ddp_find_unused = model_type in ("wavlm_temporal", "wavlm")
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    # ---- loss ----
    asl_kwargs = dict(cfg.get("robust_asl", {}))
    aux_sch_cfg = dict(cfg.get("aux_schedule", {}))
    aux_schedule = AuxWeightSchedule(**aux_sch_cfg) if aux_sch_cfg else AuxWeightSchedule()
    chunk_cfg = dict(cfg.get("chunk_aux", {}))
    aux_label_smoothing = float(chunk_cfg.get("label_smoothing", 0.1))
    future_cfg = dict(cfg.get("future_aux", {}))
    future_aux_label_smoothing = float(future_cfg.get("label_smoothing", 0.1))
    future_aux_weight = float(future_cfg.get("loss_weight", 0.3))
    # Per-class pos_weight for rare classes (e.g. BC at 3.9%).
    # Config format: train.pos_weight: [c_w, na_w, i_w, bc_w, t_w]
    # Order matches multi_targets (C, NA, I, BC, T).
    pos_weight_cfg = cfg.get("train", {}).get("pos_weight", None)
    mutex_weight_cfg = float(cfg.get("train", {}).get("mutex_weight", 0.0))
    bc_bce_weight_cfg = float(cfg.get("train", {}).get("bc_bce_weight", 0.0))
    bc_bce_pos_weight_cfg = float(cfg.get("train", {}).get("bc_bce_pos_weight", 25.0))
    # BC class index in multi_targets (C=0, NA=1, I=2, BC=3, T=4).
    multi_targets_list = list(cfg.get("labels", {}).get("multi_targets", []))
    bc_class_index_cfg = int(multi_targets_list.index("BC")) if "BC" in multi_targets_list else 3
    total_loss_fn = TotalLoss(asl_kwargs=asl_kwargs, aux_label_smoothing=aux_label_smoothing,
                              aux_schedule=aux_schedule,
                              future_aux_label_smoothing=future_aux_label_smoothing,
                              future_aux_weight=future_aux_weight,
                              pos_weight=pos_weight_cfg,
                              mutex_weight=mutex_weight_cfg,
                              bc_bce_weight=bc_bce_weight_cfg,
                              bc_bce_pos_weight=bc_bce_pos_weight_cfg,
                              bc_class_index=bc_class_index_cfg)

    # ---- mix2 ----
    mix2_cfg = dict(cfg.get("mix2", {}))
    mix2 = Mix2Selector(
        prob=float(mix2_cfg.get("prob", 0.5)),
        alpha=float(mix2_cfg.get("alpha", 0.4)),
        k_max=int(mix2_cfg.get("k_max", 3)),
        modes=tuple(mix2_cfg.get("modes", ["mixup", "multimix"])),
        seed=int(cfg["seed"]) + rank,
    )

    # ---- optimizer + curriculum scheduler ----
    base_lr = float(cfg["train"]["learning_rate"])
    weight_decay = float(cfg["train"]["weight_decay"])
    target_module = model.module if hasattr(model, "module") else model
    groups, group_names = split_param_groups(
        target_module, lora_marker="lora_", base_lr=base_lr, weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)

    max_epochs = int(cfg["train"]["epochs"])
    accum_steps = int(cfg["train"]["gradient_accumulation_steps"])
    max_steps_cfg = cfg["train"].get("max_steps_per_epoch", None)
    max_steps_per_epoch = int(max_steps_cfg) if max_steps_cfg is not None else None
    steps_per_epoch_eff = (
        min(len(train_loader), max_steps_per_epoch) if max_steps_per_epoch is not None
        else len(train_loader)
    )
    update_steps_per_epoch = max(1, steps_per_epoch_eff // max(1, accum_steps))
    total_update_steps = max(1, update_steps_per_epoch * max_epochs)
    warmup_steps = int(total_update_steps * float(cfg["train"].get("warmup_ratio", 0.05)))

    cur_cfg_dict = dict(cfg.get("curriculum", {}))
    cur_cfg = CurriculumConfig(**cur_cfg_dict) if cur_cfg_dict else CurriculumConfig()
    lambdas = make_curriculum_lambdas(
        total_steps=total_update_steps,
        warmup_steps=warmup_steps,
        steps_per_epoch=update_steps_per_epoch,
        cfg=cur_cfg,
        group_names=group_names,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    # ---- amp ----
    amp_dtype = get_amp_dtype(cfg["train"].get("amp_dtype", "bf16")) if cfg["train"].get("use_amp", True) else torch.float32
    use_amp = (amp_dtype != torch.float32)
    # bf16 does not need GradScaler (no underflow); fp16 needs it.
    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    grad_clip = float(cfg["train"]["grad_clip_norm"])
    early_stop_patience = int(cfg["train"].get("early_stop_patience", 6))
    save_epoch_ckpts = bool(cfg["train"].get("save_epoch_checkpoints", False))
    threshold_min = float(cfg["train"].get("tuned_eval_threshold_min", 0.05))
    threshold_max = float(cfg["train"].get("tuned_eval_threshold_max", 0.95))
    threshold_step = float(cfg["train"].get("tuned_eval_threshold_step", 0.01))
    log_every = int(cfg["train"].get("log_every_steps", 20))
    ema_decay = float(cfg["train"].get("ema_decay", 0.98))
    weight_ema_decay = float(cfg["train"].get("weight_ema_decay", 0.0))
    ema_model = None
    if weight_ema_decay > 0.0:
        ema_model = copy.deepcopy(target_module).cpu().eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    # ---- resume ----
    start_epoch = 0
    best_metric = -math.inf
    bad_epochs = 0
    best_path = checkpoints_dir / "best_tuned_f1.pt"
    last_path = checkpoints_dir / "last.pt"
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        missing, unexpected = target_module.load_state_dict(ckpt["model"], strict=False)
        if is_main and (missing or unexpected):
            print(f"[resume] model missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if ema_model is not None:
            if "model_ema" in ckpt:
                ema_model.load_state_dict(ckpt["model_ema"], strict=False)
            else:
                ema_model.load_state_dict(target_module.state_dict(), strict=False)
                if is_main:
                    print("[resume] no model_ema in ckpt; EMA cold-started from loaded weights", flush=True)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                if is_main:
                    print(f"[resume] optimizer load failed: {e}", flush=True)
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                pass
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_metric = float(ckpt.get("best_metric", -math.inf))
        bad_epochs = int(ckpt.get("bad_epochs", 0))
    elif args.init_model and Path(args.init_model).exists():
        ckpt = torch.load(args.init_model, map_location="cpu")
        missing, unexpected = target_module.load_state_dict(ckpt["model"], strict=False)
        if ema_model is not None:
            ema_model.load_state_dict(target_module.state_dict(), strict=False)
        if is_main:
            print(f"[init_model] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    # main training loop
    global_train_step = 0
    update_step = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        iterator = (
            tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
            if is_main else train_loader
        )
        epoch_loss_sum = 0.0
        epoch_main_sum = 0.0
        epoch_aux_sum = 0.0
        epoch_step_count = 0
        loss_ema = None

        for step, batch in enumerate(iterator):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                extra = ({"fa_feats": batch["fa_feats"].to(device, non_blocking=True)}
                         if "fa_feats" in batch else {})
                main_logits, aux_logits, fusion, future_aux_logits = model(
                    waveform=batch["waveform"].to(device, non_blocking=True),
                    input_ids=batch["input_ids"].to(device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                    context_labels=batch["context_labels"].to(device, non_blocking=True),
                    return_aux=True,
                    **extra,
                )
                main_targets = batch["label"].to(device, non_blocking=True)
                aux_targets = batch["context_labels"].to(device, non_blocking=True).long()
                future_aux_targets = None
                if "future_labels" in batch:
                    future_aux_targets = batch["future_labels"].to(device, non_blocking=True).long()

                # extra-corpus rows (e.g. RAMC pseudo-dual-channel): their I/BC
                # supervision is unreliable (overlap-free transcription style),
                # so mask those two columns out of the main loss.
                class_mask = None
                if extra_loss_mask_cols and "conv_id" in batch:
                    is_extra = torch.tensor(
                        [cid.startswith(extra_prefix) for cid in batch["conv_id"]],
                        device=device)
                    if is_extra.any():
                        class_mask = torch.ones_like(main_targets)
                        for ci in extra_loss_mask_cols:
                            class_mask[is_extra, ci] = 0.0

                # Mix2 on fusion features (and main targets only).
                # If mix triggered, re-run head on mixed fusion to get new logits.
                mixed_fusion, mixed_targets, mix_info = mix2(fusion, main_targets, training=True)
                if mix_info["mode"] != "none":
                    main_logits = (model.module.head_only(mixed_fusion)
                                   if hasattr(model, "module") else model.head_only(mixed_fusion))
                    main_targets = mixed_targets

                loss, loss_parts = total_loss_fn(
                    main_logits=main_logits,
                    main_targets=main_targets,
                    aux_logits=aux_logits,
                    aux_targets=aux_targets,
                    epoch=epoch,
                    future_aux_logits=future_aux_logits,
                    future_aux_targets=future_aux_targets,
                    class_mask=class_mask,
                )
                loss = loss / accum_steps

            if not torch.isfinite(loss):
                if is_main:
                    print(f"[WARN] non-finite loss at epoch={epoch} step={step}, skip.", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_value = float(loss.item() * accum_steps)
            epoch_loss_sum += loss_value
            epoch_main_sum += loss_parts["main"]
            epoch_aux_sum += loss_parts["aux"]
            epoch_step_count += 1
            global_train_step += 1
            loss_ema = (loss_value if loss_ema is None else
                        ema_decay * loss_ema + (1.0 - ema_decay) * loss_value)

            if (step + 1) % accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip).item())
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                if ema_model is not None:
                    with torch.no_grad():
                        src_state = target_module.state_dict()
                        ema_state = ema_model.state_dict()
                        for key, value in src_state.items():
                            if not torch.is_floating_point(value):
                                ema_state[key].copy_(value.detach().cpu())
                            else:
                                ema_state[key].mul_(weight_ema_decay).add_(value.detach().cpu(), alpha=1.0 - weight_ema_decay)
            else:
                grad_norm = float("nan")

            if is_main and (step + 1) % log_every == 0:
                lrs = [g["lr"] for g in optimizer.param_groups]
                lr_str = "/".join(f"{g}={l:.2e}" for g, l in zip(group_names, lrs))
                if writer is not None:
                    writer.add_scalar("train/loss_step", loss_value, global_train_step)
                    writer.add_scalar("train/loss_main", loss_parts["main"], global_train_step)
                    writer.add_scalar("train/loss_aux", loss_parts["aux"], global_train_step)
                    writer.add_scalar("train/aux_w", loss_parts["aux_w"], global_train_step)
                    writer.add_scalar("train/loss_fut_aux", loss_parts.get("fut_aux", 0), global_train_step)
                    writer.add_scalar("train/grad_norm", grad_norm, global_train_step)
                    for gn, lr in zip(group_names, lrs):
                        writer.add_scalar(f"train/lr_{gn}", lr, global_train_step)
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(
                        loss=f"{loss_value:.4f}",
                        ema=f"{loss_ema:.4f}",
                        m=f"{loss_parts['main']:.3f}",
                        a=f"{loss_parts['aux']:.3f}",
                        aw=f"{loss_parts['aux_w']:.2f}",
                        fa=f"{loss_parts.get('fut_aux', 0):.3f}",
                        mix=mix_info.get("mode", "none")[:4],
                    )
                if not sys.stderr.isatty():
                    print(
                        f"[train] e={epoch} s={step} g={global_train_step} "
                        f"loss={loss_value:.4f} ema={loss_ema:.4f} main={loss_parts['main']:.4f} "
                        f"aux={loss_parts['aux']:.4f} aw={loss_parts['aux_w']:.2f} "
                        f"fut_aux={loss_parts.get('fut_aux', 0):.4f} "
                        f"gnorm={'skip' if grad_norm != grad_norm else f'{grad_norm:.3f}'} mix={mix_info.get('mode')} {lr_str}",
                        flush=True,
                    )

        if is_distributed():
            torch.distributed.barrier()

        # ---- eval per epoch ----
        eval_module = model.module if hasattr(model, "module") else model
        eval_target = eval_module
        if ema_model is not None:
            eval_target = copy.deepcopy(eval_module)
            eval_target.load_state_dict(ema_model.state_dict(), strict=True)
            eval_target.to(device)
            eval_target.eval()
        local_probs, local_labels = collect_probs(
            eval_target, tuned_valid_loader, device, amp_dtype=amp_dtype, show_progress=is_main,
        )
        if is_distributed():
            gathered = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered, (local_probs, local_labels))
        else:
            gathered = [(local_probs, local_labels)]

        should_stop = False
        if is_main:
            valid_probs = np.concatenate([x[0] for x in gathered], axis=0)
            valid_labels = np.concatenate([x[1] for x in gathered], axis=0)
            metrics = compute_tuned_metrics(
                valid_labels, valid_probs, metric_label_names,
                t_min=threshold_min, t_max=threshold_max, t_step=threshold_step,
            )
            metric_value = float(metrics["tuned_macro_f1"])
            avg_train_loss = epoch_loss_sum / max(1, epoch_step_count)
            metrics["epoch"] = int(epoch)
            metrics["train_loss"] = float(avg_train_loss)
            metrics["train_main_loss"] = float(epoch_main_sum / max(1, epoch_step_count))
            metrics["train_aux_loss"] = float(epoch_aux_sum / max(1, epoch_step_count))
            metrics["best_tuned_macro_f1_before_update"] = float(
                best_metric if math.isfinite(best_metric) else -1.0)
            save_json(logs_dir / f"valid_tuned_epoch_{epoch}.json", metrics)
            np.save(intermediate_dir / f"valid_probs_epoch_{epoch}.npy", valid_probs)
            np.save(intermediate_dir / f"valid_labels_epoch_{epoch}.npy", valid_labels)
            save_ckpt(last_path, epoch, best_metric, eval_module, optimizer, scheduler, cfg, bad_epochs, ema_model=ema_model)
            if save_epoch_ckpts:
                save_ckpt(checkpoints_dir / f"epoch_{epoch:02d}.pt",
                          epoch, best_metric, eval_module, optimizer, scheduler, cfg, bad_epochs, ema_model=ema_model)
            improved = metric_value > best_metric
            if improved:
                bad_epochs = 0
                best_metric = metric_value
                save_ckpt(best_path, epoch, best_metric, eval_module, optimizer, scheduler, cfg, bad_epochs, ema_model=ema_model)
            else:
                bad_epochs += 1
                if bad_epochs >= early_stop_patience:
                    should_stop = True
            if writer is not None:
                writer.add_scalar("valid/tuned_macro_f1", metrics["tuned_macro_f1"], epoch)
                writer.add_scalar("valid/macro_f1_at_0_5", metrics["macro_f1"], epoch)
                writer.add_scalar("valid/macro_roc_auc", metrics["macro_roc_auc"], epoch)
                writer.add_scalar("train/loss_epoch", avg_train_loss, epoch)
                for n in metric_label_names:
                    writer.add_scalar(f"valid/{n}_tuned_f1", metrics[f"{n}_tuned_f1"], epoch)
                    writer.add_scalar(f"valid/{n}_threshold", metrics[f"{n}_tuned_threshold"], epoch)
            print(
                f"[Epoch {epoch}] train_loss={avg_train_loss:.6f} main={metrics['train_main_loss']:.4f} "
                f"aux={metrics['train_aux_loss']:.4f} "
                f"macro_f1@0.5={metrics['macro_f1']:.6f} "
                f"tuned_macro_f1={metrics['tuned_macro_f1']:.6f} "
                f"macro_auc={metrics['macro_roc_auc']:.6f} "
                f"best={best_metric:.6f} bad_epochs={bad_epochs} | "
                f"{format_metrics(metrics, metric_label_names)}",
                flush=True,
            )

        if is_distributed():
            flag = torch.tensor([1 if should_stop else 0], device=device, dtype=torch.int64)
            torch.distributed.broadcast(flag, src=0)
            should_stop = bool(flag.item())
        if should_stop:
            if is_main:
                print(f"[early_stop] epoch={epoch} best={best_metric:.6f}", flush=True)
            break

    cleanup_distributed()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()