"""Train a low-correlation speaker-activity TCN branch.

This branch is intentionally separated from the Whisper/Qwen stack.  It learns from
80 ms dual-channel activity sequences: energy, VAD-like masks, overlap/silence,
speaker dominance, ASR activity, short responses and optional weak context state.

Outputs use the same OOF format as tools_oof_valid.py:
  probs, labels, conv_ids, end_idxs  in order c, na, i, bc, t
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tools_audio_state_quick_oof import (  # noqa: E402
    OUT_NAMES,
    OUT_TO_CFG,
    EPS,
    event_labels,
    is_short_ack,
    list_conv_ids,
    load_audio_arrays,
    load_config,
    normalize_text,
    ref_rate_f1,
    sample_grid,
    sorted_save_npz,
    split_conversation_ids,
    tuned_macro_f1,
    write_metrics,
)


@dataclass(frozen=True)
class ActivitySample:
    conv_id: str
    end_idx: int
    label: tuple[int, ...]
    future: tuple[int, ...]


class RunningStats:
    def __init__(self, dim: int) -> None:
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sq = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x64 = np.asarray(x, dtype=np.float64)
        self.count += int(x64.shape[0])
        self.sum += x64.sum(axis=0)
        self.sq += np.square(x64).sum(axis=0)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 0:
            raise RuntimeError("empty feature statistics")
        mean = self.sum / float(self.count)
        var = np.maximum(1e-6, self.sq / float(self.count) - mean * mean)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_text_arrays_segment(text_path: Path, n_chunks: int, chunk_ms: int) -> dict[str, np.ndarray]:
    active = np.zeros((2, n_chunks), dtype=np.float32)
    short = np.zeros((2, n_chunks), dtype=np.float32)
    chars = np.zeros((2, n_chunks), dtype=np.float32)
    starts = np.zeros((2, n_chunks), dtype=np.float32)
    ends = np.zeros((2, n_chunks), dtype=np.float32)
    if not text_path.exists():
        return {"active": active, "short": short, "chars": chars, "starts": starts, "ends": ends}

    with open(text_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    offset_ms = int(float(obj.get("start_ms", 0))) if "start_ms" in obj else 0
    for utt in obj.get("utterances", []):
        ch = int(utt.get("channel_id", 1)) - 1
        if ch not in (0, 1):
            ch = 0
        raw_st = int(float(utt.get("start_ms", utt.get("start", 0))))
        raw_en = int(float(utt.get("end_ms", utt.get("end", raw_st))))
        st = raw_st - offset_ms
        en = raw_en - offset_ms
        if en <= st:
            continue
        s = max(0, min(n_chunks, st // chunk_ms))
        e = max(s + 1, min(n_chunks, int(math.ceil(en / float(chunk_ms)))))
        if s >= n_chunks:
            continue
        text = str(utt.get("text", ""))
        active[ch, s:e] = 1.0
        chars[ch, s:e] += float(len(normalize_text(text))) / max(1, e - s)
        starts[ch, s] += 1.0
        ends[ch, e - 1] += 1.0
        if is_short_ack(text):
            short[ch, s:e] = 1.0
    return {"active": active, "short": short, "chars": chars, "starts": starts, "ends": ends}


def add_pair_features(
    cols: list[np.ndarray],
    names: list[str],
    prefix: str,
    ch0: np.ndarray,
    ch1: np.ndarray,
) -> None:
    ch0 = np.asarray(ch0, dtype=np.float32)
    ch1 = np.asarray(ch1, dtype=np.float32)
    diff = ch0 - ch1
    absdiff = np.abs(diff)
    mx = np.maximum(ch0, ch1)
    mn = np.minimum(ch0, ch1)
    for name, arr in (
        ("ch0", ch0),
        ("ch1", ch1),
        ("diff", diff),
        ("absdiff", absdiff),
        ("max", mx),
        ("min", mn),
    ):
        cols.append(arr.astype(np.float32, copy=False))
        names.append(f"{prefix}_{name}")


def add_binary_state_features(
    cols: list[np.ndarray],
    names: list[str],
    prefix: str,
    ch0: np.ndarray,
    ch1: np.ndarray,
) -> None:
    a0 = np.asarray(ch0).astype(bool)
    a1 = np.asarray(ch1).astype(bool)
    any_active = a0 | a1
    overlap = a0 & a1
    silence = ~any_active
    xor = a0 ^ a1
    ch0_start = a0 & np.concatenate([[False], ~a0[:-1]])
    ch1_start = a1 & np.concatenate([[False], ~a1[:-1]])
    any_start = any_active & np.concatenate([[False], ~any_active[:-1]])
    dominance = np.where(overlap, 2.0, np.where(silence, 3.0, np.where(a0, 0.0, 1.0))) / 3.0
    for name, arr in (
        ("any", any_active),
        ("overlap", overlap),
        ("silence", silence),
        ("xor", xor),
        ("ch0_start", ch0_start),
        ("ch1_start", ch1_start),
        ("any_start", any_start),
        ("dominance", dominance),
    ):
        cols.append(np.asarray(arr, dtype=np.float32))
        names.append(f"{prefix}_{name}")


def make_activity_features(
    seq: np.ndarray | None,
    text_path: Path | None,
    audio_path: Path | None,
    n_chunks: int,
    chunk_ms: int,
    raw_ids: list[int],
    use_state: bool,
    use_text: bool,
    use_audio: bool,
) -> tuple[np.ndarray, list[str]]:
    cols: list[np.ndarray] = []
    names: list[str] = []

    if use_audio and audio_path is not None:
        audio = load_audio_arrays(audio_path, n_chunks, chunk_ms)
        add_pair_features(cols, names, "audio_loge", audio["loge"][0], audio["loge"][1])
        add_pair_features(cols, names, "audio_peak", audio["peak"][0], audio["peak"][1])
        add_binary_state_features(cols, names, "audio_abs", audio["active_abs"][0], audio["active_abs"][1])
        add_binary_state_features(cols, names, "audio_rel", audio["active_rel"][0], audio["active_rel"][1])

    if use_text and text_path is not None:
        text = load_text_arrays_segment(text_path, n_chunks, chunk_ms)
        add_binary_state_features(cols, names, "asr_active", text["active"][0], text["active"][1])
        add_binary_state_features(cols, names, "asr_short", text["short"][0], text["short"][1])
        add_pair_features(cols, names, "asr_chars", text["chars"][0], text["chars"][1])
        add_pair_features(cols, names, "asr_starts", text["starts"][0], text["starts"][1])
        add_pair_features(cols, names, "asr_ends", text["ends"][0], text["ends"][1])

    if use_state:
        if seq is None:
            raise ValueError("use_state requires context labels")
        seq = np.asarray(seq, dtype=np.int64)
        for out_name, raw_id in zip(OUT_NAMES, raw_ids):
            cols.append((seq == int(raw_id)).astype(np.float32))
            names.append(f"ctx_{out_name}_onehot")
        prev = np.concatenate([[seq[0]], seq[:-1]]) if len(seq) else seq
        cols.append((seq != prev).astype(np.float32))
        names.append("ctx_transition")

    if not cols:
        raise ValueError("at least one feature source must be enabled")
    x = np.stack(cols, axis=1).astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    return x, names


def build_samples_for_ids(
    labels_dir: Path,
    conv_ids: Iterable[str],
    raw_ids: list[int],
    context_chunks: int,
    target_chunks: int,
    stride: int,
    horizons: tuple[int, ...],
) -> list[ActivitySample]:
    samples: list[ActivitySample] = []
    for cid in conv_ids:
        seq = np.load(labels_dir / f"{cid}.npy").astype(np.int64)
        ends = sample_grid(len(seq), context_chunks, target_chunks, stride)
        if len(ends) == 0:
            continue
        ys = event_labels(seq, ends, target_chunks, raw_ids)
        for end_idx, y in zip(ends.tolist(), ys.tolist()):
            fut = []
            for h in horizons:
                pos = int(end_idx) + int(h)
                fut.append(int(seq[pos]) if pos < len(seq) else -100)
            samples.append(
                ActivitySample(
                    conv_id=cid,
                    end_idx=int(end_idx),
                    label=tuple(int(v) for v in y),
                    future=tuple(fut),
                )
            )
    return samples


def maybe_limit_samples(samples: list[ActivitySample], limit: int | None, seed: int) -> list[ActivitySample]:
    if limit is None or limit <= 0 or len(samples) <= limit:
        return samples
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), size=int(limit), replace=False)
    idx.sort()
    return [samples[int(i)] for i in idx]


def build_feature_map(
    conv_ids: Iterable[str],
    labels_dir: Path | None,
    text_dir: Path,
    audio_dir: Path,
    raw_ids: list[int],
    chunk_ms: int,
    use_state: bool,
    use_text: bool,
    use_audio: bool,
    tag: str,
) -> tuple[dict[str, np.ndarray], list[str]]:
    features: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    ids = list(conv_ids)
    t0 = time.time()
    for i, cid in enumerate(ids, start=1):
        if labels_dir is not None:
            seq = np.load(labels_dir / f"{cid}.npy").astype(np.int64)
            n_chunks = int(len(seq))
        else:
            seq = np.load(text_dir.parent / "context" / f"{cid}.npy").astype(np.int64)
            n_chunks = int(len(seq))
        x, names = make_activity_features(
            seq=seq,
            text_path=text_dir / f"{cid}.json",
            audio_path=audio_dir / f"{cid}.wav",
            n_chunks=n_chunks,
            chunk_ms=chunk_ms,
            raw_ids=raw_ids,
            use_state=use_state,
            use_text=use_text,
            use_audio=use_audio,
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError(f"feature names mismatch at {cid}")
        features[cid] = x
        if i % 20 == 0 or i == len(ids):
            print(f"[{tag}-features] conv={i}/{len(ids)} elapsed={time.time()-t0:.1f}s", flush=True)
    if feature_names is None:
        raise RuntimeError(f"no features built for {tag}")
    return features, feature_names


def fit_feature_stats(feature_map: dict[str, np.ndarray], conv_ids: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    stats: RunningStats | None = None
    for cid in conv_ids:
        x = feature_map[cid]
        if stats is None:
            stats = RunningStats(x.shape[1])
        stats.update(x)
    if stats is None:
        raise RuntimeError("no train features for stats")
    return stats.finalize()


class ActivityWindowDataset(Dataset):
    def __init__(
        self,
        samples: list[ActivitySample],
        feature_map: dict[str, np.ndarray],
        mean: np.ndarray,
        std: np.ndarray,
        context_chunks: int,
    ) -> None:
        self.samples = samples
        self.feature_map = feature_map
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-4)
        self.context_chunks = int(context_chunks)
        self.rel_time = np.linspace(0.0, 1.0, self.context_chunks, dtype=np.float32)[:, None]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        x_all = self.feature_map[s.conv_id]
        start = int(s.end_idx) - self.context_chunks
        stop = int(s.end_idx)
        if start < 0 or stop > len(x_all):
            raise IndexError(f"invalid sample window {s.conv_id}:{start}:{stop}")
        x = (x_all[start:stop] - self.mean) / self.std
        x = np.clip(x, -8.0, 8.0)
        x = np.concatenate([x.astype(np.float32, copy=False), self.rel_time], axis=1)
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(s.label, dtype=torch.float32),
            "future": torch.tensor(s.future, dtype=torch.long),
            "conv_id": s.conv_id,
            "end_idx": int(s.end_idx),
        }


class ActivityTestDataset(Dataset):
    def __init__(
        self,
        segment_ids: list[str],
        feature_map: dict[str, np.ndarray],
        mean: np.ndarray,
        std: np.ndarray,
        context_chunks: int,
    ) -> None:
        self.segment_ids = segment_ids
        self.feature_map = feature_map
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-4)
        self.context_chunks = int(context_chunks)
        self.rel_time = np.linspace(0.0, 1.0, self.context_chunks, dtype=np.float32)[:, None]

    def __len__(self) -> int:
        return len(self.segment_ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.segment_ids[idx]
        x_all = self.feature_map[sid]
        if len(x_all) < self.context_chunks:
            pad = np.repeat(x_all[:1], self.context_chunks - len(x_all), axis=0) if len(x_all) else np.zeros((self.context_chunks, len(self.mean)), dtype=np.float32)
            x_all = np.concatenate([pad, x_all], axis=0)
        x = x_all[-self.context_chunks:]
        x = (x - self.mean) / self.std
        x = np.clip(x, -8.0, 8.0)
        x = np.concatenate([x.astype(np.float32, copy=False), self.rel_time], axis=1)
        return {"x": torch.from_numpy(x), "segment_id": sid}


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * int(dilation)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=int(dilation), dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)
        return F.gelu(x + residual)


class ActivityTCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
        horizons: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.horizons = tuple(horizons)
        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        dilations = [1, 2, 4, 8, 16, 32]
        self.blocks = nn.ModuleList(
            TCNBlock(hidden, dilations[i % len(dilations)], dropout) for i in range(int(layers))
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden, max(16, hidden // 2)),
            nn.Tanh(),
            nn.Linear(max(16, hidden // 2), 1),
        )
        pool_dim = hidden * 4
        self.head = nn.Sequential(
            nn.Linear(pool_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 5),
        )
        self.future_head = nn.Linear(pool_dim, len(self.horizons) * 5) if self.horizons else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.input(x)  # [B,T,H]
        h = h.transpose(1, 2).contiguous()
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2).contiguous()
        score = self.attn(h).squeeze(-1)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        attn_pool = (h * weight).sum(dim=1)
        tail = h[:, -50:, :]
        tail_mean = tail.mean(dim=1)
        tail_max = tail.amax(dim=1)
        last = h[:, -1, :]
        pooled = torch.cat([last, tail_mean, tail_max, attn_pool], dim=-1)
        logits = self.head(pooled)
        if self.future_head is None:
            return logits, None
        future_logits = self.future_head(pooled).view(x.shape[0], len(self.horizons), 5)
        return logits, future_logits


def compute_pos_weight(samples: list[ActivitySample], power: float, cap: float) -> torch.Tensor:
    y = np.asarray([s.label for s in samples], dtype=np.float32)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    w = np.power(np.maximum(1.0, neg / np.maximum(1.0, pos)), float(power))
    w = np.minimum(float(cap), w)
    return torch.tensor(w, dtype=torch.float32)


def batch_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["x"].to(device, non_blocking=True).float(),
        batch["y"].to(device, non_blocking=True).float(),
        batch["future"].to(device, non_blocking=True).long(),
    )


def train_epoch(
    model: ActivityTCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce: nn.Module,
    device: torch.device,
    aux_weight: float,
    grad_clip: float,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        x, y, future = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, future_logits = model(x)
        loss = bce(logits, y)
        if future_logits is not None and aux_weight > 0:
            aux = F.cross_entropy(
                future_logits.reshape(-1, 5),
                future.reshape(-1),
                ignore_index=-100,
                label_smoothing=0.05,
            )
            loss = loss + float(aux_weight) * aux
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        total += float(loss.detach().cpu()) * len(y)
        count += len(y)
    return total / max(1, count)


@torch.no_grad()
def predict_valid(model: ActivityTCN, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    labels = []
    conv_ids = []
    end_idxs = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True).float()
        logits, _ = model(x)
        probs.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
        labels.append(batch["y"].numpy().astype(np.int8))
        conv_ids.extend([str(v) for v in batch["conv_id"]])
        end_idxs.extend([int(v) for v in batch["end_idx"]])
    return (
        np.concatenate(probs, axis=0),
        np.concatenate(labels, axis=0),
        np.asarray(conv_ids, dtype=str),
        np.asarray(end_idxs, dtype=np.int64),
    )


@torch.no_grad()
def predict_test(model: ActivityTCN, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, list[str]]:
    model.eval()
    probs = []
    ids: list[str] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True).float()
        logits, _ = model(x)
        probs.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
        ids.extend([str(v) for v in batch["segment_id"]])
    return np.concatenate(probs, axis=0), ids


def summarize_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    aucs = []
    for j in range(labels.shape[1]):
        y = labels[:, j]
        aucs.append(float(roc_auc_score(y, probs[:, j])) if 0 < y.sum() < len(y) else None)
    tuned, thrs = tuned_macro_f1(probs, labels)
    rr = ref_rate_f1(probs, labels)
    return {
        "auc": dict(zip(OUT_NAMES, aucs)),
        "tuned_f1": dict(zip(OUT_NAMES, tuned)),
        "tuned_thresholds": dict(zip(OUT_NAMES, thrs)),
        "tuned_macro_f1": float(np.mean(tuned)),
        "ref_rate_f1": dict(zip(OUT_NAMES, rr)),
        "ref_rate_macro_f1": float(np.mean(rr)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="outputs/oof/sota_e12_ckpt_config.yaml")
    p.add_argument("--output_root", default="outputs_activity_tcn")
    p.add_argument("--out", default="outputs/oof/activity_tcn_grid25.npz")
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight_decay", type=float, default=1.0e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--aux_weight", type=float, default=0.2)
    p.add_argument("--horizons", default="5,10,15,25")
    p.add_argument("--pos_weight_power", type=float, default=0.5)
    p.add_argument("--max_pos_weight", type=float, default=16.0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_valid_samples", type=int, default=None)
    p.add_argument("--max_train_convs", type=int, default=None)
    p.add_argument("--max_valid_convs", type=int, default=None)
    p.add_argument("--use_state", action="store_true")
    p.add_argument("--no_text", action="store_true")
    p.add_argument("--no_audio", action="store_true")
    p.add_argument("--train_all", action="store_true")
    p.add_argument("--test_root", default=None)
    p.add_argument("--output_test_probs", default=None)
    p.add_argument("--output_test_ids", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    return p.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else HERE / p


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    cfg = load_config(config_path)
    seed = int(args.seed if args.seed is not None else cfg.get("split", {}).get("seed", cfg.get("seed", 42)))
    set_seed(seed)

    output_root = resolve_path(args.output_root)
    ckpt_dir = output_root / "checkpoints"
    log_dir = output_root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = cfg["paths"]
    labels_dir = Path(paths["train_labels_dir"])
    train_text_dir = Path(paths["train_text_dir"])
    train_audio_dir = Path(paths["train_audio_dir"])
    chunk_ms = int(cfg["chunk_ms"])
    context_chunks = int(cfg["context_chunks"])
    target_chunks = int(cfg["target_chunks"])
    raw_ids = [int(cfg["labels"][OUT_TO_CFG[name]]) for name in OUT_NAMES]
    horizons = tuple(int(x) for x in str(args.horizons).split(",") if str(x).strip())
    use_text = not bool(args.no_text)
    use_audio = not bool(args.no_audio)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    print(
        f"[setup] device={device} stride={args.stride} use_audio={use_audio} "
        f"use_text={use_text} use_state={args.use_state} horizons={horizons}",
        flush=True,
    )

    all_ids = list_conv_ids(labels_dir)
    if args.train_all:
        train_ids = all_ids
        valid_ids: list[str] = []
    else:
        train_ids, valid_ids = split_conversation_ids(
            all_ids,
            valid_ratio=float(cfg.get("split", {}).get("valid_ratio", 0.1)),
            seed=seed,
        )
    if args.max_train_convs is not None and int(args.max_train_convs) > 0:
        train_ids = train_ids[: int(args.max_train_convs)]
    if args.max_valid_convs is not None and int(args.max_valid_convs) > 0:
        valid_ids = valid_ids[: int(args.max_valid_convs)]
    print(f"[split] train_convs={len(train_ids)} valid_convs={len(valid_ids)}", flush=True)

    feature_ids = sorted(set(train_ids) | set(valid_ids))
    feature_map, feature_names = build_feature_map(
        feature_ids,
        labels_dir,
        train_text_dir,
        train_audio_dir,
        raw_ids,
        chunk_ms,
        use_state=bool(args.use_state),
        use_text=use_text,
        use_audio=use_audio,
        tag="trainvalid",
    )
    mean, std = fit_feature_stats(feature_map, train_ids)
    in_dim = len(feature_names) + 1
    print(f"[features] base_dim={len(feature_names)} model_in_dim={in_dim}", flush=True)

    train_samples = build_samples_for_ids(
        labels_dir, train_ids, raw_ids, context_chunks, target_chunks, int(args.stride), horizons
    )
    valid_samples = build_samples_for_ids(
        labels_dir, valid_ids, raw_ids, context_chunks, target_chunks, int(args.stride), horizons
    ) if valid_ids else []
    train_samples = maybe_limit_samples(train_samples, args.max_train_samples, seed)
    valid_samples = maybe_limit_samples(valid_samples, args.max_valid_samples, seed + 1)
    print(f"[samples] train={len(train_samples)} valid={len(valid_samples)}", flush=True)

    train_ds = ActivityWindowDataset(train_samples, feature_map, mean, std, context_chunks)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    valid_loader = None
    if valid_samples:
        valid_ds = ActivityWindowDataset(valid_samples, feature_map, mean, std, context_chunks)
        valid_loader = DataLoader(
            valid_ds,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )

    model = ActivityTCN(in_dim, int(args.hidden), int(args.layers), float(args.dropout), horizons).to(device)
    pos_weight = compute_pos_weight(train_samples, float(args.pos_weight_power), float(args.max_pos_weight)).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))
    print(f"[loss] pos_weight={pos_weight.detach().cpu().numpy().round(3).tolist()}", flush=True)

    best_metric = -1.0
    best_state = None
    history = []
    for epoch in range(int(args.epochs)):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer, bce, device, float(args.aux_weight), float(args.grad_clip))
        scheduler.step()
        row = {"epoch": epoch, "loss": float(loss), "lr": float(scheduler.get_last_lr()[0])}
        if valid_loader is not None:
            probs, labels, conv_valid, end_valid = predict_valid(model, valid_loader, device)
            metric = summarize_metrics(probs, labels)
            row.update(metric)
            score = float(metric["tuned_macro_f1"])
            if score > best_metric:
                best_metric = score
                best_state = copy.deepcopy(model.state_dict())
                sorted_save_npz(out_path, probs, labels, conv_valid, end_valid)
                write_metrics(
                    out_path.with_suffix(".metrics.json"),
                    probs,
                    labels,
                    {
                        "config": str(config_path),
                        "out": str(out_path),
                        "epoch": int(epoch),
                        "stride": int(args.stride),
                        "train_convs": len(train_ids),
                        "valid_convs": len(valid_ids),
                        "train_samples": len(train_samples),
                        "valid_samples": len(valid_samples),
                        "feature_count": len(feature_names),
                        "feature_names": feature_names,
                        "feature_sources": {"audio": use_audio, "text": use_text, "state": bool(args.use_state)},
                        "model": "activity_tcn",
                        "note": "low-correlation speaker-activity sequence branch; weak-label valid only",
                    },
                )
            print(
                f"[epoch {epoch:02d}] loss={loss:.5f} tuned={metric['tuned_macro_f1']:.5f} "
                f"ref={metric['ref_rate_macro_f1']:.5f} best={best_metric:.5f} elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
        else:
            best_metric = float("nan")
            best_state = copy.deepcopy(model.state_dict())
            print(f"[epoch {epoch:02d}] loss={loss:.5f} elapsed={time.time()-t0:.1f}s", flush=True)
        history.append(row)
        (log_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = {
        "model": model.state_dict(),
        "mean": mean,
        "std": std,
        "feature_names": feature_names,
        "raw_ids": raw_ids,
        "out_names": OUT_NAMES,
        "args": vars(args),
        "config_path": str(config_path),
        "best_metric": best_metric,
    }
    ckpt_path = ckpt_dir / "best.pt"
    torch.save(ckpt, ckpt_path)
    print(f"[saved] checkpoint={ckpt_path}", flush=True)
    if valid_loader is not None:
        print(f"[saved] oof={out_path} metrics={out_path.with_suffix('.metrics.json')}", flush=True)

    if args.test_root:
        test_root = resolve_path(args.test_root)
        test_context_dir = test_root / "context"
        test_text_dir = test_root / "text"
        test_audio_dir = test_root / "audio"
        segment_ids = sorted(p.stem for p in test_context_dir.glob("*.npy"))
        test_feature_map, test_feature_names = build_feature_map(
            segment_ids,
            labels_dir=None,
            text_dir=test_text_dir,
            audio_dir=test_audio_dir,
            raw_ids=raw_ids,
            chunk_ms=chunk_ms,
            use_state=bool(args.use_state),
            use_text=use_text,
            use_audio=use_audio,
            tag="test",
        )
        if test_feature_names != feature_names:
            raise ValueError("test feature names mismatch")
        test_ds = ActivityTestDataset(segment_ids, test_feature_map, mean, std, context_chunks)
        test_loader = DataLoader(
            test_ds,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )
        test_probs, test_ids = predict_test(model, test_loader, device)
        order = np.argsort(np.asarray(test_ids))
        test_probs = test_probs[order].astype(np.float32)
        test_ids = [test_ids[int(i)] for i in order]
        probs_path = resolve_path(args.output_test_probs or "outputs/test_probs/activity_tcn_test_probs.npy")
        ids_path = resolve_path(args.output_test_ids or "outputs/test_probs/activity_tcn_test_ids.txt")
        probs_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(probs_path, test_probs)
        ids_path.write_text("".join(f"{sid}\n" for sid in test_ids), encoding="utf-8")
        meta = {
            "checkpoint": str(ckpt_path),
            "test_root": str(test_root),
            "shape": list(test_probs.shape),
            "feature_sources": {"audio": use_audio, "text": use_text, "state": bool(args.use_state)},
            "prob_mean": dict(zip(OUT_NAMES, test_probs.mean(axis=0).astype(float).tolist())),
        }
        probs_path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] test_probs={probs_path} ids={ids_path} shape={test_probs.shape}", flush=True)


if __name__ == "__main__":
    main()