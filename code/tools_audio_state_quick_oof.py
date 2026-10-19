"""Quick audio/state feature OOF probe for Phase B turn-taking.

This script deliberately stays outside the main training stack. It builds low-cost
state, ASR-timestamp, and raw-audio energy features, trains one lightweight binary
ranker per output label, and emits the same grid25 OOF npz format as
`tools_oof_valid.py`:

  probs, labels, conv_ids, end_idxs  with column order c, na, i, bc, t

The result is only a weak-label valid probe. Online use should be restricted to
small T-only calibration unless later evidence says otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
try:
    import soundfile as sf
except ModuleNotFoundError:
    sf = None
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

OUT_NAMES = ["c", "na", "i", "bc", "t"]
OUT_TO_CFG = {"c": "C", "na": "NA", "i": "I", "bc": "BC", "t": "T"}
WINDOWS = (5, 10, 25, 50, 125, 375)
DIST_CAP = 375
EPS = 1e-8


@dataclass
class ConstantModel:
    value: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p1 = np.full(len(x), self.value, dtype=np.float64)
        return np.stack([1.0 - p1, p1], axis=1)


class FeatureBag:
    def __init__(self, n: int) -> None:
        self.n = int(n)
        self.names: list[str] = []
        self.cols: list[np.ndarray] = []

    def add(self, name: str, values) -> None:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full(self.n, float(arr), dtype=np.float32)
        if arr.shape != (self.n,):
            raise ValueError(f"feature {name} shape={arr.shape}, expected {(self.n,)}")
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        self.names.append(name)
        self.cols.append(arr.astype(np.float32, copy=False))

    def matrix(self) -> np.ndarray:
        if not self.cols:
            return np.zeros((self.n, 0), dtype=np.float32)
        return np.stack(self.cols, axis=1).astype(np.float32, copy=False)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_conv_ids(labels_dir: Path) -> list[str]:
    return sorted(p.stem for p in labels_dir.glob("*.npy"))


def split_conversation_ids(conv_ids: Iterable[str], valid_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    ids = list(conv_ids)
    random.Random(seed).shuffle(ids)
    valid_size = max(1, int(len(ids) * valid_ratio))
    valid_ids = sorted(ids[:valid_size])
    train_ids = sorted(ids[valid_size:])
    return train_ids, valid_ids


def prefix_sum(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(x.astype(np.float64))])


def bounds(end_idxs: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e = end_idxs.astype(np.int64)
    s = np.maximum(0, e - int(width))
    d = np.maximum(1, e - s).astype(np.float64)
    return s, e, d


def win_sum(x: np.ndarray, end_idxs: np.ndarray, width: int) -> np.ndarray:
    s, e, _ = bounds(end_idxs, width)
    p = prefix_sum(x)
    return p[e] - p[s]


def win_mean(x: np.ndarray, end_idxs: np.ndarray, width: int) -> np.ndarray:
    s, e, d = bounds(end_idxs, width)
    p = prefix_sum(x)
    return (p[e] - p[s]) / d


def win_std(x: np.ndarray, end_idxs: np.ndarray, width: int) -> np.ndarray:
    s, e, d = bounds(end_idxs, width)
    p1 = prefix_sum(x)
    p2 = prefix_sum(np.asarray(x, dtype=np.float64) ** 2)
    m = (p1[e] - p1[s]) / d
    v = np.maximum(0.0, (p2[e] - p2[s]) / d - m * m)
    return np.sqrt(v)


def transition_count(values: np.ndarray, end_idxs: np.ndarray, width: int) -> np.ndarray:
    v = np.asarray(values)
    if len(v) <= 1:
        return np.zeros(len(end_idxs), dtype=np.float64)
    change = (v[1:] != v[:-1]).astype(np.float64)
    pref = np.concatenate([[0.0], np.cumsum(change)])  # len == len(v)
    s, e, _ = bounds(end_idxs, width)
    e2 = np.maximum(s, e - 1)
    return pref[e2] - pref[s]


def rising_count(mask: np.ndarray, end_idxs: np.ndarray, width: int) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    if len(m) <= 1:
        return np.zeros(len(end_idxs), dtype=np.float64)
    rise = (m[1:] & ~m[:-1]).astype(np.float64)
    pref = np.concatenate([[0.0], np.cumsum(rise)])
    s, e, _ = bounds(end_idxs, width)
    e2 = np.maximum(s, e - 1)
    return pref[e2] - pref[s]


def last_distance(mask: np.ndarray, end_idxs: np.ndarray, cap: int = DIST_CAP) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    if len(m) == 0:
        return np.ones(len(end_idxs), dtype=np.float64)
    pos = np.where(m, np.arange(len(m)), -10**9)
    last = np.maximum.accumulate(pos)
    idx = np.maximum(0, end_idxs.astype(np.int64) - 1)
    d = idx - last[idx]
    d = np.where(last[idx] < 0, cap, d)
    return np.minimum(d, cap).astype(np.float64) / float(cap)


def run_length_at_end(values: np.ndarray, end_idxs: np.ndarray, cap: int = DIST_CAP) -> np.ndarray:
    v = np.asarray(values)
    n = len(v)
    if n == 0:
        return np.zeros(len(end_idxs), dtype=np.float64)
    run_start = np.zeros(n, dtype=np.int64)
    cur = 0
    for i in range(1, n):
        if v[i] != v[i - 1]:
            cur = i
        run_start[i] = cur
    idx = np.maximum(0, end_idxs.astype(np.int64) - 1)
    rl = idx - run_start[idx] + 1
    return np.minimum(rl, cap).astype(np.float64) / float(cap)


def add_state_features(bag: FeatureBag, prefix: str, seq: np.ndarray, end_idxs: np.ndarray, raw_ids: list[int]) -> None:
    seq = np.asarray(seq, dtype=np.int64)
    last_idx = np.maximum(0, end_idxs.astype(np.int64) - 1)
    last_vals = seq[last_idx]
    bag.add(f"{prefix}_last_id_scaled", last_vals / 4.0)
    bag.add(f"{prefix}_last_run", run_length_at_end(seq, end_idxs))

    rates_by_window: dict[int, list[np.ndarray]] = {}
    for width in WINDOWS:
        rates = []
        for out_name, raw_id in zip(OUT_NAMES, raw_ids):
            m = (seq == raw_id)
            r = win_mean(m.astype(np.float32), end_idxs, width)
            rates.append(r)
            bag.add(f"{prefix}_{out_name}_rate_w{width}", r)
        rate_mat = np.stack(rates, axis=1)
        ent = -np.sum(rate_mat * np.log(np.clip(rate_mat, EPS, 1.0)), axis=1) / np.log(len(raw_ids))
        bag.add(f"{prefix}_entropy_w{width}", ent)
        bag.add(f"{prefix}_transition_rate_w{width}", transition_count(seq, end_idxs, width) / max(1, width - 1))
        rates_by_window[width] = rates

    for out_name, raw_id in zip(OUT_NAMES, raw_ids):
        m = (seq == raw_id)
        bag.add(f"{prefix}_{out_name}_last_onehot", (last_vals == raw_id).astype(np.float32))
        bag.add(f"{prefix}_{out_name}_last_dist", last_distance(m, end_idxs))
        bag.add(f"{prefix}_{out_name}_last_run", run_length_at_end(m, end_idxs))
        if 25 in rates_by_window and 125 in rates_by_window:
            j = OUT_NAMES.index(out_name)
            bag.add(f"{prefix}_{out_name}_rate_delta_25_125", rates_by_window[25][j] - rates_by_window[125][j])


def add_numeric_pair_features(bag: FeatureBag, prefix: str, x0: np.ndarray, x1: np.ndarray, end_idxs: np.ndarray) -> None:
    x0 = np.asarray(x0, dtype=np.float32)
    x1 = np.asarray(x1, dtype=np.float32)
    diff = x0 - x1
    absdiff = np.abs(diff)
    mx = np.maximum(x0, x1)
    mn = np.minimum(x0, x1)

    last_idx = np.maximum(0, end_idxs.astype(np.int64) - 1)
    bag.add(f"{prefix}_ch0_last", x0[last_idx])
    bag.add(f"{prefix}_ch1_last", x1[last_idx])
    bag.add(f"{prefix}_diff_last", diff[last_idx])
    bag.add(f"{prefix}_absdiff_last", absdiff[last_idx])
    bag.add(f"{prefix}_max_last", mx[last_idx])

    for width in WINDOWS:
        bag.add(f"{prefix}_ch0_mean_w{width}", win_mean(x0, end_idxs, width))
        bag.add(f"{prefix}_ch1_mean_w{width}", win_mean(x1, end_idxs, width))
        bag.add(f"{prefix}_ch0_std_w{width}", win_std(x0, end_idxs, width))
        bag.add(f"{prefix}_ch1_std_w{width}", win_std(x1, end_idxs, width))
        bag.add(f"{prefix}_diff_mean_w{width}", win_mean(diff, end_idxs, width))
        bag.add(f"{prefix}_absdiff_mean_w{width}", win_mean(absdiff, end_idxs, width))
        bag.add(f"{prefix}_max_mean_w{width}", win_mean(mx, end_idxs, width))
        bag.add(f"{prefix}_min_mean_w{width}", win_mean(mn, end_idxs, width))

    bag.add(f"{prefix}_slope_5_25", win_mean(mx, end_idxs, 5) - win_mean(mx, end_idxs, 25))
    bag.add(f"{prefix}_slope_25_125", win_mean(mx, end_idxs, 25) - win_mean(mx, end_idxs, 125))


def add_binary_pair_features(bag: FeatureBag, prefix: str, a0: np.ndarray, a1: np.ndarray, end_idxs: np.ndarray) -> None:
    a0 = np.asarray(a0).astype(bool)
    a1 = np.asarray(a1).astype(bool)
    any_active = a0 | a1
    overlap = a0 & a1
    silence = ~any_active
    xor = a0 ^ a1
    dom = np.where(overlap, 2, np.where(silence, 3, np.where(a0, 0, 1))).astype(np.int8)

    bag.add(f"{prefix}_any_last", any_active[np.maximum(0, end_idxs - 1)].astype(np.float32))
    bag.add(f"{prefix}_overlap_last", overlap[np.maximum(0, end_idxs - 1)].astype(np.float32))
    bag.add(f"{prefix}_silence_last", silence[np.maximum(0, end_idxs - 1)].astype(np.float32))
    bag.add(f"{prefix}_any_last_dist", last_distance(any_active, end_idxs))
    bag.add(f"{prefix}_silence_last_dist", last_distance(silence, end_idxs))
    bag.add(f"{prefix}_overlap_last_dist", last_distance(overlap, end_idxs))
    bag.add(f"{prefix}_any_run", run_length_at_end(any_active, end_idxs))
    bag.add(f"{prefix}_silence_run", run_length_at_end(silence, end_idxs))
    bag.add(f"{prefix}_overlap_run", run_length_at_end(overlap, end_idxs))

    for width in WINDOWS:
        bag.add(f"{prefix}_ch0_rate_w{width}", win_mean(a0.astype(np.float32), end_idxs, width))
        bag.add(f"{prefix}_ch1_rate_w{width}", win_mean(a1.astype(np.float32), end_idxs, width))
        bag.add(f"{prefix}_any_rate_w{width}", win_mean(any_active.astype(np.float32), end_idxs, width))
        bag.add(f"{prefix}_overlap_rate_w{width}", win_mean(overlap.astype(np.float32), end_idxs, width))
        bag.add(f"{prefix}_silence_rate_w{width}", win_mean(silence.astype(np.float32), end_idxs, width))
        bag.add(f"{prefix}_xor_rate_w{width}", win_mean(xor.astype(np.float32), end_idxs, width))
        bag.add(f"{prefix}_dom_switch_rate_w{width}", transition_count(dom, end_idxs, width) / max(1, width - 1))
        bag.add(f"{prefix}_any_rise_count_w{width}", rising_count(any_active, end_idxs, width) / max(1, width))
        bag.add(f"{prefix}_overlap_rise_count_w{width}", rising_count(overlap, end_idxs, width) / max(1, width))


def normalize_text(text: str) -> str:
    drop = " ，。！？!?、,.；;：:（）()[]【】\"'“”‘’\n\t "
    return "".join(ch for ch in str(text).strip() if ch not in drop)


def is_short_ack(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    ack_tokens = ("嗯", "哦", "噢", "啊", "呀", "诶", "哎", "对", "是", "好", "行", "可以", "没错", "哈哈", "呵呵")
    return len(t) <= 4 or any(tok in t for tok in ack_tokens)


def load_text_arrays(text_path: Path, n_chunks: int, chunk_ms: int) -> dict[str, np.ndarray]:
    active = np.zeros((2, n_chunks), dtype=np.float32)
    short = np.zeros((2, n_chunks), dtype=np.float32)
    chars = np.zeros((2, n_chunks), dtype=np.float32)
    starts = np.zeros((2, n_chunks), dtype=np.float32)
    ends = np.zeros((2, n_chunks), dtype=np.float32)
    if not text_path.exists():
        return {"active": active, "short": short, "chars": chars, "starts": starts, "ends": ends}

    with open(text_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    for utt in obj.get("utterances", []):
        ch = int(utt.get("channel_id", 1)) - 1
        if ch not in (0, 1):
            ch = 0
        st = int(float(utt.get("start_ms", utt.get("start", 0))))
        en = int(float(utt.get("end_ms", utt.get("end", st))))
        if en <= st:
            continue
        s = max(0, min(n_chunks, st // chunk_ms))
        e = max(s + 1, min(n_chunks, int(np.ceil(en / float(chunk_ms)))))
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


def read_wav_float32(audio_path: Path) -> tuple[np.ndarray, int]:
    if sf is not None:
        wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        return wav.astype(np.float32, copy=False), int(sr)

    with wave.open(str(audio_path), "rb") as f:
        channels = int(f.getnchannels())
        sample_width = int(f.getsampwidth())
        sr = int(f.getframerate())
        raw = f.readframes(int(f.getnframes()))

    if sample_width == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported wav sample_width={sample_width} path={audio_path}")

    if channels <= 0:
        raise ValueError(f"invalid wav channels={channels} path={audio_path}")
    if len(arr) % channels:
        arr = arr[: len(arr) - (len(arr) % channels)]
    return arr.reshape(-1, channels).astype(np.float32, copy=False), sr


def load_audio_arrays(audio_path: Path, n_chunks: int, chunk_ms: int) -> dict[str, np.ndarray]:
    loge = np.full((2, n_chunks), -8.0, dtype=np.float32)
    peak = np.full((2, n_chunks), -8.0, dtype=np.float32)
    active_abs = np.zeros((2, n_chunks), dtype=np.float32)
    active_rel = np.zeros((2, n_chunks), dtype=np.float32)
    if not audio_path.exists():
        return {"loge": loge, "peak": peak, "active_abs": active_abs, "active_rel": active_rel}

    wav, sr = read_wav_float32(audio_path)
    if wav.shape[1] == 1:
        wav = np.repeat(wav, 2, axis=1)
    elif wav.shape[1] > 2:
        wav = wav[:, :2]
    spc = max(1, int(round(float(sr) * chunk_ms / 1000.0)))
    usable = min(n_chunks, wav.shape[0] // spc)
    if usable <= 0:
        return {"loge": loge, "peak": peak, "active_abs": active_abs, "active_rel": active_rel}

    chunks = wav[: usable * spc].reshape(usable, spc, 2)
    rms = np.sqrt(np.mean(chunks.astype(np.float64) ** 2, axis=1) + EPS).astype(np.float32)
    pk = np.max(np.abs(chunks), axis=1).astype(np.float32)
    loge[:, :usable] = np.log10(rms.T + EPS)
    peak[:, :usable] = np.log10(pk.T + EPS)

    valid_loge = loge[:, :usable]
    active_abs[:, :usable] = (valid_loge > -3.0).astype(np.float32)
    for ch in (0, 1):
        thr = float(np.quantile(valid_loge[ch], 0.60))
        active_rel[ch, :usable] = (valid_loge[ch] > thr).astype(np.float32)
    return {"loge": loge, "peak": peak, "active_abs": active_abs, "active_rel": active_rel}


def add_text_features(bag: FeatureBag, text_path: Path, n_chunks: int, chunk_ms: int, end_idxs: np.ndarray) -> None:
    arr = load_text_arrays(text_path, n_chunks, chunk_ms)
    add_binary_pair_features(bag, "asr_active", arr["active"][0], arr["active"][1], end_idxs)
    add_binary_pair_features(bag, "asr_short", arr["short"][0], arr["short"][1], end_idxs)
    add_numeric_pair_features(bag, "asr_chars", arr["chars"][0], arr["chars"][1], end_idxs)
    add_numeric_pair_features(bag, "asr_starts", arr["starts"][0], arr["starts"][1], end_idxs)
    add_numeric_pair_features(bag, "asr_ends", arr["ends"][0], arr["ends"][1], end_idxs)


def add_audio_features(bag: FeatureBag, audio_path: Path, n_chunks: int, chunk_ms: int, end_idxs: np.ndarray) -> None:
    arr = load_audio_arrays(audio_path, n_chunks, chunk_ms)
    add_numeric_pair_features(bag, "audio_loge", arr["loge"][0], arr["loge"][1], end_idxs)
    add_numeric_pair_features(bag, "audio_peak", arr["peak"][0], arr["peak"][1], end_idxs)
    add_binary_pair_features(bag, "audio_abs", arr["active_abs"][0], arr["active_abs"][1], end_idxs)
    add_binary_pair_features(bag, "audio_rel", arr["active_rel"][0], arr["active_rel"][1], end_idxs)


def event_labels(seq: np.ndarray, end_idxs: np.ndarray, target_chunks: int, raw_ids: list[int]) -> np.ndarray:
    y = np.zeros((len(end_idxs), len(raw_ids)), dtype=np.int8)
    for i, e in enumerate(end_idxs.astype(np.int64)):
        fut = seq[e : e + target_chunks]
        for j, rid in enumerate(raw_ids):
            y[i, j] = int(np.any(fut == rid))
    return y


def sample_grid(seq_len: int, context_chunks: int, target_chunks: int, stride: int) -> np.ndarray:
    max_end = seq_len - target_chunks
    if max_end < context_chunks:
        return np.zeros(0, dtype=np.int64)
    return np.arange(context_chunks, max_end + 1, stride, dtype=np.int64)


def conv_features(
    seq: np.ndarray,
    end_idxs: np.ndarray,
    raw_ids: list[int],
    chunk_ms: int,
    text_path: Path | None,
    audio_path: Path | None,
    use_text: bool,
    use_audio: bool,
) -> tuple[np.ndarray, list[str]]:
    bag = FeatureBag(len(end_idxs))
    add_state_features(bag, "ctx", seq, end_idxs, raw_ids)
    if use_text and text_path is not None:
        add_text_features(bag, text_path, len(seq), chunk_ms, end_idxs)
    if use_audio and audio_path is not None:
        add_audio_features(bag, audio_path, len(seq), chunk_ms, end_idxs)
    return bag.matrix(), bag.names


def collect_one_conv(
    idx: int,
    cid: str,
    labels_dir: Path,
    text_dir: Path,
    audio_dir: Path,
    raw_ids: list[int],
    context_chunks: int,
    target_chunks: int,
    stride: int,
    chunk_ms: int,
    use_text: bool,
    use_audio: bool,
) -> tuple[int, str, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str] | None]:
    seq = np.load(labels_dir / f"{cid}.npy").astype(np.int64)
    ends = sample_grid(len(seq), context_chunks, target_chunks, stride)
    if len(ends) == 0:
        return idx, cid, None, None, None, None, None
    x, names = conv_features(
        seq=seq,
        end_idxs=ends,
        raw_ids=raw_ids,
        chunk_ms=chunk_ms,
        text_path=text_dir / f"{cid}.json",
        audio_path=audio_dir / f"{cid}.wav",
        use_text=use_text,
        use_audio=use_audio,
    )
    y = event_labels(seq, ends, target_chunks, raw_ids)
    return (
        idx,
        cid,
        x.astype(np.float32, copy=False),
        y.astype(np.int8, copy=False),
        np.array([cid] * len(ends)),
        ends.astype(np.int64, copy=False),
        names,
    )


def collect_split(
    conv_ids: list[str],
    labels_dir: Path,
    text_dir: Path,
    audio_dir: Path,
    raw_ids: list[int],
    context_chunks: int,
    target_chunks: int,
    stride: int,
    chunk_ms: int,
    use_text: bool,
    use_audio: bool,
    tag: str,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    all_conv: list[np.ndarray] = []
    all_end: list[np.ndarray] = []
    feature_names: list[str] | None = None
    t0 = time.time()

    def append_result(
        result: tuple[int, str, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str] | None]
    ) -> None:
        nonlocal feature_names
        _, cid, x, y, conv_arr, end_arr, names = result
        if x is None or y is None or conv_arr is None or end_arr is None or names is None:
            return
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError(f"feature names mismatch at {cid}")
        xs.append(x)
        ys.append(y)
        all_conv.append(conv_arr)
        all_end.append(end_arr)

    actual_workers = max(1, min(int(workers), len(conv_ids)))
    if actual_workers == 1:
        for idx, cid in enumerate(conv_ids, start=1):
            append_result(
                collect_one_conv(
                    idx,
                    cid,
                    labels_dir,
                    text_dir,
                    audio_dir,
                    raw_ids,
                    context_chunks,
                    target_chunks,
                    stride,
                    chunk_ms,
                    use_text,
                    use_audio,
                )
            )
            if idx % 20 == 0 or idx == len(conv_ids):
                n = sum(len(v) for v in all_end)
                print(f"[{tag}] conv={idx}/{len(conv_ids)} samples={n} elapsed={time.time()-t0:.1f}s", flush=True)
    else:
        print(f"[{tag}] workers={actual_workers}", flush=True)
        results = []
        samples = 0
        with ProcessPoolExecutor(max_workers=actual_workers) as ex:
            futures = [
                ex.submit(
                    collect_one_conv,
                    idx,
                    cid,
                    labels_dir,
                    text_dir,
                    audio_dir,
                    raw_ids,
                    context_chunks,
                    target_chunks,
                    stride,
                    chunk_ms,
                    use_text,
                    use_audio,
                )
                for idx, cid in enumerate(conv_ids, start=1)
            ]
            for done, fut in enumerate(as_completed(futures), start=1):
                result = fut.result()
                results.append(result)
                if result[5] is not None:
                    samples += len(result[5])
                if done % 20 == 0 or done == len(futures):
                    print(
                        f"[{tag}] conv={done}/{len(conv_ids)} samples={samples} elapsed={time.time()-t0:.1f}s",
                        flush=True,
                    )
        for result in sorted(results, key=lambda item: item[0]):
            append_result(result)

    if not xs:
        raise RuntimeError(f"no samples collected for {tag}")
    return (
        np.concatenate(xs, axis=0).astype(np.float32),
        np.concatenate(ys, axis=0).astype(np.int8),
        np.concatenate(all_conv, axis=0),
        np.concatenate(all_end, axis=0).astype(np.int64),
        feature_names or [],
    )


def maybe_subsample(x: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples <= 0 or len(x) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(x), size=max_samples, replace=False)
    keep.sort()
    return x[keep], y[keep]


def train_one_model(
    j: int,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    max_iter: int,
    pos_weight_power: float,
    max_pos_weight: float,
) -> tuple[int, object, str]:
    name = OUT_NAMES[j]
    yy = y[:, j].astype(np.int64)
    pos = int(yy.sum())
    neg = int(len(yy) - pos)
    if pos == 0 or neg == 0:
        value = float(pos / max(1, len(yy)))
        return j, ConstantModel(value), f"[fit] {name}: constant p={value:.6f}"
    pos_w = min(float(max_pos_weight), max(1.0, (neg / max(1, pos)) ** float(pos_weight_power)))
    sw = np.ones(len(yy), dtype=np.float32)
    sw[yy == 1] = pos_w
    clf = HistGradientBoostingClassifier(
        max_iter=int(max_iter),
        learning_rate=0.045,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.02,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=15,
        random_state=seed + j,
        verbose=0,
    )
    t0 = time.time()
    clf.fit(x, yy, sample_weight=sw)
    msg = f"[fit] {name}: pos={pos} neg={neg} pos_w={pos_w:.3f} trees={clf.n_iter_} elapsed={time.time()-t0:.1f}s"
    return j, clf, msg


def train_models(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    max_iter: int,
    pos_weight_power: float,
    max_pos_weight: float,
    fit_workers: int = 1,
) -> list:
    actual_workers = max(1, min(int(fit_workers), len(OUT_NAMES)))
    if actual_workers == 1:
        results = [
            train_one_model(j, x, y, seed, max_iter, pos_weight_power, max_pos_weight)
            for j in range(len(OUT_NAMES))
        ]
    else:
        print(f"[fit] workers={actual_workers}", flush=True)
        with ProcessPoolExecutor(max_workers=actual_workers) as ex:
            futures = [
                ex.submit(train_one_model, j, x, y, seed, max_iter, pos_weight_power, max_pos_weight)
                for j in range(len(OUT_NAMES))
            ]
            results = [fut.result() for fut in as_completed(futures)]
    models: list[object | None] = [None] * len(OUT_NAMES)
    for j, model, msg in sorted(results, key=lambda item: item[0]):
        print(msg, flush=True)
        models[j] = model
    return models


def predict_models(models: list, x: np.ndarray) -> np.ndarray:
    probs = np.zeros((len(x), len(models)), dtype=np.float32)
    for j, model in enumerate(models):
        probs[:, j] = model.predict_proba(x)[:, 1].astype(np.float32)
    return probs


def tuned_macro_f1(probs: np.ndarray, labels: np.ndarray) -> tuple[list[float], list[float]]:
    f1s: list[float] = []
    thrs: list[float] = []
    for j in range(labels.shape[1]):
        best = (-1.0, 0.5)
        for thr in np.arange(0.05, 0.95, 0.01):
            f = f1_score(labels[:, j], probs[:, j] >= thr, zero_division=0)
            if f > best[0]:
                best = (float(f), float(thr))
        f1s.append(best[0])
        thrs.append(best[1])
    return f1s, thrs


def ref_rate_f1(probs: np.ndarray, labels: np.ndarray) -> list[float]:
    ref = {"c": 0.964, "na": 0.862, "i": 0.097, "bc": 0.031, "t": 0.450}
    f1s = []
    for j, name in enumerate(OUT_NAMES):
        k = int(round(ref[name] * len(probs)))
        pred = np.zeros(len(probs), dtype=np.int8)
        pred[np.argsort(probs[:, j])[-k:]] = 1
        f1s.append(float(f1_score(labels[:, j], pred, zero_division=0)))
    return f1s


def write_metrics(path: Path, probs: np.ndarray, labels: np.ndarray, meta: dict) -> None:
    aucs = []
    for j in range(labels.shape[1]):
        if labels[:, j].sum() == 0 or labels[:, j].sum() == len(labels):
            aucs.append(None)
        else:
            aucs.append(float(roc_auc_score(labels[:, j], probs[:, j])))
    tuned, thrs = tuned_macro_f1(probs, labels)
    rr = ref_rate_f1(probs, labels)
    out = {
        **meta,
        "auc": dict(zip(OUT_NAMES, aucs)),
        "tuned_f1": dict(zip(OUT_NAMES, tuned)),
        "tuned_thresholds": dict(zip(OUT_NAMES, thrs)),
        "tuned_macro_f1": float(np.mean(tuned)),
        "ref_rate_f1": dict(zip(OUT_NAMES, rr)),
        "ref_rate_macro_f1": float(np.mean(rr)),
        "label_rate": dict(zip(OUT_NAMES, labels.mean(axis=0).astype(float).tolist())),
        "prob_mean": dict(zip(OUT_NAMES, probs.mean(axis=0).astype(float).tolist())),
    }
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def sorted_save_npz(path: Path, probs: np.ndarray, labels: np.ndarray, conv_ids: np.ndarray, end_idxs: np.ndarray) -> None:
    order = np.lexsort((end_idxs, conv_ids.astype(str)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        probs=probs[order].astype(np.float32),
        labels=labels[order].astype(np.int8),
        conv_ids=conv_ids[order].astype(str),
        end_idxs=end_idxs[order].astype(np.int64),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="outputs/oof/sota_e12_ckpt_config.yaml")
    p.add_argument("--out", default="outputs/oof/audio_state_quick_grid25.npz")
    p.add_argument("--grid_stride", type=int, default=25)
    p.add_argument("--no_text", action="store_true")
    p.add_argument("--no_audio", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--workers", type=int, default=max(1, min(128, os.cpu_count() or 1)))
    p.add_argument("--fit_workers", type=int, default=max(1, min(5, os.cpu_count() or 1)))
    p.add_argument("--max_iter", type=int, default=180)
    p.add_argument("--pos_weight_power", type=float, default=0.5)
    p.add_argument("--max_pos_weight", type=float, default=12.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = here / config_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = here / out_path

    cfg = load_config(config_path)
    labels_cfg = cfg["labels"]
    raw_ids = [int(labels_cfg[OUT_TO_CFG[name]]) for name in OUT_NAMES]
    chunk_ms = int(cfg["chunk_ms"])
    context_chunks = int(cfg["context_chunks"])
    target_chunks = int(cfg["target_chunks"])
    seed = int(cfg.get("split", {}).get("seed", cfg.get("seed", 42)))

    labels_dir = Path(cfg["paths"]["train_labels_dir"])
    text_dir = Path(cfg["paths"]["train_text_dir"])
    audio_dir = Path(cfg["paths"]["train_audio_dir"])
    train_ids, valid_ids = split_conversation_ids(
        list_conv_ids(labels_dir),
        valid_ratio=float(cfg.get("split", {}).get("valid_ratio", 0.1)),
        seed=seed,
    )
    print(f"[split] train_convs={len(train_ids)} valid_convs={len(valid_ids)} stride={args.grid_stride}")
    print(f"[features] state=on text={not args.no_text} audio={not args.no_audio} workers={args.workers}")

    x_train, y_train, _, _, feat_names = collect_split(
        train_ids,
        labels_dir,
        text_dir,
        audio_dir,
        raw_ids,
        context_chunks,
        target_chunks,
        int(args.grid_stride),
        chunk_ms,
        use_text=not args.no_text,
        use_audio=not args.no_audio,
        tag="train",
        workers=int(args.workers),
    )
    x_fit, y_fit = maybe_subsample(x_train, y_train, args.max_train_samples, seed)
    if len(x_fit) != len(x_train):
        print(f"[subsample] fit_samples={len(x_fit)} / train_samples={len(x_train)}")

    x_valid, y_valid, conv_valid, end_valid, feat_names_valid = collect_split(
        valid_ids,
        labels_dir,
        text_dir,
        audio_dir,
        raw_ids,
        context_chunks,
        target_chunks,
        int(args.grid_stride),
        chunk_ms,
        use_text=not args.no_text,
        use_audio=not args.no_audio,
        tag="valid",
        workers=int(args.workers),
    )
    if feat_names != feat_names_valid:
        raise ValueError("feature names mismatch between train and valid")

    print(f"[matrix] train={x_train.shape} fit={x_fit.shape} valid={x_valid.shape} features={len(feat_names)}")
    models = train_models(
        x_fit,
        y_fit,
        seed=seed,
        max_iter=int(args.max_iter),
        pos_weight_power=float(args.pos_weight_power),
        max_pos_weight=float(args.max_pos_weight),
        fit_workers=int(args.fit_workers),
    )
    probs = predict_models(models, x_valid)

    sorted_save_npz(out_path, probs, y_valid, conv_valid, end_valid)
    meta = {
        "config": str(config_path),
        "out": str(out_path),
        "grid_stride": int(args.grid_stride),
        "workers": int(args.workers),
        "fit_workers": int(args.fit_workers),
        "train_convs": len(train_ids),
        "valid_convs": len(valid_ids),
        "train_samples": int(len(x_train)),
        "fit_samples": int(len(x_fit)),
        "valid_samples": int(len(x_valid)),
        "feature_count": len(feat_names),
        "feature_sources": {"state": True, "text": not args.no_text, "audio": not args.no_audio},
        "feature_names": feat_names,
        "note": "weak-label valid probe; not an artificial-label truth claim",
    }
    metrics_path = out_path.with_suffix(".metrics.json")
    write_metrics(metrics_path, probs, y_valid, meta)
    print(f"[saved] {out_path} probs={probs.shape} labels={y_valid.shape}")
    print(f"[saved] {metrics_path}")


if __name__ == "__main__":
    main()
