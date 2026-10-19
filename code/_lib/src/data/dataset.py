import json
import random
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


@dataclass
class TrainSample:
    conv_id: str
    end_idx: int
    label: int


@dataclass
class TrainSampleMulti:
    conv_id: str
    end_idx: int
    # (BC, I, T) or any configured order
    label_vec: Tuple[int, ...]


FA_DIM = 5
_FA_GAP_CAP_MS = 5000.0


def fa_rasterize(fa_json: Optional[Dict], start_ms: int, end_ms: int, chunk_ms: int) -> np.ndarray:
    """Char-level FA timestamps -> per-chunk grid [n_chunks, 5]:
    speak_ch0, speak_ch1, overlap, gap_ch0, gap_ch1 (gap = time since that
    channel's last char end, log-normalized, capped 5s). Vectorized."""
    n = (end_ms - start_ms) // chunk_ms
    out = np.zeros((n, FA_DIM), np.float32)
    if not fa_json:
        return out
    norm = np.log1p(_FA_GAP_CAP_MS)
    chunk_hi = start_ms + (np.arange(n, dtype=np.float64) + 1) * chunk_ms
    for k in (1, 2):
        starts, ends = [], []
        for u in fa_json.get("utts", []):
            if u["channel_id"] != k:
                continue
            for c in u["chars"]:
                starts.append(c["s"])
                ends.append(c["e"])
        if not starts:
            out[:, 2 + k] = 1.0
            continue
        s_arr = np.asarray(starts, np.float64)
        e_arr = np.asarray(ends, np.float64)
        # speak: fill chunk range per span (only spans touching the window)
        m = (s_arr < end_ms) & (e_arr > start_ms)
        for s, e in zip(s_arr[m], e_arr[m]):
            a = max(0, int((s - start_ms) // chunk_ms))
            b = min(n, int(np.ceil((e - start_ms) / chunk_ms)))
            if b > a:
                out[a:b, k - 1] = 1.0
        # gap: time since latest char end <= chunk_hi
        e_sorted = np.sort(e_arr)
        idx = np.searchsorted(e_sorted, chunk_hi, side="right") - 1
        gap = np.where(idx >= 0, chunk_hi - e_sorted[np.clip(idx, 0, None)], _FA_GAP_CAP_MS)
        gap = np.clip(gap, 0.0, _FA_GAP_CAP_MS)
        gap[out[:, k - 1] > 0] = 0.0
        out[:, 2 + k] = np.log1p(gap) / norm
    out[:, 2] = out[:, 0] * out[:, 1]
    return out


def list_conv_ids(labels_dir: Path) -> List[str]:
    return sorted([p.stem for p in labels_dir.glob("*.npy")])


def split_conversation_ids(
    conv_ids: Sequence[str], valid_ratio: float, seed: int
) -> Dict[str, List[str]]:
    conv_ids = list(conv_ids)
    random.Random(seed).shuffle(conv_ids)
    valid_size = max(1, int(len(conv_ids) * valid_ratio))
    valid_ids = sorted(conv_ids[:valid_size])
    train_ids = sorted(conv_ids[valid_size:])
    return {"train": train_ids, "valid": valid_ids}


def build_train_samples(
    labels_dir: Path,
    conv_ids: Sequence[str],
    context_chunks: int,
    target_chunks: int,
    stride: int,
    positive_ids: Sequence[int],
    max_samples: Optional[int] = None,
) -> List[TrainSample]:
    samples: List[TrainSample] = []
    pos_set = set(positive_ids)
    for conv_id in conv_ids:
        labels = np.load(labels_dir / f"{conv_id}.npy")
        max_end = labels.shape[0] - target_chunks
        for end_idx in range(context_chunks, max_end + 1, stride):
            future = labels[end_idx : end_idx + target_chunks]
            y = int(any(int(x) in pos_set for x in future))
            samples.append(TrainSample(conv_id=conv_id, end_idx=end_idx, label=y))
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


def build_train_samples_multitask(
    labels_dir: Path,
    conv_ids: Sequence[str],
    context_chunks: int,
    target_chunks: int,
    stride: int,
    label_ids: Dict[str, int],
    target_labels: Sequence[str] = ("BC", "I", "T"),
    max_samples: Optional[int] = None,
) -> List[TrainSampleMulti]:
    samples: List[TrainSampleMulti] = []
    target_id_list = [int(label_ids[k]) for k in target_labels]
    for conv_id in conv_ids:
        labels = np.load(labels_dir / f"{conv_id}.npy")
        max_end = labels.shape[0] - target_chunks
        for end_idx in range(context_chunks, max_end + 1, stride):
            future = labels[end_idx : end_idx + target_chunks]
            y_vec = tuple(int(any(int(x) == tid for x in future)) for tid in target_id_list)
            samples.append(TrainSampleMulti(conv_id=conv_id, end_idx=end_idx, label_vec=y_vec))
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


def _speaker_token(channel_id: int) -> str:
    return "[SPK1]" if channel_id == 1 else "[SPK2]"


def build_text_context(
    utterances: Iterable[Dict],
    start_ms: int,
    end_ms: int,
    max_utterances: int = 120,
) -> str:
    selected = []
    for utt in utterances:
        utt_start = int(utt.get("start_ms", 0))
        utt_end = int(utt.get("end_ms", utt_start))
        if utt_end <= start_ms or utt_start >= end_ms:
            continue
        text = str(utt.get("text", "")).strip()
        if not text:
            continue
        selected.append(f"{_speaker_token(int(utt.get('channel_id', 1)))} {text}")
    if not selected:
        return "[SPK1] <silence> [SPK2] <silence>"
    if len(selected) > max_utterances:
        selected = selected[-max_utterances:]
    return " ".join(selected)


def _read_wav_slice(path: Path, start_ms: int, end_ms: int) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        n_ch = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        start_frame = max(0, int(start_ms * sr / 1000))
        end_frame = max(start_frame + 1, int(end_ms * sr / 1000))
        total_frames = wf.getnframes()
        end_frame = min(end_frame, total_frames)
        read_frames = max(1, end_frame - start_frame)
        wf.setpos(start_frame)
        raw = wf.readframes(read_frames)

    if sampwidth == 2:
        dtype = np.int16
        scale = 32768.0
    elif sampwidth == 4:
        dtype = np.int32
        scale = 2147483648.0
    elif sampwidth == 1:
        dtype = np.uint8
        scale = 128.0
    else:
        raise RuntimeError(f"Unsupported wav sample width {sampwidth} for {path}")

    data = np.frombuffer(raw, dtype=dtype)
    if n_ch > 1:
        data = data.reshape(-1, n_ch)
    else:
        data = data[:, None]

    if sampwidth == 1:
        data = (data.astype(np.float32) - 128.0) / scale
    else:
        data = data.astype(np.float32) / scale
    return data, sr


class TurnTakingTrainDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[TrainSample | TrainSampleMulti],
        train_audio_dir: Path,
        train_text_dir: Path,
        train_labels_dir: Path,
        context_chunks: int,
        target_chunks: int,
        chunk_ms: int,
        sample_rate: int,
        future_horizons: tuple[int, ...] | None = None,
        fa_dir: Path | None = None,
        audio_augment: Optional[Dict] = None,
    ) -> None:
        self.samples = list(samples)
        self.train_audio_dir = train_audio_dir
        self.train_text_dir = train_text_dir
        self.train_labels_dir = train_labels_dir
        self.context_chunks = context_chunks
        self.target_chunks = target_chunks
        self.chunk_ms = chunk_ms
        self.sample_rate = sample_rate
        self.future_horizons = tuple(future_horizons) if future_horizons else ()
        self.fa_dir = Path(fa_dir) if fa_dir else None
        # SpecAugment-style waveform-domain augmentation; None or {enabled: false} disables it.
        self._aug_cfg = audio_augment if (audio_augment and audio_augment.get("enabled", False)) else None

    def _apply_audio_augment(self, wave: torch.Tensor) -> torch.Tensor:
        # wave: [C=2, T] float; mutate per-channel independently.
        cfg = self._aug_cfg
        if cfg is None:
            return wave
        C, T = wave.shape
        sr = self.sample_rate
        # white noise at controlled SNR (applied first so subsequent time-mask zeros remain truly 0).
        ns = cfg.get("noise", {})
        if ns and random.random() < float(ns.get("apply_prob", 0.0)):
            snr_lo = float(ns.get("snr_db_min", 15.0))
            snr_hi = float(ns.get("snr_db_max", 30.0))
            snr_db = random.uniform(snr_lo, snr_hi)
            sig_rms = wave.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            noise_rms = sig_rms * (10.0 ** (-snr_db / 20.0))
            wave = wave + torch.randn_like(wave) * noise_rms
        # time mask: zero-out N random segments per channel; keeps labels invariant.
        tm = cfg.get("time_mask", {})
        if tm and random.random() < float(tm.get("apply_prob", 0.0)):
            n_masks = int(tm.get("num", 1))
            max_w = int(float(tm.get("max_ms", 0)) * sr / 1000)
            if max_w > 0:
                for c in range(C):
                    for _ in range(n_masks):
                        w = random.randint(1, max_w)
                        if w >= T:
                            continue
                        s = random.randint(0, T - w)
                        wave[c, s : s + w] = 0.0
        return wave

    @lru_cache(maxsize=256)
    def _load_fa_json(self, conv_id: str) -> Optional[Dict]:
        p = self.fa_dir / f"{conv_id}.json"
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    @lru_cache(maxsize=256)
    def _load_labels(self, conv_id: str) -> np.ndarray:
        return np.load(self.train_labels_dir / f"{conv_id}.npy")

    @lru_cache(maxsize=256)
    def _load_text_json(self, conv_id: str) -> Dict:
        with open(self.train_text_dir / f"{conv_id}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_wave_segment(self, conv_id: str, start_ms: int, end_ms: int) -> torch.Tensor:
        wav_path = self.train_audio_dir / f"{conv_id}.wav"
        audio, src_sr = _read_wav_slice(wav_path, start_ms, end_ms)
        wave = torch.from_numpy(audio.T)  # [C, T]
        if wave.shape[0] == 1:
            wave = wave.repeat(2, 1)
        elif wave.shape[0] > 2:
            wave = wave[:2]

        if src_sr != self.sample_rate:
            wave = torchaudio.functional.resample(wave, src_sr, self.sample_rate)

        expected_frames = int((end_ms - start_ms) * self.sample_rate / 1000)
        if wave.shape[1] < expected_frames:
            pad = expected_frames - wave.shape[1]
            wave = torch.nn.functional.pad(wave, (0, pad))
        elif wave.shape[1] > expected_frames:
            wave = wave[:, :expected_frames]
        return wave

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        labels = self._load_labels(sample.conv_id)
        end_idx = sample.end_idx
        start_idx = end_idx - self.context_chunks

        context_labels = labels[start_idx:end_idx].astype(np.int64)
        start_ms = start_idx * self.chunk_ms
        end_ms = end_idx * self.chunk_ms

        text_json = self._load_text_json(sample.conv_id)
        text = build_text_context(text_json.get("utterances", []), start_ms, end_ms)
        wave = self._load_wave_segment(sample.conv_id, start_ms, end_ms)
        if self._aug_cfg is not None:
            wave = self._apply_audio_augment(wave)

        out = {
            "conv_id": sample.conv_id,
            "end_idx": end_idx,
            "waveform": wave,
            "text": text,
            "context_labels": torch.from_numpy(context_labels),
        }
        if self.fa_dir is not None:
            fa = fa_rasterize(self._load_fa_json(sample.conv_id), start_ms, end_ms, self.chunk_ms)
            out["fa_feats"] = torch.from_numpy(fa)
        if hasattr(sample, "label_vec"):
            out["label"] = torch.tensor(sample.label_vec, dtype=torch.float32)
        else:
            out["label"] = torch.tensor(float(sample.label), dtype=torch.float32)

        if self.future_horizons:
            n_labels = len(labels)
            futs = []
            for h in self.future_horizons:
                pos = end_idx + h
                if pos < n_labels:
                    futs.append(int(labels[pos]))
                else:
                    futs.append(-100)  # ignore_index for out-of-range
            out["future_labels"] = torch.tensor(futs, dtype=torch.long)
        return out


class TurnTakingTestDataset(Dataset):
    def __init__(
        self,
        test_root: Path,
        sample_rate: int,
    ) -> None:
        self.sample_rate = sample_rate
        self.base = test_root
        self.audio_dir = self.base / "audio"
        self.text_dir = self.base / "text"
        self.context_dir = self.base / "context"
        self.segment_ids = sorted([p.stem for p in self.context_dir.glob("*.npy")])
        self.fa_dir = None
        self.chunk_ms = 80

    def set_fa(self, fa_dir, chunk_ms: int = 80) -> None:
        self.fa_dir = Path(fa_dir) if fa_dir else None
        self.chunk_ms = chunk_ms

    @lru_cache(maxsize=1024)
    def _load_fa_json(self, seg_id: str) -> Optional[Dict]:
        p = self.fa_dir / f"{seg_id}.json"
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.segment_ids)

    @lru_cache(maxsize=512)
    def _load_text_json(self, seg_id: str) -> Dict:
        with open(self.text_dir / f"{seg_id}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def __getitem__(self, idx: int) -> Dict:
        # 修复 5-30s 不定长录音 collate bug：
        # 训练时 context_chunks=375 (30s/80ms)，wav=30s*16000Hz
        # 私域 test 长度不一致 → 必须 left-pad/right-trunc 到固定 (NA=4 pad)
        CTX_TARGET_LEN = 375
        WAV_TARGET_LEN = 30 * self.sample_rate  # 30s
        PAD_NA = 4  # baseline labels: NA=4

        seg_id = self.segment_ids[idx]
        context_labels = np.load(self.context_dir / f"{seg_id}.npy").astype(np.int64)
        n_ctx = len(context_labels)
        if n_ctx > CTX_TARGET_LEN:
            context_labels = context_labels[-CTX_TARGET_LEN:]
        elif n_ctx < CTX_TARGET_LEN:
            context_labels = np.pad(context_labels, (CTX_TARGET_LEN - n_ctx, 0), constant_values=PAD_NA)

        text_json = self._load_text_json(seg_id)
        start_ms = int(text_json.get("start_ms", 0))
        end_ms = int(text_json.get("end_ms", 30000))
        text = build_text_context(text_json.get("utterances", []), start_ms, end_ms)

        wav_path = self.audio_dir / f"{seg_id}.wav"
        audio, src_sr = _read_wav_slice(wav_path, start_ms, end_ms)
        wave = torch.from_numpy(audio.T)
        if wave.shape[0] == 1:
            wave = wave.repeat(2, 1)
        elif wave.shape[0] > 2:
            wave = wave[:2]
        if src_sr != self.sample_rate:
            wave = torchaudio.functional.resample(wave, src_sr, self.sample_rate)
        cur = wave.shape[1]
        if cur > WAV_TARGET_LEN:
            wave = wave[:, -WAV_TARGET_LEN:]
        elif cur < WAV_TARGET_LEN:
            wave = torch.nn.functional.pad(wave, (WAV_TARGET_LEN - cur, 0))

        out = {
            "segment_id": seg_id,
            "waveform": wave,
            "text": text,
            "context_labels": torch.from_numpy(context_labels),
        }
        if self.fa_dir is not None:
            fa = fa_rasterize(self._load_fa_json(seg_id), start_ms, end_ms, self.chunk_ms)
            # fa 也归一化到 CTX_TARGET_LEN
            if fa.shape[0] > CTX_TARGET_LEN:
                fa = fa[-CTX_TARGET_LEN:]
            elif fa.shape[0] < CTX_TARGET_LEN:
                pad_width = ((CTX_TARGET_LEN - fa.shape[0], 0),) + ((0, 0),) * (fa.ndim - 1)
                fa = np.pad(fa, pad_width, constant_values=0)
            out["fa_feats"] = torch.from_numpy(fa)
        return out


def build_collate_fn(tokenizer, text_max_length: int):
    def _collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        tokenized = tokenizer(
            texts,
            max_length=text_max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        waves = [b["waveform"] for b in batch]
        max_len = max(w.shape[1] for w in waves)
        padded_waves = []
        for w in waves:
            if w.shape[1] < max_len:
                w = torch.nn.functional.pad(w, (0, max_len - w.shape[1]))
            padded_waves.append(w)

        out = {
            "waveform": torch.stack(padded_waves, dim=0),
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "context_labels": torch.stack([b["context_labels"] for b in batch], dim=0),
        }
        if "fa_feats" in batch[0]:
            out["fa_feats"] = torch.stack([b["fa_feats"] for b in batch], dim=0)

        if "label" in batch[0]:
            out["label"] = torch.stack([b["label"] for b in batch], dim=0)
            out["conv_id"] = [b["conv_id"] for b in batch]
            out["end_idx"] = [b["end_idx"] for b in batch]
            if "future_labels" in batch[0]:
                out["future_labels"] = torch.stack([b["future_labels"] for b in batch], dim=0)
        else:
            out["segment_id"] = [b["segment_id"] for b in batch]
        return out

    return _collate
