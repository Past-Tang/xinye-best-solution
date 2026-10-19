"""Acoustic overlap probability for test via callhome-zho segmentation model.

PyanNet (segmentation-3.0 arch) fine-tuned on CALLHOME Mandarin; powerset 7:
  [empty, s1, s2, s3, s1+2, s1+3, s2+3] -> P(overlap) = sum of last 3.
Input: 10s mono 16k windows. We feed test 30s (mono mix, 8k->16k) as three
10s chunks, concat frames, then pool tail windows.

Output npy columns: [ov_mean_full, ov_mean_5s, ov_mean_2s, ov_max_2s]

Usage:
  python tools_acoustic_overlap.py \
    --audio /data/test/audio \
    --ckpt models/callhome-zho/snapshots/c80be888e7fcfd8f9431e5fb57886c7289e5986c/model.safetensors \
    --out outputs/test_probs/acoustic_overlap_test.npy
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import soundfile as sf
import torch
from pathlib import Path

ROOT = Path("/workspace/xinye")
CKPT_DEFAULT = ROOT / ".cache/huggingface/models--diarizers-community--speaker-segmentation-fine-tuned-callhome-zho/snapshots/c80be888e7fcfd8f9431e5fb57886c7289e5986c/model.safetensors"
OUT_DEFAULT = ROOT / "code_src/pb/outputs/test_probs/acoustic_overlap_test.npy"
AUDIO_DEFAULT = ROOT / "data/test/audio"
SR = 16000


def build_model(ckpt_path):
    from pyannote.audio.models.segmentation import PyanNet
    from safetensors.torch import load_file
    m = PyanNet(sincnet={"stride": 10}, sample_rate=SR,
                lstm={"hidden_size": 128, "num_layers": 4, "bidirectional": True,
                      "monolithic": True, "dropout": 0.0},
                linear={"hidden_size": 128, "num_layers": 2})
    # specifications needed before classifier exists; emulate 7-class powerset
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    m.specifications = Specifications(
        problem=Problem.MONO_LABEL_CLASSIFICATION,
        resolution=Resolution.FRAME,
        duration=10.0,
        classes=["e", "s1", "s2", "s3", "s12", "s13", "s23"],
    )
    m.build()
    sd = load_file(ckpt_path)
    sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print("missing:", missing, "unexpected:", unexpected)
    return m.eval()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default=str(AUDIO_DEFAULT),
                   help="dir of test *.wav files")
    p.add_argument("--ckpt", default=str(CKPT_DEFAULT),
                   help="callhome-zho safetensors path")
    p.add_argument("--out", default=str(OUT_DEFAULT),
                   help="output npy path (a sibling _ids.txt is auto-generated)")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    import torchaudio
    model = build_model(args.ckpt).to(args.device)
    test_audio = Path(args.audio)
    ids = sorted(p.stem for p in test_audio.glob("*.wav"))
    feats = np.zeros((len(ids), 4), np.float32)
    B = 32
    for bi in range(0, len(ids), B):
        chunk_ids = ids[bi:bi + B]
        wavs = []
        for sid in chunk_ids:
            w, sr = sf.read(test_audio / f"{sid}.wav", dtype="float32")
            mono = torch.from_numpy(w.mean(1) if w.ndim > 1 else w)[None]
            mono = torchaudio.functional.resample(mono, sr, SR)[0]
            need = SR * 30
            mono = torch.nn.functional.pad(mono, (0, max(0, need - len(mono))))[:need]
            wavs.append(mono)
        x = torch.stack(wavs).to(args.device)              # [B, 480000]
        xs = x.view(len(chunk_ids) * 3, 1, SR * 10)        # 3x 10s chunks
        logits = model(xs)                                 # [B*3, F, 7]
        p = torch.softmax(logits, -1)
        ov = p[..., 4:].sum(-1)                            # P(>=2 speakers)
        F = ov.shape[1]
        ov = ov.view(len(chunk_ids), 3 * F).cpu().numpy()  # frames over 30s
        fps = 3 * F / 30.0
        n5, n2 = int(5 * fps), int(2 * fps)
        feats[bi:bi + len(chunk_ids), 0] = ov.mean(1)
        feats[bi:bi + len(chunk_ids), 1] = ov[:, -n5:].mean(1)
        feats[bi:bi + len(chunk_ids), 2] = ov[:, -n2:].mean(1)
        feats[bi:bi + len(chunk_ids), 3] = ov[:, -n2:].max(1)
        if bi % 320 == 0:
            print(f"{bi}/{len(ids)}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, feats)
    ids_path = out.with_name(out.stem + "_ids.txt")
    with open(ids_path, "w") as f:
        f.write("\n".join(ids) + "\n")
    print("saved", out, feats.shape, "ids->", ids_path)


if __name__ == "__main__":
    main()
