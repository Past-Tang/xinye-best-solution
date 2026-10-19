"""WavLMTemporalModel — 第一性原理破局架构.

核心理论 (基于 deep probe 数据驱动洞察):
  1. Whisper-large-v3 是 ASR 模型, 主动学习对说话人/韵律/重叠不变 → 与 turn-taking 错配
     替换为 WavLM-large (SSL 自监督, 在说话人验证/分离/diarization 上是 SOTA, 保留韵律)
  2. frame-level |ch0-ch1| 必须在 pool 之前计算 (deep probe: BC 0.57→0.63 仅因这一改动)
  3. tail pool 必须是 1-2s 范围 (probe: N=10/25/50 全部 >0.6, N=300 暴跌到 0.52)
  4. 多尺度 tail 拼接给 best AUC (probe: all_concat T=0.66 vs 单尺度最高 0.62)

I/O 契约与 MultimodalLoRAChunkAuxModel/Temporal 一致: forward 4-tuple,
复用 train.py / Mix2 / infer.py 不需任何改动 (按 cfg.model.type='wavlm_temporal' 选择)。
"""
from __future__ import annotations

from typing import Dict, Tuple

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
from transformers import AutoModel, WavLMModel

# 复用现有组件 (TextEncoder, ChunkAuxHead, FutureAuxHead)
_BASELINE_ROOT = Path(__file__).resolve().parent.parent / "_lib"
if str(_BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINE_ROOT))

from src.models.multimodal_baseline import TextEncoder  # noqa: E402

from .chunk_aux_head import ChunkAuxHead
from .model import FutureAuxHead


def wrap_wavlm_lora(encoder: nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "out_proj"),
                    enable_gradient_checkpointing: bool = True):
    """WavLM 与 Whisper 同样有 q/k/v/out_proj. PEFT LoraConfig 直接适用."""
    cfg = LoraConfig(
        r=int(r), lora_alpha=int(alpha), target_modules=list(target_modules),
        lora_dropout=float(dropout), bias="none", task_type=None,
    )
    peft_model = get_peft_model(encoder, cfg)
    if enable_gradient_checkpointing:
        if hasattr(peft_model, "gradient_checkpointing_enable"):
            peft_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        # WavLM 输入是 raw waveform, 没有 token embedding;
        # enable_input_require_grads() 会调 get_input_embeddings() 失败. 跳过即可——
        # 因为 LoRA 参数本身就 requires_grad=True, ckpt 重算时上游梯度从 LoRA 流入。
        try:
            if hasattr(peft_model, "enable_input_require_grads"):
                peft_model.enable_input_require_grads()
        except NotImplementedError:
            pass
    return peft_model


class FrameDownsampler(nn.Module):
    """[B,Tw,Din] WavLM 50Hz -> [B,375,Dout] 12.5Hz (与 ctx 对齐).

    Whisper-large-v3 也是 50Hz 输出, 1500 帧/30s, 与 WavLM 同。stride=4 下采到 375 帧。
    """
    def __init__(self, in_dim: int, out_dim: int, target_frames: int = 375):
        super().__init__()
        # WavLM 30s→约 1499 帧(因为 conv stride 边界)，下采到 374-375
        self.target = int(target_frames)
        self.down = nn.Conv1d(in_dim, out_dim, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, h):  # [B,Tw,D]
        x = h.transpose(1, 2)         # [B,D,Tw]
        x = self.down(x)              # [B,Dout,~375]
        if x.shape[-1] != self.target:
            if x.shape[-1] < self.target:
                x = F.pad(x, (0, self.target - x.shape[-1]))
            else:
                x = x[..., :self.target]
        x = self.norm(x.transpose(1, 2))
        return self.act(x)            # [B,375,Dout]


class MultiScaleTailPool(nn.Module):
    """Multi-scale tail mean pool (deep probe 证最优特征结构)."""
    def __init__(self, dim: int, ns=(10, 25, 50)):
        super().__init__()
        self.ns = tuple(ns)
        self.norm = nn.LayerNorm(dim * len(self.ns))

    def forward(self, seq):  # [B,T,D]
        T = seq.shape[1]
        parts = []
        for n in self.ns:
            k = min(n, T)
            parts.append(seq[:, -k:, :].mean(dim=1))   # [B,D]
        out = torch.cat(parts, dim=-1)                 # [B,D*len(ns)]
        return self.norm(out)


class MultimodalWavLMTemporalModel(nn.Module):
    """WavLM 主干 + frame-level |diff| + multi-scale tail pool."""

    def __init__(self, cfg: Dict):
        super().__init__()
        audio_cfg = cfg["audio_encoder"]
        text_cfg = cfg["text_encoder"]
        ctx_cfg = cfg["context_encoder"]
        head_cfg = cfg["fusion_head"]
        lora_cfg = cfg.get("lora", {})
        chunk_cfg = cfg.get("chunk_aux", {})
        temp_cfg = cfg.get("temporal", {})

        # ---- WavLM encoder + LoRA (单塔, 双声道共享) ----
        wavlm_path = audio_cfg["model_name"]
        # AutoModel: WavLM/Hubert 同族接口一致, 按 config.json 自动选型
        encoder = AutoModel.from_pretrained(wavlm_path)
        # WavLM feature extractor (CNN 部分) 永远 freeze
        for p in encoder.feature_extractor.parameters():
            p.requires_grad_(False)
        self.audio_encoder = wrap_wavlm_lora(
            encoder,
            r=int(lora_cfg.get("r", 16)), alpha=int(lora_cfg.get("alpha", 32)),
            dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=tuple(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"])),
            enable_gradient_checkpointing=bool(lora_cfg.get("gradient_checkpointing", True)),
        )
        self.audio_dim = int(encoder.config.hidden_size)  # 1024

        # ---- 帧级下采样到 375 帧 (ctx 对齐) ----
        self.frame_mid = int(temp_cfg.get("frame_mid", 320))
        self.target_frames = int(chunk_cfg.get("target_frames", 375))
        self.down_ch = FrameDownsampler(self.audio_dim, self.frame_mid, self.target_frames)
        # |diff| 单独 downsampler (重要: deep probe 证 frame-level diff 必须在 pool 前)
        self.down_diff = FrameDownsampler(self.audio_dim, self.frame_mid, self.target_frames)

        # ---- Context labels: 帧级 embedding ----
        self.ctx_embed_dim = int(ctx_cfg.get("embed_dim", 16))
        self.ctx_frame_embed = nn.Embedding(int(ctx_cfg["vocab_size"]), self.ctx_embed_dim)

        # ---- 帧级融合: [ch0, ch1, |diff|, ctx] ----
        frame_in = self.frame_mid * 3 + self.ctx_embed_dim
        self.d_model = int(temp_cfg.get("d_model", 384))
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_in, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.target_frames, self.d_model) * 0.02)

        # ---- 时序 Transformer ----
        n_layers = int(temp_cfg.get("layers", 2))
        n_heads = int(temp_cfg.get("heads", 6))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_heads,
            dim_feedforward=self.d_model * 4, dropout=float(temp_cfg.get("dropout", 0.1)),
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ---- Multi-scale tail pool (probe 证最优) ----
        self.tail_ns = tuple(temp_cfg.get("tail_ns", [10, 25, 50]))
        self.tail_pool = MultiScaleTailPool(self.d_model, ns=self.tail_ns)
        self.tail_out_dim = self.d_model * len(self.tail_ns)

        # ---- Text: frozen Qwen3-0.6B (全局条件) ----
        self.text_encoder = TextEncoder(
            model_name=text_cfg["model_name"],
            freeze_backbone=bool(text_cfg.get("freeze_backbone", True)),
            tail_ratio=float(text_cfg.get("tail_ratio", 0.3)),
        )

        # ---- Aux heads (复用) ----
        self.use_aux = bool(chunk_cfg.get("enabled", True))
        if self.use_aux:
            self.chunk_aux_head = ChunkAuxHead(
                in_dim=self.audio_dim,
                mid_dim=int(chunk_cfg.get("mid_dim", 256)),
                num_classes=int(chunk_cfg.get("num_classes", 5)),
                whisper_frames=int(chunk_cfg.get("whisper_frames", 1500)),
                target_frames=self.target_frames,
            )
        future_cfg = cfg.get("future_aux", {})
        self.use_future_aux = bool(future_cfg.get("enabled", False))
        if self.use_future_aux:
            self.future_aux_head = FutureAuxHead(
                in_dim=self.audio_dim,
                mid_dim=int(future_cfg.get("mid_dim", 256)),
                num_classes=int(chunk_cfg.get("num_classes", 5)),
                horizons=tuple(future_cfg.get("horizons", [5, 10, 15, 25])),
                tail_frames=int(future_cfg.get("tail_frames", 150)),
            )

        # ---- Fusion + main head ----
        fusion_in = self.tail_out_dim + self.text_encoder.out_dim
        self.fusion_in = int(fusion_in)
        h1, h2 = head_cfg["hidden_dims"]
        dropout = float(head_cfg.get("dropout", 0.3))
        num_targets = len(cfg.get("labels", {}).get("multi_targets", []))
        self.num_targets = num_targets if num_targets > 0 else 5
        self.head = nn.Sequential(
            nn.Linear(fusion_in, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h2, self.num_targets),
        )

    def _encode_channels(self, waveform):
        """waveform [B,2,T] -> (h0, h1) each [B,Tw,1024]."""
        B, C, T = waveform.shape
        if C != 2:
            raise ValueError(f"expected dual-channel, got C={C}")
        wav = waveform.reshape(B * 2, T)
        # WavLM 期望 [B,T] float32, 自动 normalize
        out = self.audio_encoder(input_values=wav)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        Tw, D = hidden.shape[1], hidden.shape[2]
        hidden = hidden.view(B, 2, Tw, D)
        return hidden[:, 0], hidden[:, 1]

    def _build_fusion(self, waveform, context_labels):
        h0, h1 = self._encode_channels(waveform)               # [B,Tw,1024] x2
        # frame-level |diff| 在 pool 之前算 (deep probe 关键)
        diff = (h0 - h1).abs()                                 # [B,Tw,1024]
        f0 = self.down_ch(h0)                                   # [B,375,frame_mid]
        f1 = self.down_ch(h1)                                   # 共享下采权重
        fd = self.down_diff(diff)                               # 独立下采保证 |diff| 信号
        ctx = self.ctx_frame_embed(context_labels.long())[:, :self.target_frames]  # [B,375,ctx_dim]
        T = min(f0.shape[1], ctx.shape[1])
        frame = torch.cat([f0[:, :T], f1[:, :T], fd[:, :T], ctx[:, :T]], dim=-1)
        frame = self.frame_proj(frame) + self.pos_embed[:, :T, :]
        seq = self.temporal(frame)                              # [B,T,d_model]
        tail_vec = self.tail_pool(seq)                          # [B, d_model*3]
        return tail_vec, h0, h1

    def forward(self, waveform, input_ids, attention_mask, context_labels, return_aux=True):
        tail_vec, h0, h1 = self._build_fusion(waveform, context_labels)

        if return_aux and self.use_aux:
            aux_logits = self.chunk_aux_head(0.5 * (h0 + h1))   # 复用 chunk_aux 在 mean(ch) 上
        else:
            aux_logits = None
        if return_aux and self.use_future_aux:
            future_aux_logits = self.future_aux_head(h0, h1)
        else:
            future_aux_logits = None

        text_pool = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        fusion = torch.cat([tail_vec, text_pool], dim=-1)
        main_logits = self.head(fusion)
        return main_logits, aux_logits, fusion, future_aux_logits

    def head_only(self, fusion):
        return self.head(fusion)