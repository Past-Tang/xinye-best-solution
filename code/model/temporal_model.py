"""MultimodalTemporalModel — 时序保留 + 双声道保留架构.

破局动机（信息论级别，已用 train 标签时序诊断验证）:
  当前 SOTA(MultimodalLoRAChunkAuxModel) 在三处摧毁了 I/BC/T 判别所需的信息:
    1. mean-over-channel 抹掉双声道身份  -> I(插话)/BC(附和) 是"谁对谁"的交互事件
    2. tail attention pool 把 1500 帧塌缩成 1 个向量 -> 事件由"最近 2s"触发的时序结构丢失
    3. 独立 sigmoid 无类间互斥 -> 单类互斥标签与输出空间 mismatch

本架构的修复（每条对应一个缺陷）:
  1. 双声道各自编码, 不 mean; 在帧级拼接 [ch0, ch1, |ch0-ch1|]
  2. 对齐到 375 帧后送入双向 Transformer, 全程不做全局 pool;
     用"末端 K 帧 + 可学习 query"做注意力坍缩, 直接对齐"近端触发"物理规律
  3. 主头并联 sigmoid(多标签) 与 softmax(互斥) 两分支, 训练双 loss

I/O 契约与 MultimodalLoRAChunkAuxModel 完全一致, 以复用 train.py / Mix2 / infer.py:
  forward(...) -> (main_logits[B,5], aux_logits[B,5,375] or None,
                   fusion[B,D], future_aux_logits[B,H,5] or None)
  head_only(fusion[B,D]) -> main_logits[B,5]   (Mix2 重算路径)

缺陷 3(类间互斥) 的修复放在 loss 侧(TotalLoss.mutex_weight): 对 main_logits 直接
施加 softmax-CE, 仅作用于单类活跃的样本行, 因此无需第 5 个输出, 4 元组契约保持不变。
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import WhisperGpuMel, FutureAuxHead
from .chunk_aux_head import ChunkAuxHead
from .lora_wrapper import wrap_whisper_lora

import sys
from pathlib import Path

_BASELINE_ROOT = Path(__file__).resolve().parent.parent / "_lib"
if str(_BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINE_ROOT))

from src.models.multimodal_baseline import (  # noqa: E402
    TextEncoder,
)
from transformers import WhisperModel  # noqa: E402


class FrameDownsampler(nn.Module):
    """Whisper 50Hz hidden [B,1500,D] -> 12.5Hz frame grid [B,375,mid].

    Conv1d(kernel=stride=4) 与 ChunkAuxHead 同构, 保证与 context 的 375 帧像素级对齐。
    """

    def __init__(self, in_dim: int, mid_dim: int, whisper_frames: int = 1500, target_frames: int = 375) -> None:
        super().__init__()
        stride = whisper_frames // target_frames
        self.target_frames = int(target_frames)
        self.down = nn.Conv1d(in_dim, mid_dim, kernel_size=stride, stride=stride)
        self.norm = nn.LayerNorm(mid_dim)
        self.act = nn.GELU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.transpose(1, 2)            # [B, D, T]
        x = self.down(x)                       # [B, mid, ~375]
        if x.shape[-1] != self.target_frames:
            if x.shape[-1] < self.target_frames:
                x = F.pad(x, (0, self.target_frames - x.shape[-1]))
            else:
                x = x[..., : self.target_frames]
        x = self.norm(x.transpose(1, 2))       # [B, 375, mid]
        return self.act(x)


class TailQueryPool(nn.Module):
    """末端 K 帧 + 可学习 query 的注意力坍缩.

    动机: train 标签诊断证明 I/BC/T 事件由"最近 ~2s(末端帧)"触发。
    用一个可学习 query 对最后 tail_k 帧做 cross-attention, 既保留时序敏感性,
    又输出固定 [B,D] 供下游 head 与 Mix2 使用。
    """

    def __init__(self, dim: int, tail_k: int = 50, num_heads: int = 4) -> None:
        super().__init__()
        self.tail_k = int(tail_k)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, T, D] -> 取末端 tail_k 帧
        T = seq.shape[1]
        tail = seq[:, max(0, T - self.tail_k):, :]   # [B, k, D]
        B = tail.shape[0]
        q = self.query.expand(B, -1, -1)             # [B, 1, D]
        out, _ = self.attn(q, tail, tail)            # [B, 1, D]
        return self.norm(out.squeeze(1))             # [B, D]


class MultimodalTemporalModel(nn.Module):
    """双声道保留 + 375 帧时序 Transformer + 末端 query 坍缩.

    数据流:
      waveform[B,2,T] -> 每声道 Whisper-LoRA -> ch0/ch1 hidden [B,1500,1280]
        -> FrameDownsampler 各自 -> [B,375,mid]
        -> 帧级拼接 [ch0, ch1, |ch0-ch1|] + context embed -> [B,375,Dfuse]
        -> 双向 TransformerEncoder(因 valid 评测非流式, 双向可看全 30s) -> [B,375,Dmodel]
        -> TailQueryPool 末端 K 帧坍缩 -> tail_vec[B,Dmodel]
      text[B,L] -> frozen Qwen -> text_pool[B,1024] (全局条件)
      fusion = concat([tail_vec, text_pool]) -> [B, Dmodel+1024]
        -> main head -> 5 logits
      aux: chunk_aux_head(ch0_375 经 mid->1280 适配? 复用 audio_hidden) + future_aux_head
    """

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        audio_cfg = cfg["audio_encoder"]
        text_cfg = cfg["text_encoder"]
        ctx_cfg = cfg["context_encoder"]
        head_cfg = cfg["fusion_head"]
        lora_cfg = cfg.get("lora", {})
        chunk_cfg = cfg.get("chunk_aux", {})
        temp_cfg = cfg.get("temporal", {})

        # ---- Audio: Whisper encoder + GPU mel + LoRA (单塔, 双声道共享权重) ----
        whisper_path = audio_cfg["model_name"]
        self.audio_mel = WhisperGpuMel(whisper_path, sample_rate=int(cfg["sample_rate"]))
        _attn_impl = audio_cfg.get("attn_implementation", None)
        _wm_kwargs = {}
        if _attn_impl:
            _wm_kwargs["attn_implementation"] = str(_attn_impl)
        whisper_full = WhisperModel.from_pretrained(whisper_path, **_wm_kwargs)
        encoder = whisper_full.encoder
        del whisper_full
        self.hidden_size = int(encoder.config.d_model)  # 1280
        self.audio_encoder = wrap_whisper_lora(
            encoder,
            r=int(lora_cfg.get("r", 16)),
            alpha=int(lora_cfg.get("alpha", 32)),
            dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=tuple(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"])),
            enable_gradient_checkpointing=bool(lora_cfg.get("gradient_checkpointing", True)),
        )

        # ---- 帧级下采样 (每声道独立, 共享权重) -> 375 帧 ----
        self.frame_mid = int(temp_cfg.get("frame_mid", 320))
        self.downsampler = FrameDownsampler(
            in_dim=self.hidden_size, mid_dim=self.frame_mid,
            whisper_frames=int(chunk_cfg.get("whisper_frames", 1500)),
            target_frames=int(chunk_cfg.get("target_frames", 375)),
        )

        # ---- Context labels: 帧级 embedding (与 audio 375 帧对齐拼接) ----
        self.ctx_embed_dim = int(ctx_cfg.get("embed_dim", 16))
        self.ctx_frame_embed = nn.Embedding(int(ctx_cfg["vocab_size"]), self.ctx_embed_dim)

        # ---- FA char-timestamp grid channels (optional) ----
        self.fa_dim = int(cfg.get("fa_feats", {}).get("dim", 0))

        # ---- 帧级融合维度: [ch0, ch1, |ch0-ch1|] + ctx (+ fa) ----
        frame_fuse_in = self.frame_mid * 3 + self.ctx_embed_dim + self.fa_dim
        self.d_model = int(temp_cfg.get("d_model", 384))
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_fuse_in, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        # 帧位置编码(375 个位置, 可学习)
        self.pos_embed = nn.Parameter(torch.randn(1, int(chunk_cfg.get("target_frames", 375)), self.d_model) * 0.02)

        # ---- 时序 Transformer (双向; valid 评测非流式) ----
        n_layers = int(temp_cfg.get("layers", 2))
        n_heads = int(temp_cfg.get("heads", 6))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_heads,
            dim_feedforward=self.d_model * 4, dropout=float(temp_cfg.get("dropout", 0.1)),
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ---- 末端 query 坍缩 ----
        self.tail_pool = TailQueryPool(
            dim=self.d_model, tail_k=int(temp_cfg.get("tail_k", 50)), num_heads=n_heads,
        )

        # ---- Text: frozen Qwen3-0.6B (全局条件) ----
        self.text_encoder = TextEncoder(
            model_name=text_cfg["model_name"],
            freeze_backbone=bool(text_cfg.get("freeze_backbone", True)),
            tail_ratio=float(text_cfg.get("tail_ratio", 0.3)),
        )

        # ---- Chunk auxiliary head (复用, 吃 ch0 的 1280 维 hidden) ----
        self.use_aux = bool(chunk_cfg.get("enabled", True))
        if self.use_aux:
            self.chunk_aux_head = ChunkAuxHead(
                in_dim=self.hidden_size,
                mid_dim=int(chunk_cfg.get("mid_dim", 256)),
                num_classes=int(chunk_cfg.get("num_classes", 5)),
                whisper_frames=int(chunk_cfg.get("whisper_frames", 1500)),
                target_frames=int(chunk_cfg.get("target_frames", 375)),
            )

        # ---- Future auxiliary head (复用) ----
        future_cfg = cfg.get("future_aux", {})
        self.use_future_aux = bool(future_cfg.get("enabled", False))
        if self.use_future_aux:
            self.future_aux_head = FutureAuxHead(
                in_dim=self.hidden_size,
                mid_dim=int(future_cfg.get("mid_dim", 256)),
                num_classes=int(chunk_cfg.get("num_classes", 5)),
                horizons=tuple(future_cfg.get("horizons", [5, 10, 15, 25])),
                tail_frames=int(future_cfg.get("tail_frames", 150)),
            )

        # ---- Fusion + main head ----
        fusion_in = self.d_model + self.text_encoder.out_dim
        h1, h2 = head_cfg["hidden_dims"]
        dropout = float(head_cfg.get("dropout", 0.3))
        num_targets = len(cfg.get("labels", {}).get("multi_targets", []))
        self.num_targets = num_targets if num_targets > 0 else 5
        self.fusion_in = int(fusion_in)
        self.head = nn.Sequential(
            nn.Linear(fusion_in, h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, self.num_targets),
        )

    # ---------- audio per-channel hidden ----------
    def _encode_channels(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """waveform [B,2,T] -> (ch0_hidden, ch1_hidden) each [B,1500,1280]."""
        B, C, T = waveform.shape
        if C != 2:
            raise ValueError(f"expected dual-channel waveform, got C={C}")
        wav = waveform.reshape(B * 2, T)
        mel = self.audio_mel(wav)
        out = self.audio_encoder(input_features=mel)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hidden = hidden.view(B, 2, hidden.shape[1], hidden.shape[2])
        return hidden[:, 0], hidden[:, 1]

    def _build_fusion(self, waveform, context_labels, fa_feats=None):
        """返回 (fusion[B,fusion_in], ch0_hidden, ch1_hidden) 供主 forward 复用。"""
        ch0_hidden, ch1_hidden = self._encode_channels(waveform)   # [B,1500,1280] x2
        f0 = self.downsampler(ch0_hidden)                           # [B,375,mid]
        f1 = self.downsampler(ch1_hidden)                           # [B,375,mid]
        fdiff = torch.abs(f0 - f1)                                  # channel mismatch
        ctx = self.ctx_frame_embed(context_labels.long())          # [B,375,ctx_dim]
        # 对齐帧数(ctx 375, audio 375)
        T = min(f0.shape[1], ctx.shape[1])
        parts = [f0[:, :T], f1[:, :T], fdiff[:, :T], ctx[:, :T]]
        if self.fa_dim > 0:
            if fa_feats is None:
                fa_feats = f0.new_zeros(f0.shape[0], T, self.fa_dim)
            parts.append(fa_feats[:, :T].to(f0.dtype))
        frame = torch.cat(parts, dim=-1)
        frame = self.frame_proj(frame)                              # [B,T,d_model]
        frame = frame + self.pos_embed[:, :frame.shape[1], :]
        seq = self.temporal(frame)                                  # [B,T,d_model]
        tail_vec = self.tail_pool(seq)                              # [B,d_model]
        return tail_vec, ch0_hidden, ch1_hidden

    def forward(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_labels: torch.Tensor,
        return_aux: bool = True,
        fa_feats: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        tail_vec, ch0_hidden, ch1_hidden = self._build_fusion(waveform, context_labels, fa_feats=fa_feats)

        if return_aux and self.use_aux:
            aux_logits = self.chunk_aux_head(0.5 * (ch0_hidden + ch1_hidden))
        else:
            aux_logits = None
        if return_aux and self.use_future_aux:
            future_aux_logits = self.future_aux_head(ch0_hidden, ch1_hidden)
        else:
            future_aux_logits = None

        text_pool = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)  # [B,1024]
        fusion = torch.cat([tail_vec, text_pool], dim=-1)           # [B, fusion_in]
        main_logits = self.head(fusion)
        return main_logits, aux_logits, fusion, future_aux_logits

    def head_only(self, fusion: torch.Tensor) -> torch.Tensor:
        return self.head(fusion)