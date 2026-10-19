from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_dualch import DualChannelWhisperAudioEncoder


class TemporalAlignedModel(nn.Module):
    """
    【破局证明版】严格物理对齐的端到端时序大一统模型 (Temporal Aligned Engine)
    
    1. 抛弃 Text：纯净的 Audio-Context 交响。
    2. 抛弃 AttentionPooling：拒绝时序粉碎。
    3. 严格的 Master Clock 对齐：Whisper 1500帧 强行对齐 Context 的 375帧。
    4. 时序因果推断：用 GRU 扫描这 30秒，用最后一帧 (t=0) 的极限边缘去预测未来 2s。
    """
    def __init__(self, cfg: Dict):
        super().__init__()
        audio_cfg = cfg["audio_encoder"]
        # Hack: clear out unused parameters from DualChannelWhisperAudioEncoder to prevent DDP crash
        class DummyExtractor(nn.Module):
            def __init__(self, e):
                super().__init__()
                self.encoder = e.encoder
                self._gpu_log_mel = e._gpu_log_mel
                self._build_input_features_dual = e._build_input_features_dual
                self._encode_dual = e._encode_dual
                self.hidden_size = e.hidden_size
                self.freeze = e.freeze

        _e = DualChannelWhisperAudioEncoder(
            model_name=audio_cfg["model_name"],
            sample_rate=cfg["sample_rate"],
            proj_dim=int(audio_cfg["proj_dim"]),
            freeze=bool(audio_cfg.get("freeze", True)),
            use_cross_attn=False,
        )
        self.audio_extractor = DummyExtractor(_e)
        whisper_dim = self.audio_extractor.hidden_size
        
        ctx_cfg = cfg["context_encoder"]
        self.ctx_embed = nn.Embedding(ctx_cfg["vocab_size"], ctx_cfg["embed_dim"])
        
        self.audio_conv = nn.Conv1d(whisper_dim * 2, 512, kernel_size=4, stride=4)
        self.audio_norm = nn.LayerNorm(512)
        self.audio_gelu = nn.GELU()
        
        fusion_dim = 512 + ctx_cfg["embed_dim"]
        gru_hidden = 256
        self.temporal_engine = nn.GRU(
            input_size=fusion_dim, 
            hidden_size=gru_hidden, 
            num_layers=1, 
            batch_first=True,
            bidirectional=False 
        )
        
        num_targets = len(cfg.get("labels", {}).get("multi_targets", []))
        self.num_targets = num_targets if num_targets > 0 else 1
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_targets)
        )

    def forward(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_labels: torch.Tensor,
    ) -> torch.Tensor:
        B = waveform.shape[0]
        
        hidden_ch0, hidden_ch1 = self.audio_extractor._encode_dual(waveform) 
        audio_cat = torch.cat([hidden_ch0, hidden_ch1], dim=-1).transpose(1, 2) 
        
        aligned_audio = self.audio_conv(audio_cat)
        aligned_audio = aligned_audio.transpose(1, 2) 
        aligned_audio = self.audio_norm(aligned_audio)
        aligned_audio = self.audio_gelu(aligned_audio)
        
        expected_len = context_labels.shape[1]
        if aligned_audio.shape[1] < expected_len:
            pad = expected_len - aligned_audio.shape[1]
            aligned_audio = F.pad(aligned_audio, (0, 0, 0, pad)) 
        elif aligned_audio.shape[1] > expected_len:
            aligned_audio = aligned_audio[:, :expected_len, :]
            
        ctx_feat = self.ctx_embed(context_labels.long())
        fused_seq = torch.cat([aligned_audio, ctx_feat], dim=-1)
        gru_out, _ = self.temporal_engine(fused_seq)
        last_frame_state = gru_out[:, -1, :] 
        
        logits = self.head(last_frame_state)
        
        if self.num_targets == 1:
            return logits.squeeze(-1)
        return logits


class MimiTemporalAlignedModel(nn.Module):
    """
    【Mimi时空对齐版】严格物理对齐的端到端时序大一统模型 (Mimi Temporal Aligned Engine)
    
    1. 抛弃 Text：纯净的 Audio-Context 交响。
    2. 丢弃 Conv1D：Mimi 原生 12.5Hz (80ms/帧) 提取特征，直接与 Context 的 375帧 0损像素级对齐！
    3. 全放开微调：Mimi 仅 77M，全参数放开 (freeze: False)，彻底攻克双声道物理盲区。
    4. 时序因果推断：用 GRU 扫描 30秒，用最后一帧的极限边缘预测未来 2s。
    """
    def __init__(self, cfg: Dict):
        super().__init__()
        audio_cfg = cfg["audio_encoder"]
        self.freeze_audio = bool(audio_cfg.get("freeze", False))
        
        # 1. 载入 Mimi 77M Model
        from transformers import MimiModel
        self.mimi = MimiModel.from_pretrained(audio_cfg["model_name"])
        
        # 根据配置决定是否冻结音频编码器
        for p in self.mimi.parameters():
            p.requires_grad = not self.freeze_audio
            
        mimi_dim = 512 # Mimi hidden dimension is 512
        
        ctx_cfg = cfg["context_encoder"]
        self.ctx_embed = nn.Embedding(ctx_cfg["vocab_size"], ctx_cfg["embed_dim"])
        
        # 不需要 Conv1d 做 Stride 下采样，因为 Mimi 出来就是完美的 375 帧！
        self.audio_norm = nn.LayerNorm(mimi_dim * 2) # 双声道拼接
        self.audio_proj = nn.Sequential(
            nn.Linear(mimi_dim * 2, 512),
            nn.GELU(),
            nn.LayerNorm(512)
        )
        
        fusion_dim = 512 + ctx_cfg["embed_dim"]
        gru_hidden = 256
        self.temporal_engine = nn.GRU(
            input_size=fusion_dim, 
            hidden_size=gru_hidden, 
            num_layers=1, 
            batch_first=True,
            bidirectional=False 
        )
        
        num_targets = len(cfg.get("labels", {}).get("multi_targets", []))
        self.num_targets = num_targets if num_targets > 0 else 1
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_targets)
        )

    def forward(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_labels: torch.Tensor,
    ) -> torch.Tensor:
        # waveform shape: [B, 2, T_wav]
        B = waveform.shape[0]
        
        # 1. 拆分双声道并分别喂入 Mimi
        waveform_ch0 = waveform[:, 0:1, :] # [B, 1, T_wav]
        waveform_ch1 = waveform[:, 1:2, :] # [B, 1, T_wav]
        
        # 仅在模型处于训练状态且音频编码器未冻结时，才启用梯度计算
        is_grad_enabled = self.training and (not self.freeze_audio)
        context_manager = torch.enable_grad() if is_grad_enabled else torch.no_grad()
        
        with context_manager:
            # Step 1.1: mimi.encoder converts T_wav to T_enc (e.g. 750)
            x_ch0 = self.mimi.encoder(waveform_ch0) # [B, 512, 750]
            # Step 1.2: mimi.encoder_transformer processes temporal dependencies
            tf_out_ch0 = self.mimi.encoder_transformer(x_ch0.transpose(1, 2))
            # Step 1.3: mimi.downsample downsamples 2x to T_mimi (e.g. 375)
            feat_ch0 = self.mimi.downsample(tf_out_ch0.last_hidden_state.transpose(1, 2)) # [B, 512, 375]
            
            x_ch1 = self.mimi.encoder(waveform_ch1) # [B, 512, 750]
            tf_out_ch1 = self.mimi.encoder_transformer(x_ch1.transpose(1, 2))
            feat_ch1 = self.mimi.downsample(tf_out_ch1.last_hidden_state.transpose(1, 2)) # [B, 512, 375]
            
        # 2. 拼接双声道并在时间维度转置
        audio_cat = torch.cat([feat_ch0, feat_ch1], dim=1).transpose(1, 2) # [B, 375, 1024]
        
        # 3. 投影到 512 维
        aligned_audio = self.audio_norm(audio_cat)
        aligned_audio = self.audio_proj(aligned_audio) # [B, 375, 512]
        
        # 4. 强制对齐到 expected_len
        expected_len = context_labels.shape[1]
        if aligned_audio.shape[1] < expected_len:
            pad = expected_len - aligned_audio.shape[1]
            aligned_audio = F.pad(aligned_audio, (0, 0, 0, pad)) 
        elif aligned_audio.shape[1] > expected_len:
            aligned_audio = aligned_audio[:, :expected_len, :]
            
        # 5. 拼接 Context 标签并送入 GRU 时序引擎
        ctx_feat = self.ctx_embed(context_labels.long())
        fused_seq = torch.cat([aligned_audio, ctx_feat], dim=-1)
        gru_out, _ = self.temporal_engine(fused_seq)
        last_frame_state = gru_out[:, -1, :] 
        
        logits = self.head(last_frame_state)
        
        if self.num_targets == 1:
            return logits.squeeze(-1)
        return logits
