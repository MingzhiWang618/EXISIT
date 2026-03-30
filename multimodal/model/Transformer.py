import torch
import torch.nn as nn
import torch.nn.functional as F
from archieve.single_modality.EEGFormer import EEGFormer


class MultimodalEmotionRecognitionWithAttention(nn.Module):
    """
    - 每个模态先线性映射到 768 维。
    - EEG 先经过 BiGRU（可处理 (B, T, F) 或 (B, F)），再线性到 768。
    - 将每个模态视为一个 token -> (B, num_modalities, 768) -> TransformerEncoder (batch_first=True)
    - Transformer 输出在 token 维度上做 mean pooling -> (B, 768) -> 投影到 512 -> 分类头
    """
    def __init__(self, num_classes, modality_config, feature_dims,
                 transformer_heads=8, transformer_layers=4, transformer_ff=2048, dropout=0.1):
        super().__init__()
        self.modality_config = modality_config
        self.feature_dims = feature_dims
        self.eeg_backbone = EEGFormer(eeg_channel=30)
        # per-modality projection to 768
        if modality_config.get('vision', False):
            self.vision_proj = nn.Sequential()
            self.vision_proj.add_module('project_t', nn.Linear(feature_dims['vision'], 768))
            self.vision_proj.add_module('project_t_layer_norm', nn.LayerNorm(768))
        else:
            self.vision_proj = None

        if modality_config.get('audio', False):
            self.audio_proj = nn.Sequential()
            self.audio_proj.add_module('project_t', nn.Linear(feature_dims['audio'], 768))
            self.audio_proj.add_module('project_t_layer_norm', nn.LayerNorm(768))
        else:
            self.audio_proj = None

        if modality_config.get('eeg', False):
            
            # BiGRU to process temporal EEG -> hidden size h, bidirectional -> output dim 2*h
            # 然后再投影到 768
            self.eeg_gru_hidden = 384  # 2*384 = 768 after bidirectional concat if we use hidden->we'll pool
            # We will use batch_first GRU expecting (B, T, F)
            self.eeg_bigru = nn.GRU(
                input_size=feature_dims['eeg'],
                hidden_size=self.eeg_gru_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            # If we want to reduce GRU outputs, we'll do mean over time on the BiGRU outputs (B, T, 2*h)
            self.eeg_proj = nn.Sequential()
            self.eeg_proj.add_module('project_t', nn.Linear(self.eeg_gru_hidden*2, 768))
            self.eeg_proj.add_module('project_t_layer_norm', nn.LayerNorm(768))
        else:
            self.eeg_bigru = None
            self.eeg_proj = None

        # Transformer-based fusion: embed_dim = 768 (per token)
        self.transformer_fusion = TransformerFusion(
            embed_dim=768,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff,
            dropout=dropout
        )

        # Classification head (输入是 transformer 投影到 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, *features):
        """
        features 顺序必须与 modality_config 中的 (vision, audio, eeg) 一致并且只传启用的模态。
        - vision, audio: tensors with shape (B, feat_dim)
        - eeg: either (B, T, feat_dim) or (B, feat_dim) (handled automatically)
        """
        tokens = []  # collect per-modality tokens (B, 768)

        idx = 0
        # vision
        if self.modality_config.get('vision', False):
            vision_feat = features[idx]; idx += 1
            # expect (B, feat_dim)
            v_tok = self.vision_proj(vision_feat)  # (B, 768)
            tokens.append(v_tok.unsqueeze(1))  # (B, 1, 768)

        # audio
        if self.modality_config.get('audio', False):
            audio_feat = features[idx]; idx += 1
            a_tok = self.audio_proj(audio_feat)  # (B, 768)
            tokens.append(a_tok.unsqueeze(1))

        # eeg
        if self.modality_config.get('eeg', False):
            eeg_feat = features[idx]; idx += 1
            # if eeg_feat is (B, F), add time dim
            if eeg_feat.dim() == 2:
                eeg_in = eeg_feat.unsqueeze(1)  # (B, 1, F)
            elif eeg_feat.dim() == 3:
                eeg_in = eeg_feat  # (B, T, F)
            else:
                raise ValueError("EEG input must be 2D (B, F) or 3D (B, T, F)")

            # BiGRU -> outputs (B, T, 2*h). We'll pool over time (mean) to get (B, 2*h)
            eeg_in = self.eeg_backbone(eeg_in)
            gru_out, _ = self.eeg_bigru(eeg_in)  # (B, T, 2*h)
            
            # mean over time (T)
            pooled = gru_out.mean(dim=1)  # (B, 2*h)
            e_tok = self.eeg_proj(pooled)  # (B, 768)
            tokens.append(e_tok.unsqueeze(1))

        # stack tokens -> (B, num_modalities, 768)
        fused_seq = torch.cat(tokens, dim=1)  # sequence of modality tokens

        # Transformer fusion expects (B, seq, embed_dim)
        fused_output = self.transformer_fusion(fused_seq)  # returns (B, 512) in our TransformerFusion design

        # classification head
        x = self.relu(self.fc1(fused_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerFusion(nn.Module):
    """
    Transformer encoder stack for modality-token fusion.
    Input: (B, seq, embed_dim) where embed_dim should be 768
    Output: (B, 512)  -- after mean pooling over seq and linear projection + layernorm
    """
    def __init__(self, embed_dim, num_heads=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(embed_dim, 512)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, x):
        """
        x: (B, seq, embed_dim)
        returns: (B, 512)
        """
        # transformer encoding (B, seq, embed_dim)
        enc = self.transformer_encoder(x)

        # pooling across tokens (modalities) -- simple mean pooling
        pooled = enc.mean(dim=1)  # (B, embed_dim)

        out = self.output_projection(pooled)  # (B, 512)
        out = self.layer_norm(out)
        return out
