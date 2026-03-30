import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# AV 模块
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_scores = self.attn_weights(lstm_outputs).squeeze(-1)
        if lengths is not None:
            mask = torch.arange(lstm_outputs.size(1),
                                device=lstm_outputs.device)[None, :] < lengths[:, None]
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.bmm(attn_weights.unsqueeze(1), lstm_outputs).squeeze(1)


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=rnn_dropout,
        )
        self.out_dim    = hidden_dim * 2
        self.attention  = AttentionPooling(self.out_dim)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
            if lengths is None:
                lengths = torch.ones(x.size(0), dtype=torch.long, device=x.device)

        if lengths is not None:
            lengths_cpu = lengths.cpu()
            sorted_lengths, sort_idx = lengths_cpu.sort(descending=True)
            x             = x[sort_idx]
            packed        = pack_padded_sequence(x, sorted_lengths,
                                                 batch_first=True, enforce_sorted=True)
            packed_out, _ = self.rnn(packed)
            output, _     = pad_packed_sequence(packed_out, batch_first=True)
            output        = output[sort_idx.argsort()]
        else:
            output, _ = self.rnn(x)

        feat = self.attention(output, lengths)
        return self.dropout(self.layer_norm(feat))          # [B, hidden*2]


# ═══════════════════════════════════════════════════════════════════════════════
# EEG 模块（与 Student 结构完全一致）
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialAttention(nn.Module):
    def __init__(self, num_nodes: int, in_features: int):
        super().__init__()
        self.W1 = nn.Linear(in_features, in_features, bias=False)
        self.W2 = nn.Linear(in_features, in_features, bias=False)
        self.temperature = 1e-1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, N, D]
        lhs = self.W1(x)                     # [B*T, N, D]
        rhs = self.W2(x).transpose(-1, -2)   # [B*T, D, N]
        S   = torch.bmm(lhs, rhs)            # [B*T, N, N]
        return F.softmax(S / self.temperature, dim=-1)


class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 num_nodes: int, dropout: float = 0.5):
        super().__init__()
        self.W    = nn.Linear(in_features, out_features, bias=True)
        self.bn   = nn.LayerNorm(out_features)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    @staticmethod
    def _norm_adj(A: torch.Tensor) -> torch.Tensor:
        deg     = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        d_inv_s = deg.pow(-0.5)
        return d_inv_s * A * d_inv_s.transpose(-1, -2)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        out = self.W(torch.bmm(self._norm_adj(A), x))  # [B*T, N, out_features]
        out = self.bn(out)
        out = F.relu(out)
        return self.drop(out)


class SGCN(nn.Module):
    def __init__(self, num_nodes: int, in_features: int,
                 hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.spatial_attn = SpatialAttention(num_nodes, in_features)
        self.gcn1         = GraphConvLayer(in_features, hidden_dim, num_nodes, dropout)
        self.gcn2         = GraphConvLayer(hidden_dim,  out_dim,   num_nodes, dropout)
        self.num_nodes    = num_nodes
        self.out_dim      = out_dim

    def forward(self, x: torch.Tensor, pcc: torch.Tensor):
        """
        x  : [B, T, N, F]
        pcc: [B, T, N, N]
        return: h [B, T, N*out_dim],  S_out [B, T, N, N]
        """
        B, T, N, D = x.shape
        x_flat = x.reshape(B * T, N, D)

        S      = self.spatial_attn(x_flat)                        # [B*T, N, N]
        A_base = F.relu(pcc).reshape(B * T, N, N)
        A      = S * A_base + torch.eye(N, device=x.device)      # [B*T, N, N]

        h = self.gcn2(self.gcn1(x_flat, A), A)                   # [B*T, N, out_dim]

        S_out = S.view(B, T, N, N)
        return h.reshape(B, T, N * self.out_dim), S_out


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.v  = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h: torch.Tensor):
        score   = self.v(self.fc(h)).squeeze(-1)                 # [B, T]
        weights = F.softmax(score, dim=-1)                       # [B, T]
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1) # [B, H]
        return context, weights


class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size: int, lstm_hidden: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = TemporalAttention(lstm_hidden * 2)
        self.drop = nn.Dropout(dropout)

        for name, param in self.bilstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor):
        h, _ = self.bilstm(x)
        return self.attn(self.drop(h))                    # [B, H*2], [B, T]


# ═══════════════════════════════════════════════════════════════════════════════
# 分类器工厂
# ═══════════════════════════════════════════════════════════════════════════════

def build_classifier(in_dim: int, fc_hidden: int,
                     num_classes: int, dropout: float) -> nn.Sequential:
    clf = nn.Sequential(
        nn.Linear(in_dim, fc_hidden),
        nn.BatchNorm1d(fc_hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(fc_hidden, num_classes),
    )
    for m in clf.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return clf


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher Model
# ═══════════════════════════════════════════════════════════════════════════════

class TeacherModel(nn.Module):
    """
    EEG 分支结构与 Student 完全一致（SpatialAttention → SGCN → AttentionBiLSTM），
    AV 分支独立编码，两路特征 concat 后送入单一分类头。

    forward 输出（蒸馏对齐项与 Student 完全对应）：
        logits   [B, C]
        eeg_feat [B, lstm_hidden*2]   ← 对齐 Student eeg_feat
        attn_t   [B, T]               ← 对齐 Student attn_t
        S_attn   [B, T, N, N]         ← 对齐 Student S_attn
        av_ctx   [B, av_dim]
    """

    def __init__(
        self,
        # ── AV ────────────────────────────────────────────────────────────────
        audio_input_dim:  int,
        vision_input_dim: int,
        av_hidden_dim:    int   = 128,
        av_num_layers:    int   = 2,
        av_dropout:       float = 0.5,
        # ── EEG（与 Student 超参对齐）─────────────────────────────────────────
        num_nodes:        int   = 32,
        eeg_in_features:  int   = 4,
        gcn_hidden:       int   = 64,
        gcn_out:          int   = 64,
        lstm_hidden:      int   = 128,
        lstm_layers:      int   = 2,
        eeg_dropout:      float = 0.5,
        # ── 分类器 ─────────────────────────────────────────────────────────────
        fc_hidden:        int   = 256,
        num_classes:      int   = 2,
    ):
        super().__init__()

        # AV 编码器
        self.audio_encoder  = ModalityEncoder(audio_input_dim,  av_hidden_dim,
                                              av_num_layers, av_dropout)
        self.vision_encoder = ModalityEncoder(vision_input_dim, av_hidden_dim,
                                              av_num_layers, av_dropout)
        self.av_dim = self.audio_encoder.out_dim + self.vision_encoder.out_dim

        # EEG 编码器（与 Student 结构完全一致）
        self.sgcn = SGCN(
            num_nodes=num_nodes, in_features=eeg_in_features,
            hidden_dim=gcn_hidden, out_dim=gcn_out, dropout=eeg_dropout,
        )
        self.abilstm = AttentionBiLSTM(
            input_size=num_nodes * gcn_out,
            lstm_hidden=lstm_hidden,
            num_layers=lstm_layers,
            dropout=eeg_dropout,
        )
        self.eeg_dim = lstm_hidden * 2

        # 单一分类头：concat(eeg_feat, av_ctx) → logits
        self.classifier = build_classifier(
            in_dim=self.eeg_dim + self.av_dim,
            fc_hidden=fc_hidden,
            num_classes=num_classes,
            dropout=max(av_dropout, eeg_dropout),
        )

    def forward(self, eeg, pcc, audio, vision,
                audio_lengths=None, vision_lengths=None):
        # AV 编码
        a_feat = self.audio_encoder(audio,  audio_lengths)    # [B, av_dim/2]
        v_feat = self.vision_encoder(vision, vision_lengths)   # [B, av_dim/2]
        av_ctx = torch.cat([a_feat, v_feat], dim=-1)           # [B, av_dim]

        # EEG 编码（流程与 Student 完全一致）
        gcn_out, S_attn  = self.sgcn(eeg, pcc)                # [B,T,N*gcn_out], [B,T,N,N]
        eeg_feat, attn_t = self.abilstm(gcn_out)              # [B, eeg_dim], [B, T]

        # Concat → 单头分类
        fused  = torch.cat([eeg_feat, av_ctx], dim=-1)        # [B, eeg_dim+av_dim]
        logits = self.classifier(fused)                       # [B, C]

        return {
            'logits'  : logits,
            'eeg_feat': eeg_feat,
            'attn_t'  : attn_t,
            'S_attn'  : S_attn,
            'av_ctx'  : av_ctx,
            'fused'   : fused,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher 训练损失
# ═══════════════════════════════════════════════════════════════════════════════

class TeacherLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs: dict, labels: torch.Tensor) -> dict:
        loss = self.ce(outputs['logits'], labels)
        return {'loss': loss}


# ═══════════════════════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    B, T, N, F = 4, 5, 30, 4
    T_a, F_a   = 10, 40
    T_v, F_v   = 10, 512

    model = TeacherModel(
        audio_input_dim=F_a, vision_input_dim=F_v,
        num_nodes=N, eeg_in_features=F, num_classes=2,
    )

    eeg    = torch.randn(B, T, N, F)
    pcc    = torch.randn(B, T, N, N)
    audio  = torch.randn(B, T_a, F_a)
    vision = torch.randn(B, T_v, F_v)
    labels = torch.randint(0, 2, (B,))

    out     = model(eeg, pcc, audio, vision)
    loss_fn = TeacherLoss()
    loss    = loss_fn(out, labels)

    print(f"logits   : {out['logits'].shape}")      # [4, 2]
    print(f"eeg_feat : {out['eeg_feat'].shape}")    # [4, 256]
    print(f"attn_t   : {out['attn_t'].shape}")      # [4, 5]
    print(f"S_attn   : {out['S_attn'].shape}")      # [4, 5, 30, 30]
    print(f"av_ctx   : {out['av_ctx'].shape}")      # [4, 512]
    print(f"loss     : {loss['loss'].item():.4f}")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params   : {total:,}")