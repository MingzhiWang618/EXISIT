import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Audio 模块
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
    """单模态 BiLSTM + 注意力池化编码器（audio）"""
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
        return self.dropout(self.layer_norm(feat))   # [B, hidden*2]


# ═══════════════════════════════════════════════════════════════════════════════
# EEG 模块（结构与原版完全一致，仅 av_dim → audio_dim）
# ═══════════════════════════════════════════════════════════════════════════════

class AVCrossAttention(nn.Module):
    def __init__(self, av_dim: int, eeg_dim: int, dk: int = 64):
        super().__init__()
        self.Wq    = nn.Linear(av_dim,  dk, bias=False)
        self.Wk    = nn.Linear(eeg_dim, dk, bias=False)
        self.scale = dk ** -0.5
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)

    def forward(self, av: torch.Tensor, eeg_seq: torch.Tensor) -> torch.Tensor:
        """
        av      : [B*T, av_dim]
        eeg_seq : [B*T, N, eeg_dim]
        return  : [B*T, N]
        """
        q     = self.Wq(av).unsqueeze(1)
        k     = self.Wk(eeg_seq)
        score = torch.bmm(q, k.transpose(1, 2)) * self.scale
        return score.squeeze(1)


class SpatialAttention(nn.Module):
    def __init__(self, in_features: int, av_dim: int, dk: int = 64):
        super().__init__()
        self.Wq          = nn.Linear(in_features, dk, bias=False)
        self.Wk          = nn.Linear(in_features, dk, bias=False)
        self.temperature = 1
        self.av_cross_attn = AVCrossAttention(av_dim, in_features, dk)
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)

    def forward(self, x: torch.Tensor, av_bt: torch.Tensor) -> torch.Tensor:
        Q     = self.Wq(x)
        K     = self.Wk(x)
        S     = torch.bmm(Q, K.transpose(1, 2))
        alpha = self.av_cross_attn(av_bt, x)
        S     = S + alpha.unsqueeze(1)
        return F.softmax(S / self.temperature, dim=-1)


class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 num_nodes: int, dropout: float = 0.5):
        super().__init__()
        self.W    = nn.Linear(in_features, out_features, bias=True)
        self.bn   = nn.BatchNorm1d(out_features)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    @staticmethod
    def _norm_adj(A: torch.Tensor) -> torch.Tensor:
        deg     = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        d_inv_s = deg.pow(-0.5)
        return d_inv_s * A * d_inv_s.transpose(-1, -2)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        out = self.W(torch.bmm(self._norm_adj(A), x))
        out = out.transpose(1, 2)
        out = self.bn(out)
        out = out.transpose(1, 2)
        out = F.relu(out)
        return self.drop(out)


class SGCN(nn.Module):
    def __init__(self, num_nodes: int, in_features: int,
                 hidden_dim: int, out_dim: int,
                 av_dim: int, dk: int = 64, dropout: float = 0.5):
        super().__init__()
        self.spatial_attn = SpatialAttention(in_features, av_dim, dk)
        self.gcn1         = GraphConvLayer(in_features, hidden_dim, num_nodes, dropout)
        self.gcn2         = GraphConvLayer(hidden_dim,  out_dim,   num_nodes, dropout)
        self.num_nodes    = num_nodes
        self.out_dim      = out_dim

    def forward(self, x: torch.Tensor, pcc: torch.Tensor,
                av_ctx: torch.Tensor):
        B, T, N, D = x.shape
        x_flat = x.reshape(B * T, N, D)
        av_bt  = av_ctx.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)

        S      = self.spatial_attn(x_flat, av_bt)
        A_base = F.relu(pcc).reshape(B * T, N, N)
        A      = S * A_base + torch.eye(N, device=x.device).unsqueeze(0)

        h      = self.gcn2(self.gcn1(x_flat, A), A)
        S_out  = S.view(B, T, N, N)
        return h.reshape(B, T, N * self.out_dim), S_out


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int, av_dim: int, dk: int = 64):
        super().__init__()
        self.fc            = nn.Linear(hidden_size, hidden_size)
        self.v             = nn.Linear(hidden_size, 1, bias=False)
        self.av_cross_attn = AVCrossAttention(av_dim, hidden_size, dk)

    def forward(self, h: torch.Tensor, av_ctx: torch.Tensor):
        e       = self.v(self.fc(h)).squeeze(-1)
        beta    = self.av_cross_attn(av_ctx, h)
        weights = F.softmax(e + beta, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        return context, weights


class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size: int, lstm_hidden: int,
                 num_layers: int = 2, dropout: float = 0.5,
                 av_dim: int = 128, dk: int = 64):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size, hidden_size=lstm_hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = TemporalAttention(lstm_hidden * 2, av_dim, dk)
        self.drop = nn.Dropout(dropout)

        for name, param in self.bilstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor, av_ctx: torch.Tensor):
        h, _ = self.bilstm(x)
        return self.attn(self.drop(h), av_ctx)


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
# Teacher Model（单 Audio 指导 EEG）
# ═══════════════════════════════════════════════════════════════════════════════

class TeacherModel(nn.Module):
    """
    原版双模态（Audio + Vision）→ 单模态（Audio only）。

    改动点（仅此一处）：
        删除 vision_encoder
        av_ctx = a_feat   （原来是 cat([a_feat, v_feat])）
        av_dim = audio_encoder.out_dim  （原来是 out_dim * 2）

    其余 SGCN / AttentionBiLSTM / 分类器 / 损失 结构完全不变。

    两路分类头：
        audio_logits : [B, C]   Audio 分支
        eeg_logits   : [B, C]   EEG 分支（Audio 指导空间/时序注意力）

    蒸馏输出：
        eeg_feat  [B, lstm_hidden*2]
        attn_t    [B, T]
        S_attn    [B, T, N, N]
        av_ctx    [B, audio_dim]
    """

    def __init__(
        self,
        # ── Audio ─────────────────────────────────────────────────────────────
        audio_input_dim:  int,
        audio_hidden_dim: int   = 128,
        audio_num_layers: int   = 2,
        audio_dropout:    float = 0.5,
        # ── EEG ───────────────────────────────────────────────────────────────
        num_nodes:        int   = 8,
        eeg_in_features:  int   = 5,
        gcn_hidden:       int   = 64,
        gcn_out:          int   = 64,
        lstm_hidden:      int   = 128,
        lstm_layers:      int   = 2,
        eeg_dropout:      float = 0.5,
        dk:               int   = 64,
        # ── 分类器 ─────────────────────────────────────────────────────────────
        fc_hidden:        int   = 256,
        num_classes:      int   = 7,
    ):
        super().__init__()

        # ── Audio 编码器（原来的双编码器，现在只保留 audio） ──────────────────
        self.audio_encoder = ModalityEncoder(
            input_dim  = audio_input_dim,
            hidden_dim = audio_hidden_dim,
            num_layers = audio_num_layers,
            dropout    = audio_dropout,
        )
        # av_ctx 直接是 audio 特征，维度 = hidden_dim * 2
        self.av_dim = self.audio_encoder.out_dim   # ← 原来是 out_dim * 2

        # ── EEG 编码器（完全不变，只是 av_dim 变小了）────────────────────────
        self.sgcn = SGCN(
            num_nodes=num_nodes, in_features=eeg_in_features,
            hidden_dim=gcn_hidden, out_dim=gcn_out,
            av_dim=self.av_dim, dk=dk, dropout=eeg_dropout,
        )
        self.abilstm = AttentionBiLSTM(
            input_size  = num_nodes * gcn_out,
            lstm_hidden = lstm_hidden,
            num_layers  = lstm_layers,
            dropout     = eeg_dropout,
            av_dim      = self.av_dim,
            dk          = dk,
        )
        self.eeg_dim = lstm_hidden * 2

        # ── 两路分类头 ────────────────────────────────────────────────────────
        self.audio_classifier = build_classifier(
            self.av_dim, fc_hidden, num_classes, audio_dropout)
        self.eeg_classifier   = build_classifier(
            self.eeg_dim, fc_hidden, num_classes, eeg_dropout)

    def forward(
        self,
        eeg:           torch.Tensor,
        pcc:           torch.Tensor,
        audio:         torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        eeg   : [B, T, N, F]
        pcc   : [B, T, N, N]
        audio : [B, T_a, F_a]
        """
        # Audio 编码 → av_ctx（原来 cat([a,v])，现在直接是 a_feat）
        av_ctx = self.audio_encoder(audio, audio_lengths)  # [B, av_dim]

        # EEG 编码（结构完全不变）
        gcn_out, S_attn  = self.sgcn(eeg, pcc, av_ctx)
        eeg_feat, attn_t = self.abilstm(gcn_out, av_ctx)

        audio_logits = self.audio_classifier(av_ctx)
        eeg_logits   = self.eeg_classifier(eeg_feat)

        return {
            'audio_logits': audio_logits,   # 原 av_logits
            'eeg_logits':   eeg_logits,
            'eeg_feat':     eeg_feat,
            'attn_t':       attn_t,
            'S_attn':       S_attn,
            'av_ctx':       av_ctx,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher 训练损失（不变）
# ═══════════════════════════════════════════════════════════════════════════════

class TeacherLoss(nn.Module):
    def __init__(self, w_audio: float = 0.4, w_eeg: float = 0.6):
        super().__init__()
        self.w_audio = w_audio
        self.w_eeg   = w_eeg
        self.ce      = nn.CrossEntropyLoss()

    def forward(self, outputs: dict, labels: torch.Tensor) -> dict:
        l_audio = self.ce(outputs['audio_logits'], labels)
        l_eeg   = self.ce(outputs['eeg_logits'],   labels)
        total   = self.w_audio * l_audio + self.w_eeg * l_eeg
        return {'loss': total, 'loss_audio': l_audio, 'loss_eeg': l_eeg}


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    B, T, N, F = 4, 5, 8, 5     # PME4: 8 电极, 5 频带
    T_a, F_a   = 296, 25         # PME4 opensmile: [296, 25]

    model = TeacherModel(
        audio_input_dim  = F_a,
        audio_hidden_dim = 128,
        num_nodes        = N,
        eeg_in_features  = F,
        num_classes      = 7,
    )

    eeg    = torch.randn(B, T, N, F)
    pcc    = torch.randn(B, T, N, N)
    audio  = torch.randn(B, T_a, F_a)
    labels = torch.randint(0, 7, (B,))

    out     = model(eeg, pcc, audio)
    loss_fn = TeacherLoss()
    losses  = loss_fn(out, labels)

    print(f"audio_logits : {out['audio_logits'].shape}")   # [4, 7]
    print(f"eeg_logits   : {out['eeg_logits'].shape}")     # [4, 7]
    print(f"eeg_feat     : {out['eeg_feat'].shape}")       # [4, 256]
    print(f"attn_t       : {out['attn_t'].shape}")         # [4, 5]
    print(f"S_attn       : {out['S_attn'].shape}")         # [4, 5, 8, 8]
    print(f"loss         : {losses['loss'].item():.4f}")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params       : {total:,}")