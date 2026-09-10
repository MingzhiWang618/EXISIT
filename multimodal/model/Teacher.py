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
# EEG 模块
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
        return  : [B*T, N]   per-node AV-guided score
        """
        q     = self.Wq(av).unsqueeze(1)                  # [B*T, 1, dk]
        k     = self.Wk(eeg_seq)                          # [B*T, N, dk]
        score = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B*T, 1, N]
        return score.squeeze(1)                            # [B*T, N]


class SpatialAttention(nn.Module):
    """
    在特征空间做 Q-K 内积得到 [B*T, N, N] 的节点关系矩阵，
    再用 AV cross-attention 得到每个节点的标量偏置 alpha [B*T, N]，
    叠加到矩阵的行上后做 softmax。

    与 Student 的区别：Student 直接用两个不同线性变换做内积；
    Teacher 在此基础上额外引入 AV 指导的行偏置，使空间图感知到
    音视频语义信息。
    """
    def __init__(self, in_features: int, av_dim: int, dk: int = 64):
        super().__init__()
        # Q-K 投影：与 Student 结构对齐，但映射到 dk 维以控制参数量
        self.Wq = nn.Linear(in_features, dk, bias=False)
        self.Wk = nn.Linear(in_features, dk, bias=False)
        self.temperature = 1e-1
        
        # AV 引导的行偏置
        self.av_cross_attn = AVCrossAttention(av_dim, in_features, dk)

        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)

    def forward(self, x: torch.Tensor, av_bt: torch.Tensor) -> torch.Tensor:
        """
        x     : [B*T, N, F]
        av_bt : [B*T, av_dim]
        return: [B*T, N, N]  softmax 后的空间注意力矩阵
        """
        Q = self.Wq(x)                                    # [B*T, N, dk]
        K = self.Wk(x)                                    # [B*T, N, dk]
        S = torch.bmm(Q, K.transpose(1, 2))  # [B*T, N, N]

        # alpha [B*T, N] → 作为行偏置广播到 [B*T, N, N]
        alpha = self.av_cross_attn(av_bt, x)              # [B*T, N]
        S = S + alpha.unsqueeze(1)                        # [B*T, N, N]

        return F.softmax(S / self.temperature, dim=-1)                       # [B*T, N, N]


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
        # x: [B*T, N, F_in]  A: [B*T, N, N]
        out = self.W(torch.bmm(self._norm_adj(A), x))    # [B*T, N, F]
        out = out.transpose(1, 2)                         # [B*T, F, N]
        out = self.bn(out)
        out = out.transpose(1, 2)                         # [B*T, N, F]
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
        """
        x     : [B, T, N, F]
        pcc   : [B, T, N, N]
        av_ctx: [B, av_dim]
        """
        B, T, N, D = x.shape
        x_flat = x.reshape(B * T, N, D)
        av_bt  = av_ctx.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)

        S = self.spatial_attn(x_flat, av_bt)              # [B*T, N, N]
        A_base = F.relu(pcc).reshape(B * T, N, N)  
        A      = S * A_base
        A      = A + torch.eye(N, device=x.device).unsqueeze(0)

        h = self.gcn2(self.gcn1(x_flat, A), A)           # [B*T, N, out_dim]

        S_out = S.view(B, T, N, N)                        # [B, T, N, N]
        return h.reshape(B, T, N * self.out_dim), S_out


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int, av_dim: int, dk: int = 64):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.v  = nn.Linear(hidden_size, 1, bias=False)
        self.av_cross_attn = AVCrossAttention(av_dim, hidden_size, dk)

    def forward(self, h: torch.Tensor, av_ctx: torch.Tensor):
        e       = self.v(torch.tanh(self.fc(h))).squeeze(-1)          # [B, T]
        beta    = self.av_cross_attn(av_ctx, h)           # [B, T]
        weights = F.softmax(e + beta, dim=-1)             # [B, T]
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # [B, 2H]
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
        context, weights = self.attn(h, av_ctx)
        return self.drop(context), weights                # stable attention; dropout only on representation


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
    两路独立分类头：
        AV  分支  →  av_logits  [B, C]
        EEG 分支  →  eeg_logits [B, C]  ← AV 指导空间/时序注意力

    蒸馏输出：
        eeg_feat  [B, lstm_hidden*2]   Student 特征层对齐
        attn_t    [B, T]               Student 时序注意力对齐
        S_attn    [B, T, N, N]         Student 空间注意力对齐
    """

    def __init__(
        self,
        # ── AV ────────────────────────────────────────────────────────────────
        audio_input_dim:  int,
        vision_input_dim: int,
        av_hidden_dim:    int   = 128,
        av_num_layers:    int   = 2,
        av_dropout:       float = 0.5,
        # ── EEG ───────────────────────────────────────────────────────────────
        num_nodes:        int   = 32,
        eeg_in_features:  int   = 4,
        gcn_hidden:       int   = 64,
        gcn_out:          int   = 64,
        lstm_hidden:      int   = 128,
        lstm_layers:      int   = 2,
        eeg_dropout:      float = 0.5,
        dk:               int   = 64,
        # ── 分类器 ─────────────────────────────────────────────────────────────
        fc_hidden:        int   = 256,
        num_classes:      int   = 2,
    ):
        super().__init__()

        self.audio_encoder  = ModalityEncoder(audio_input_dim,  av_hidden_dim,
                                              av_num_layers, av_dropout)
        self.vision_encoder = ModalityEncoder(vision_input_dim, av_hidden_dim,
                                              av_num_layers, av_dropout)
        self.av_dim = self.audio_encoder.out_dim + self.vision_encoder.out_dim

        self.sgcn = SGCN(
            num_nodes=num_nodes, in_features=eeg_in_features,
            hidden_dim=gcn_hidden, out_dim=gcn_out,
            av_dim=self.av_dim, dk=dk, dropout=eeg_dropout,
        )
        self.abilstm = AttentionBiLSTM(
            input_size=num_nodes * gcn_out,
            lstm_hidden=lstm_hidden, num_layers=lstm_layers,
            dropout=eeg_dropout, av_dim=self.av_dim, dk=dk,
        )
        self.eeg_dim = lstm_hidden * 2

        self.av_classifier  = build_classifier(self.av_dim,  fc_hidden,
                                               num_classes, av_dropout)
        self.eeg_classifier = build_classifier(self.eeg_dim, fc_hidden,
                                               num_classes, eeg_dropout)

    def forward(self, eeg, pcc, audio, vision,
                audio_lengths=None, vision_lengths=None):
        a_feat = self.audio_encoder(audio,  audio_lengths)
        v_feat = self.vision_encoder(vision, vision_lengths)
        av_ctx = torch.cat([a_feat, v_feat], dim=-1)

        gcn_out, S_attn  = self.sgcn(eeg, pcc, av_ctx)
        eeg_feat, attn_t = self.abilstm(gcn_out, av_ctx)

        av_logits  = self.av_classifier(av_ctx)
        eeg_logits = self.eeg_classifier(eeg_feat)
        return {
            'av_logits' : av_logits,
            'eeg_logits': eeg_logits,
            'eeg_feat'  : eeg_feat,
            'attn_t'    : attn_t,
            'S_attn'    : S_attn,
            'av_ctx'    : av_ctx,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher 训练损失
# ═══════════════════════════════════════════════════════════════════════════════

class TeacherLoss(nn.Module):
    def __init__(self, w_av: float = 0.4, w_eeg: float = 0.6):
        super().__init__()
        self.w_av  = w_av
        self.w_eeg = w_eeg
        self.ce    = nn.CrossEntropyLoss()

    def forward(self, outputs: dict, labels: torch.Tensor) -> dict:
        l_av  = self.ce(outputs['av_logits'],  labels)
        l_eeg = self.ce(outputs['eeg_logits'], labels)
        total = self.w_av * l_av + self.w_eeg * l_eeg
        return {'loss': total, 'loss_av': l_av, 'loss_eeg': l_eeg}


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

    print(f"av_logits  : {out['av_logits'].shape}")    # [4, 2]
    print(f"eeg_logits : {out['eeg_logits'].shape}")   # [4, 2]
    print(f"eeg_feat   : {out['eeg_feat'].shape}")     # [4, 256]
    print(f"attn_t     : {out['attn_t'].shape}")       # [4, 5]
    print(f"S_attn     : {out['S_attn'].shape}")       # [4, 5, 30, 30]
    print(f"loss       : {loss['loss'].item():.4f}")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params     : {total:,}")
