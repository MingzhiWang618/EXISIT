# Teacher_proto.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# AV 模块（与原 Teacher 完全一致）
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs, lengths=None):
        attn_scores = self.attn_weights(lstm_outputs).squeeze(-1)
        if lengths is not None:
            mask = torch.arange(lstm_outputs.size(1),
                                device=lstm_outputs.device)[None, :] < lengths[:, None]
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.bmm(attn_weights.unsqueeze(1), lstm_outputs).squeeze(1)


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.5):
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

    def forward(self, x, lengths=None):
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
        return self.dropout(self.layer_norm(feat))


# ═══════════════════════════════════════════════════════════════════════════════
# EEG 模块（与原 Teacher / Student 完全一致）
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialAttention(nn.Module):
    def __init__(self, num_nodes, in_features):
        super().__init__()
        self.W1 = nn.Linear(in_features, in_features, bias=False)
        self.W2 = nn.Linear(in_features, in_features, bias=False)
        self.temperature = 1e-1

    def forward(self, x):
        lhs = self.W1(x)
        rhs = self.W2(x).transpose(-1, -2)
        return F.softmax(torch.bmm(lhs, rhs) / self.temperature, dim=-1)


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, dropout=0.5):
        super().__init__()
        self.W    = nn.Linear(in_features, out_features, bias=True)
        self.bn   = nn.LayerNorm(out_features)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    @staticmethod
    def _norm_adj(A):
        deg     = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        d_inv_s = deg.pow(-0.5)
        return d_inv_s * A * d_inv_s.transpose(-1, -2)

    def forward(self, x, A):
        out = self.W(torch.bmm(self._norm_adj(A), x))
        return self.drop(F.relu(self.bn(out)))


class SGCN(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.spatial_attn = SpatialAttention(num_nodes, in_features)
        self.gcn1         = GraphConvLayer(in_features, hidden_dim, num_nodes, dropout)
        self.gcn2         = GraphConvLayer(hidden_dim,  out_dim,   num_nodes, dropout)
        self.num_nodes    = num_nodes
        self.out_dim      = out_dim

    def forward(self, x, pcc):
        B, T, N, D = x.shape
        x_flat = x.reshape(B * T, N, D)
        S      = self.spatial_attn(x_flat)
        A      = S * F.relu(pcc).reshape(B * T, N, N) + torch.eye(N, device=x.device)
        h      = self.gcn2(self.gcn1(x_flat, A), A)
        return h.reshape(B, T, N * self.out_dim), S.view(B, T, N, N)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.v  = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h):
        weights = F.softmax(self.v(self.fc(h)).squeeze(-1), dim=-1)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        return context, weights


class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, lstm_hidden, num_layers=2, dropout=0.5):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size, hidden_size=lstm_hidden,
            num_layers=num_layers, batch_first=True,
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

    def forward(self, x):
        h, _ = self.bilstm(x)
        return self.attn(self.drop(h))


def build_classifier(in_dim, fc_hidden, num_classes, dropout):
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
# Prototype-Based Modality Rebalancing Loss
# ═══════════════════════════════════════════════════════════════════════════════

class PrototypeLoss(nn.Module):
    """
    论文 Section 3.1 完整实现。

    两个模态特征 z_av [B, D_av]，z_eeg [B, D_eeg] 各自维护一套 prototype，
    prototype 在每个 forward 中用当前 batch 的特征均值更新（EMA）。

    返回：
        l_proto_av  : AV 模态 prototype loss
        l_proto_eeg : EEG 模态 prototype loss
        lambda_av   : 动态权重（标量）
        lambda_eeg  : 动态权重（标量）
    """

    def __init__(self, num_classes: int, av_dim: int, eeg_dim: int,
                 momentum: float = 0.9):
        super().__init__()
        self.num_classes = num_classes
        self.momentum    = momentum

        # prototype buffer，不参与梯度，随 batch 动态更新
        self.register_buffer('proto_av',  torch.zeros(num_classes, av_dim))
        self.register_buffer('proto_eeg', torch.zeros(num_classes, eeg_dim))
        self.register_buffer('initialized', torch.tensor(False))

    # ── prototype 更新（EMA）────────────────────────────────────────────────
    @torch.no_grad()
    def _update_prototypes(self, z_av, z_eeg, labels):
        for k in range(self.num_classes):
            mask = (labels == k)
            if mask.sum() == 0:
                continue
            mean_av  = z_av[mask].mean(0)
            mean_eeg = z_eeg[mask].mean(0)
            if not self.initialized:
                self.proto_av[k]  = mean_av
                self.proto_eeg[k] = mean_eeg
            else:
                self.proto_av[k]  = self.momentum * self.proto_av[k]  \
                                    + (1 - self.momentum) * mean_av
                self.proto_eeg[k] = self.momentum * self.proto_eeg[k] \
                                    + (1 - self.momentum) * mean_eeg
        self.initialized.fill_(True)

    # ── 单模态 prototype loss（论文公式 2、3）───────────────────────────────
    @staticmethod
    def _proto_loss(z: torch.Tensor,
                    prototypes: torch.Tensor,
                    labels: torch.Tensor):
        """
        z         : [B, D]
        prototypes: [K, D]
        labels    : [B]
        """
        # 欧氏距离 [B, K]
        dist = torch.cdist(z, prototypes, p=2)          # [B, K]
        # softmax 概率（负距离）：论文公式 2
        log_prob = F.log_softmax(-dist, dim=-1)          # [B, K]
        # 负对数似然：论文公式 3
        loss = F.nll_loss(log_prob, labels)
        # 各样本属于真实类的概率（用于收敛率计算）
        prob = log_prob.exp()                             # [B, K]
        correct_prob = prob[torch.arange(len(labels)), labels]  # [B]
        return loss, correct_prob

    # ── 动态权重（论文公式 6）───────────────────────────────────────────────
    @staticmethod
    def _compute_lambdas(r_av: torch.Tensor,
                         r_eeg: torch.Tensor):
        ratio = r_eeg / (r_av + 1e-8)
        if ratio > 1:
            # av 收敛慢，给 av 加权
            lam_av  = torch.clamp(ratio - 1, 0.0, 1.0)
            lam_eeg = torch.tensor(0.0, device=r_av.device)
        else:
            # eeg 收敛慢，给 eeg 加权
            lam_av  = torch.tensor(0.0, device=r_av.device)
            lam_eeg = torch.clamp(1.0 / (ratio + 1e-8) - 1, 0.0, 1.0)
        return lam_av, lam_eeg

    def forward(self, z_av, z_eeg, labels):
        """
        z_av  : [B, av_dim]   AV 融合特征（av_ctx）
        z_eeg : [B, eeg_dim]  EEG 特征（eeg_feat）
        labels: [B]
        """
        # 用当前 batch 更新 prototype
        self._update_prototypes(z_av.detach(), z_eeg.detach(), labels)

        # 各模态 prototype loss 及样本正确类概率
        l_av,  p_av  = self._proto_loss(z_av,  self.proto_av,  labels)
        l_eeg, p_eeg = self._proto_loss(z_eeg, self.proto_eeg, labels)

        # 收敛率：当前 batch 正确类概率之和（论文公式 4）
        r_av  = p_av.sum()
        r_eeg = p_eeg.sum()

        # 动态权重（论文公式 6）
        lam_av, lam_eeg = self._compute_lambdas(r_av, r_eeg)

        return l_av, l_eeg, lam_av, lam_eeg


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher Model（含 Prototype Rebalancing）
# ═══════════════════════════════════════════════════════════════════════════════

class TeacherModel(nn.Module):
    def __init__(
        self,
        audio_input_dim:  int,
        vision_input_dim: int,
        av_hidden_dim:    int   = 128,
        av_num_layers:    int   = 2,
        av_dropout:       float = 0.5,
        num_nodes:        int   = 32,
        eeg_in_features:  int   = 4,
        gcn_hidden:       int   = 64,
        gcn_out:          int   = 64,
        lstm_hidden:      int   = 128,
        lstm_layers:      int   = 2,
        eeg_dropout:      float = 0.5,
        fc_hidden:        int   = 256,
        num_classes:      int   = 2,
    ):
        super().__init__()

        self.audio_encoder  = ModalityEncoder(audio_input_dim,  av_hidden_dim,
                                              av_num_layers, av_dropout)
        self.vision_encoder = ModalityEncoder(vision_input_dim, av_hidden_dim,
                                              av_num_layers, av_dropout)
        self.av_dim  = self.audio_encoder.out_dim + self.vision_encoder.out_dim

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

        self.classifier = build_classifier(
            in_dim=self.eeg_dim + self.av_dim,
            fc_hidden=fc_hidden,
            num_classes=num_classes,
            dropout=max(av_dropout, eeg_dropout),
        )

    def forward(self, eeg, pcc, audio, vision,
                audio_lengths=None, vision_lengths=None):
        a_feat = self.audio_encoder(audio,  audio_lengths)
        v_feat = self.vision_encoder(vision, vision_lengths)
        av_ctx = torch.cat([a_feat, v_feat], dim=-1)       # [B, av_dim]

        gcn_out, S_attn  = self.sgcn(eeg, pcc)
        eeg_feat, attn_t = self.abilstm(gcn_out)           # [B, eeg_dim]

        fused  = torch.cat([eeg_feat, av_ctx], dim=-1)
        logits = self.classifier(fused)

        return {
            'logits'  : logits,
            'eeg_feat': eeg_feat,
            'av_ctx'  : av_ctx,
            'attn_t'  : attn_t,
            'S_attn'  : S_attn,
            'fused'   : fused,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher 训练损失（论文公式 5）
# ═══════════════════════════════════════════════════════════════════════════════

class TeacherLoss(nn.Module):
    """
    L_teacher = (1 - α) · L_CE + α · (λ_av · L_proto_av + λ_eeg · L_proto_eeg)
    论文中 α = 0.5
    """
    def __init__(self, num_classes, av_dim, eeg_dim, alpha=0.5, momentum=0.9):
        super().__init__()
        self.alpha      = alpha
        self.ce         = nn.CrossEntropyLoss()
        self.proto_loss = PrototypeLoss(num_classes, av_dim, eeg_dim, momentum)

    def forward(self, outputs: dict, labels: torch.Tensor) -> dict:
        l_ce = self.ce(outputs['logits'], labels)

        l_av, l_eeg, lam_av, lam_eeg = self.proto_loss(
            outputs['av_ctx'], outputs['eeg_feat'], labels,
        )

        l_proto = lam_av * l_av + lam_eeg * l_eeg

        loss = (1 - self.alpha) * l_ce + self.alpha * l_proto

        return {
            'loss'     : loss,
            'loss_ce'  : l_ce,
            'loss_av'  : l_av,
            'loss_eeg' : l_eeg,
            'lam_av'   : lam_av,
            'lam_eeg'  : lam_eeg,
        }