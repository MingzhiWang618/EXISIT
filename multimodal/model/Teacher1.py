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
            lengths_cpu   = lengths.cpu()
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
        return self.dropout(self.layer_norm(feat))           # [B, hidden*2]


# ═══════════════════════════════════════════════════════════════════════════════
# 公共注意力组件
# ═══════════════════════════════════════════════════════════════════════════════

class AVCrossAttention(nn.Module):
    def __init__(self, av_dim: int, eeg_dim: int, dk: int = 64):
        super().__init__()
        self.Wq    = nn.Linear(av_dim,  dk, bias=False)
        self.Wk    = nn.Linear(eeg_dim, dk, bias=False)
        self.scale = dk ** -0.5
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)

    def forward(self, av: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        q     = self.Wq(av).unsqueeze(1)
        k     = self.Wk(seq)
        score = torch.bmm(q, k.transpose(1, 2)) * self.scale
        return score.squeeze(1)                              # [..., N]


# ═══════════════════════════════════════════════════════════════════════════════
# EEG 模块 —— 区内
# ═══════════════════════════════════════════════════════════════════════════════

class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.5):
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


class IntraRegionBlock(nn.Module):
    def __init__(self, n_nodes: int, in_f: int, out_f: int, dropout: float = 0.5):
        super().__init__()
        self.A_intra = nn.Parameter(torch.ones(n_nodes, n_nodes))
        self.gcn     = GraphConvLayer(in_f, out_f, dropout)
        self.pooling = AttentionPooling(out_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = torch.sigmoid(self.A_intra)
        A = A.unsqueeze(0).expand(x.size(0), -1, -1)
        h = self.gcn(x, A)
        return self.pooling(h)                               # [B*T, out_f]


# ═══════════════════════════════════════════════════════════════════════════════
# EEG 模块 —— 区间（AV 指导）
# ═══════════════════════════════════════════════════════════════════════════════

class InterRegionSpatialAttention(nn.Module):
    def __init__(self, num_regions: int, in_features: int,
                 av_dim: int, dk: int = 64):
        super().__init__()
        self.W1            = nn.Linear(in_features, 1, bias=False)
        self.W2            = nn.Linear(num_regions, num_regions, bias=False)
        self.W3            = nn.Linear(in_features, 1, bias=False)
        self.Vs            = nn.Linear(num_regions, num_regions, bias=True)
        self.av_cross_attn = AVCrossAttention(av_dim, in_features, dk)

    def forward(self, x: torch.Tensor, av_bt: torch.Tensor):
        # x:     [B*T, R, gcn_dim]
        # av_bt: [B*T, av_dim]
        lhs   = self.W2(self.W1(x).squeeze(-1))               # [B*T, R]
        rhs   = self.W3(x).squeeze(-1)                        # [B*T, R]
        S     = torch.bmm(lhs.unsqueeze(2), rhs.unsqueeze(1)) # [B*T, R, R]
        S     = self.Vs(S)                                    # [B*T, R, R]
        alpha = self.av_cross_attn(av_bt, x)                  # [B*T, R]
        S     = S + alpha.unsqueeze(1)                        # [B*T, R, R]
        return F.softmax(S, dim=-1), alpha                    # [B*T,R,R], [B*T,R]


# ═══════════════════════════════════════════════════════════════════════════════
# EEG 模块 —— 时序（AV 指导）
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int, av_dim: int, dk: int = 64):
        super().__init__()
        self.fc            = nn.Linear(hidden_size, hidden_size)
        self.v             = nn.Linear(hidden_size, 1, bias=False)
        self.av_cross_attn = AVCrossAttention(av_dim, hidden_size, dk)

    def forward(self, h: torch.Tensor, av_ctx: torch.Tensor):
        # h:      [B, T, hidden]
        # av_ctx: [B, av_dim]
        e            = self.v(self.fc(h)).squeeze(-1)         # [B, T]
        beta         = self.av_cross_attn(av_ctx, h)          # [B, T]
        # print("e:", e)
        # print("beta:", beta)
        attn_weights = F.softmax(e + beta, dim=-1)            # [B, T] softmax后
        context      = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)
        return context, attn_weights, beta                    # [B,hidden], [B,T], [B,T]


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
        return self.attn(self.drop(h), av_ctx)  # [B,2H], [B,T], [B,T]


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
    def __init__(
        self,
        # ── AV ──────────────────────────────────────────────────────────────
        audio_input_dim:  int,
        vision_input_dim: int,
        av_hidden_dim:    int   = 128,
        av_num_layers:    int   = 2,
        av_dropout:       float = 0.5,
        # ── EEG ─────────────────────────────────────────────────────────────
        eeg_in_features:  int   = 4,
        gcn_dim:          int   = 64,
        lstm_hidden:      int   = 128,
        lstm_layers:      int   = 2,
        eeg_dropout:      float = 0.5,
        dk:               int   = 64,
        # ── 分类器 ───────────────────────────────────────────────────────────
        fc_hidden:        int   = 256,
        num_classes:      int   = 2,
    ):
        super().__init__()

        # ── 1. AV 编码器 ────────────────────────────────────────────────────
        self.audio_encoder  = ModalityEncoder(audio_input_dim,  av_hidden_dim,
                                              av_num_layers, av_dropout)
        self.vision_encoder = ModalityEncoder(vision_input_dim, av_hidden_dim,
                                              av_num_layers, av_dropout)
        self.av_dim = self.audio_encoder.out_dim + self.vision_encoder.out_dim

        # ── 2. 脑区定义 ─────────────────────────────────────────────────────
        electrode_names = [
            'Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6',
            'T7','C3','Cz','C4','T8','CP5','CP1','CP2','CP6',
            'P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10'
        ]
        region_definition = {
            "Fp": ['Fp1','Fp2'],
            "F" : ['F7','F3','Fz','F4','F8'],
            "FC": ['FC5','FC1','FC2','FC6'],
            "T" : ['T7','T8'],
            "C" : ['C3','Cz','C4'],
            "CP": ['CP5','CP1','CP2','CP6'],
            "PO": ['P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10'],
        }
        self.num_regions = len(region_definition)
        self.gcn_dim     = gcn_dim
        for i, (_, elecs) in enumerate(region_definition.items()):
            idx = torch.tensor([electrode_names.index(e) for e in elecs])
            self.register_buffer(f'region_idx_{i}', idx)

        # ── 3. Level1：区内 GCN ─────────────────────────────────────────────
        self.intra_blocks = nn.ModuleList([
            IntraRegionBlock(n_nodes=len(elecs), in_f=eeg_in_features,
                             out_f=gcn_dim, dropout=eeg_dropout)
            for elecs in region_definition.values()
        ])

        # ── 4. Level2：区间 SpatialAttention（AV 指导） ──────────────────────
        self.inter_attn = InterRegionSpatialAttention(
            num_regions=self.num_regions,
            in_features=gcn_dim,
            av_dim=self.av_dim,
            dk=dk,
        )
        self.inter_proj = nn.Linear(gcn_dim, gcn_dim)
        self.inter_bn   = nn.BatchNorm1d(gcn_dim)
        self.inter_drop = nn.Dropout(eeg_dropout)

        # ── 5. 时序编码：BiLSTM + TemporalAttention（AV 指导） ──────────────
        self.abilstm = AttentionBiLSTM(
            input_size=self.num_regions * gcn_dim,
            lstm_hidden=lstm_hidden,
            num_layers=lstm_layers,
            dropout=eeg_dropout,
            av_dim=self.av_dim,
            dk=dk,
        )
        self.eeg_dim = lstm_hidden * 2

        # ── 6. 两路独立分类头 ───────────────────────────────────────────────
        self.av_classifier  = build_classifier(self.av_dim,  fc_hidden,
                                               num_classes, av_dropout)
        self.eeg_classifier = build_classifier(self.eeg_dim, fc_hidden,
                                               num_classes, eeg_dropout)

    def forward(self, eeg: torch.Tensor,
                audio: torch.Tensor, vision: torch.Tensor,
                audio_lengths:  Optional[torch.Tensor] = None,
                vision_lengths: Optional[torch.Tensor] = None) -> dict:
        B, T, N, D = eeg.shape

        # ── Step1: AV 编码 ──────────────────────────────────────────────────
        a_feat = self.audio_encoder(audio,  audio_lengths)
        v_feat = self.vision_encoder(vision, vision_lengths)
        av_ctx = torch.cat([a_feat, v_feat], dim=-1)         # [B, av_dim]

        # ── Step2: 区内 GCN ─────────────────────────────────────────────────
        x_flat = eeg.view(B * T, N, D)
        region_reps = []
        for i, block in enumerate(self.intra_blocks):
            idx = getattr(self, f'region_idx_{i}')
            z_r = block(x_flat[:, idx, :])                   # [B*T, gcn_dim]
            region_reps.append(z_r)

        # ── Step3: 区间 SpatialAttention（AV 指导） ─────────────────────────
        h_inter = torch.stack(region_reps, dim=1)            # [B*T, R, gcn_dim]
        av_bt   = av_ctx.unsqueeze(1).expand(B, T, -1) \
                        .reshape(B * T, -1)                  # [B*T, av_dim]

        S, alpha = self.inter_attn(h_inter, av_bt)           # [B*T,R,R], [B*T,R]
        h_inter  = torch.bmm(S, h_inter)                     # [B*T, R, gcn_dim]

        h_inter = self.inter_proj(h_inter)
        h_inter = F.relu(h_inter.transpose(1, 2))
        h_inter = self.inter_bn(h_inter)
        h_inter = self.inter_drop(h_inter.transpose(1, 2))   # [B*T, R, gcn_dim]

        # S 是 softmax 后的概率分布，直接用于蒸馏
        R_inter = S.view(B, T, self.num_regions, self.num_regions)  # [B, T, R, R]
        alpha   = alpha.view(B, T, self.num_regions)                # [B, T, R]

        # ── Step4: 展平 → 时序序列 ──────────────────────────────────────────
        z_seq = h_inter.reshape(B, T, self.num_regions * self.gcn_dim)  # [B, T, 448]

        # ── Step5: BiLSTM + TemporalAttention（AV 指导） ────────────────────
        # attn_weights: softmax后的时序注意力权重，用于蒸馏
        # beta:         AV cross-attention 的原始分数
        eeg_feat, attn_weights, beta = self.abilstm(z_seq, av_ctx)

        # ── Step6: 分类 ─────────────────────────────────────────────────────
        av_logits  = self.av_classifier(av_ctx)
        eeg_logits = self.eeg_classifier(eeg_feat)
        # print(attn_weights)
        # print(R_inter)
        return {
            'av_logits'   : av_logits,     # [B, C]
            'eeg_logits'  : eeg_logits,    # [B, C]
            'eeg_feat'    : eeg_feat,      # [B, lstm_hidden*2]
            'attn_weights': attn_weights,  # [B, T]  softmax后 ← 用于蒸馏
            'R_inter'     : R_inter,       # [B, T, R, R]  softmax后 ← 用于蒸馏
            'alpha'       : alpha,         # [B, T, R]  原始分数
            'beta'        : beta,          # [B, T]  原始分数
            'av_ctx'      : av_ctx,        # [B, av_dim]
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
    B, T, N, D = 4, 5, 30, 4
    T_a, F_a   = 10, 40
    T_v, F_v   = 10, 512

    model = TeacherModel(
        audio_input_dim=F_a, vision_input_dim=F_v,
        eeg_in_features=D, num_classes=2,
    )

    eeg    = torch.randn(B, T, N, D)
    audio  = torch.randn(B, T_a, F_a)
    vision = torch.randn(B, T_v, F_v)
    labels = torch.randint(0, 2, (B,))

    out     = model(eeg, audio, vision)
    loss_fn = TeacherLoss()
    loss    = loss_fn(out, labels)

    print(f"av_logits    : {out['av_logits'].shape}")      # [4, 2]
    print(f"eeg_logits   : {out['eeg_logits'].shape}")     # [4, 2]
    print(f"eeg_feat     : {out['eeg_feat'].shape}")       # [4, 128]
    print(f"attn_weights : {out['attn_weights'].shape}")   # [4, 5]
    print(f"R_inter      : {out['R_inter'].shape}")        # [4, 5, 7, 7]
    print(f"alpha        : {out['alpha'].shape}")          # [4, 5, 7]
    print(f"beta         : {out['beta'].shape}")           # [4, 5]
    print(f"loss         : {loss['loss'].item():.4f}")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params       : {total:,}")