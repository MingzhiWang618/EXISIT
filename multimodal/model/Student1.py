import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 区内模块
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores  = self.w(x).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.5):
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
        out = self.W(torch.bmm(self._norm_adj(A), x))  # [B*T, N, F]
        out = self.bn(out)
        out = F.relu(out)
        return self.drop(out)


class IntraRegionBlock(nn.Module):
    def __init__(self, n_nodes: int, in_f: int, out_f: int, dropout: float = 0.5):
        super().__init__()
        self.gcn     = GraphConvLayer(in_f, out_f, dropout)
        self.pooling = AttentionPooling(out_f)

    def forward(self, x: torch.Tensor, pcc: torch.Tensor) -> torch.Tensor:
        # x:   [B*T, n_nodes, in_f]
        # pcc: [B*T, n_nodes, n_nodes]
        A = torch.abs(pcc) + torch.eye(x.size(1), device=x.device).unsqueeze(0)
        h = self.gcn(x, A)       # [B*T, N, out_f]
        return self.pooling(h)   # [B*T, out_f]


# ═══════════════════════════════════════════════════════════════════════════════
# 区间模块
# ═══════════════════════════════════════════════════════════════════════════════

class InterRegionSpatialAttention(nn.Module):
    def __init__(self, num_regions: int, in_features: int):
        super().__init__()
        self.W1 = nn.Linear(in_features, in_features, bias=False)
        self.W2 = nn.Linear(in_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, R, gcn_dim]
        lhs = self.W1(x)                           # [B*T, R, D]
        rhs = self.W2(x).transpose(-1, -2)         # [B*T, D, R]
        S   = torch.bmm(lhs, rhs)                  # [B*T, R, R]
        S   = torch.sigmoid(S)
        return F.softmax(S, dim=-1)                # [B*T, R, R]


# ═══════════════════════════════════════════════════════════════════════════════
# 时序模块
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.v  = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h: torch.Tensor):
        e       = self.v(self.fc(h)).squeeze(-1)   # [B, T]
        weights = F.softmax(e, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        return context, weights                    # [B, hidden], [B, T]


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
        self.drop = nn.Dropout(dropout if num_layers > 1 else 0.0)
        self.attn = TemporalAttention(lstm_hidden * 2)

        for name, param in self.bilstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor):
        h, _ = self.bilstm(x)
        h    = self.drop(h)
        return self.attn(h)                        # [B, 2H], [B, T]


# ═══════════════════════════════════════════════════════════════════════════════
# Student Model
# ═══════════════════════════════════════════════════════════════════════════════

class StudentModel(nn.Module):
    def __init__(
        self,
        eeg_in_features: int   = 4,
        gcn_dim:         int   = 64,
        lstm_hidden:     int   = 128,
        lstm_layers:     int   = 2,
        fc_hidden:       int   = 256,
        num_classes:     int   = 2,
        dropout:         float = 0.5,
    ):
        super().__init__()

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

        # 注册每个脑区的电极索引
        for i, (_, elecs) in enumerate(region_definition.items()):
            idx = torch.tensor([electrode_names.index(e) for e in elecs])
            self.register_buffer(f'region_idx_{i}', idx)

        # ── Level1：区内 GCN ──────────────────────────────────────────────────
        self.intra_blocks = nn.ModuleList([
            IntraRegionBlock(
                n_nodes = len(elecs),
                in_f    = eeg_in_features,
                out_f   = gcn_dim,
                dropout = dropout,
            )
            for elecs in region_definition.values()
        ])

        # ── Level2：区间 SpatialAttention ─────────────────────────────────────
        self.inter_attn = InterRegionSpatialAttention(
            num_regions=self.num_regions,
            in_features=gcn_dim,
        )
        self.inter_proj = nn.Linear(gcn_dim, gcn_dim)
        self.inter_bn   = nn.LayerNorm(gcn_dim)
        self.inter_drop = nn.Dropout(dropout)

        # ── 时序编码 ──────────────────────────────────────────────────────────
        self.abilstm = AttentionBiLSTM(
            input_size  = self.num_regions * gcn_dim,
            lstm_hidden = lstm_hidden,
            num_layers  = lstm_layers,
            dropout     = dropout,
        )
        self.eeg_dim = lstm_hidden * 2

        # ── 分类器 ────────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self.eeg_dim, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, eeg: torch.Tensor, pcc: torch.Tensor) -> dict:
        # eeg: [B, T, N, D]
        # pcc: [B, T, N, N]
        B, T, N, D = eeg.shape
        x_flat   = eeg.view(B * T, N, D)
        pcc_flat = pcc.view(B * T, N, N)

        # ── Step1: 区内 GCN ───────────────────────────────────────────────────
        region_reps = []
        for i, block in enumerate(self.intra_blocks):
            idx     = getattr(self, f'region_idx_{i}')
            pcc_sub = pcc_flat[:, idx][:, :, idx]        # [B*T, n_i, n_i]
            z_r     = block(x_flat[:, idx, :], pcc_sub)  # [B*T, gcn_dim]
            region_reps.append(z_r)

        # ── Step2: 区间 SpatialAttention ──────────────────────────────────────
        h_inter = torch.stack(region_reps, dim=1)         # [B*T, R, gcn_dim]
        S       = self.inter_attn(h_inter)                # [B*T, R, R]
        h_inter = torch.bmm(S, h_inter)                   # [B*T, R, gcn_dim]

        h_inter = self.inter_proj(h_inter)
        h_inter = F.relu(h_inter)
        h_inter = self.inter_bn(h_inter)
        h_inter = self.inter_drop(h_inter)

        R_inter = S.view(B, T, self.num_regions, self.num_regions)

        # ── Step3: 展平 → 时序序列 ────────────────────────────────────────────
        z_seq = h_inter.reshape(B, T, self.num_regions * self.gcn_dim)

        # ── Step4: BiLSTM + 时序注意力 ────────────────────────────────────────
        eeg_feat, attn_t = self.abilstm(z_seq)            # [B, 2H], [B, T]

        # ── Step5: 分类 ───────────────────────────────────────────────────────
        logits = self.classifier(eeg_feat)

        return {
            'logits'  : logits,    # [B, C]
            'eeg_feat': eeg_feat,  # [B, lstm_hidden*2]
            'attn_t'  : attn_t,    # [B, T]
            'R_inter' : R_inter,   # [B, T, R, R]
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    B, T, N, D = 4, 5, 30, 4

    model = StudentModel(eeg_in_features=D, num_classes=2)

    eeg = torch.randn(B, T, N, D)
    pcc = torch.randn(B, T, N, N)
    pcc = (pcc + pcc.transpose(-1, -2)) / 2   # 保证对称
    pcc = pcc.clamp(-1, 1)                    # 保证值域

    out = model(eeg, pcc)
    print(f"logits  : {out['logits'].shape}")   # [4, 2]
    print(f"eeg_feat: {out['eeg_feat'].shape}") # [4, 256]
    print(f"attn_t  : {out['attn_t'].shape}")   # [4, 5]
    print(f"R_inter : {out['R_inter'].shape}")  # [4, 5, 7, 7]

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params  : {total:,}")