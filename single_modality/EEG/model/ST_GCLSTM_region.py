import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Attention Pooling
# =========================
class AttentionPooling(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.w = nn.Linear(in_dim, 1, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [B, N, Dim]
        scores = self.w(x).squeeze(-1)          # [B, N]
        weights = F.softmax(scores, dim=-1)
        z = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return z, weights


# =========================
# GCN Layer
# =========================
class GraphConvLayer(nn.Module):
    def __init__(self, in_f, out_f, dropout=0.5):
        super().__init__()
        self.W = nn.Linear(in_f, out_f)
        self.bn = nn.BatchNorm1d(out_f)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, A):
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        norm_A = A / deg
        out = torch.matmul(norm_A, x)
        out = self.W(out)
        out = F.relu(out)

        out = out.transpose(1, 2)
        out = self.bn(out)
        out = out.transpose(1, 2)

        return self.drop(out)


# =========================
# Intra-Region Block
# =========================
class IntraRegionBlock(nn.Module):
    def __init__(self, n_nodes, in_f, out_f, dropout=0.5):
        super().__init__()
        self.A_intra = nn.Parameter(torch.ones(n_nodes, n_nodes))
        self.gcn = GraphConvLayer(in_f, out_f, dropout)
        self.pooling = AttentionPooling(out_f)

    def forward(self, x):
        A = torch.sigmoid(self.A_intra)
        h = self.gcn(x, A)
        z_region, _ = self.pooling(h)
        return z_region


# =========================
# Inter-Region Self-Attention
# =========================
class InterRegionAttention(nn.Module):
    def __init__(self, dim=64, num_heads=4, dropout=0.5):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B*T, 7, 64]
        out, attn_weights = self.attn(x, x, x)
        out = self.norm(x + out)                # residual + LN
        return self.drop(out), attn_weights
        # attn_weights: [B*T, 7, 7]


# =========================
# Region Gate
# =========================
class RegionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B*T, R, dim]
        gates = self.fc(x)      # [B*T, R, 1]
        out = x * gates
        return out, gates


# =========================
# 主模型
# =========================
class ST_GCLSTM(nn.Module):
    def __init__(
        self,
        in_features=5,
        gcn_dim=64,
        num_heads=4,
        lstm_hidden=128,
        num_classes=5,
        dropout=0.5
    ):
        super().__init__()

        # ===== 脑区定义 =====
        self.electrode_names = [
            'Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6',
            'T7','C3','Cz','C4','T8','CP5','CP1','CP2','CP6','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10'
        ]
        self.region_definition = {
            "Fp": ['Fp1','Fp2'], "F": ['F7','F3','Fz','F4','F8'], "FC": ['FC5','FC1','FC2','FC6'],
            "T": ['T7','T8'], "C": ['C3','Cz','C4'], "CP": ['CP5','CP1','CP2','CP6'],
            "PO": ['P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10']
        }

        self.region_indices = [
            torch.tensor([self.electrode_names.index(e) for e in electrodes])
            for _, electrodes in self.region_definition.items()
        ]
        self.num_regions = len(self.region_indices)

        # ===== Level 1: Intra =====
        self.intra_blocks = nn.ModuleList([
            IntraRegionBlock(len(idx), in_features, gcn_dim, dropout)
            for idx in self.region_indices
        ])

        # ===== Level 2: Inter =====
        self.inter_attn = InterRegionAttention(dim=gcn_dim, num_heads=num_heads, dropout=dropout)
        self.region_gate = RegionGate(gcn_dim)

        # ===== Temporal =====
        self.bilstm = nn.LSTM(
            input_size=self.num_regions * gcn_dim,  # 7 * 64 = 448
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.temporal_attn = AttentionPooling(lstm_hidden * 2)

        # ===== Classifier =====
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, T, N, D]
        B, T, N, D = x.shape
        device = x.device

        x_flat = x.view(B * T, N, D)

        # ===== 1. Intra =====
        region_reps = []
        for i, block in enumerate(self.intra_blocks):
            idx = self.region_indices[i].to(device)
            z_r = block(x_flat[:, idx, :])
            region_reps.append(z_r)

        # ===== 2. Inter =====
        h_inter = torch.stack(region_reps, dim=1)           # [B*T, 7, 64]

        h_inter, attn_weights = self.inter_attn(h_inter)    # [B*T, 7, 64], [B*T, 7, 7]

        h_inter, region_gates = self.region_gate(h_inter)   # [B*T, 7, 64], [B*T, 7, 1]

        # ===== 3. Temporal =====
        z_seq = h_inter.reshape(B, T, -1)                   # [B, T, 448]
        h_lstm, _ = self.bilstm(z_seq)

        ctx, t_weights = self.temporal_attn(h_lstm)

        # ===== 4. Classifier =====
        logits = self.classifier(ctx)

        return logits, attn_weights, region_gates, t_weights


def test_model():
    B, T, N, D = 2, 10, 30, 5
    x = torch.randn(B, T, N, D)
    model = ST_GCLSTM(in_features=D, gcn_dim=64, num_heads=4, lstm_hidden=128, num_classes=5, dropout=0.5)
    logits, attn_weights, region_gates, t_weights = model(x)
    print("logits:       ", logits.shape)        # [2, 5]
    print("attn_weights: ", attn_weights.shape)  # [2*10, 7, 7]
    print("region_gates: ", region_gates.shape)  # [2*10, 7, 1]
    print("t_weights:    ", t_weights.shape)     # [2, 10]

if __name__ == "__main__":
    test_model()