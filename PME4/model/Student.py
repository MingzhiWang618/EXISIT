from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, num_nodes: int, in_features: int):
        super().__init__()
        self.W1 = nn.Linear(in_features, in_features, bias=False)
        self.W2 = nn.Linear(in_features, in_features, bias=False)
        self.temperature = 1e-1

    def forward(self, x):
        # x: [B*T, N, D]
        lhs = self.W1(x)                    # [B*T, N, D]
        rhs = self.W2(x).transpose(-1, -2)  # [B*T, D, N]
        S = torch.bmm(lhs, rhs)
        return F.softmax(S / self.temperature, dim=-1)  # [B*T, N, N]


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
    def __init__(self, num_nodes, in_features,
                 hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.spatial_attn = SpatialAttention(num_nodes, in_features)
        self.gcn1      = GraphConvLayer(in_features, hidden_dim, num_nodes, dropout)
        self.gcn2      = GraphConvLayer(hidden_dim,  out_dim,   num_nodes, dropout)
        self.num_nodes = num_nodes
        self.out_dim   = out_dim

    def forward(self, x: torch.Tensor, pcc: torch.Tensor):
        """
        x  : [B, T, N, F]
        pcc: [B, T, N, N]

        Returns
        -------
        h     : [B, T, N*out_dim]
        S_out : [B, T, N, N]      ← 用于与 Teacher S_attn 蒸馏对齐
        """
        B, T, N, D = x.shape
        x_flat = x.reshape(B * T, N, D)

        S      = self.spatial_attn(x_flat)                        # [B*T, N, N]

        A_base = F.relu(pcc).reshape(B * T, N, N)                 # [B*T, N, N]
        A      = S * A_base + torch.eye(N, device=x.device)       # [B*T, N, N]

        h = self.gcn2(self.gcn1(x_flat, A), A)                   # [B*T, N, out_dim]

        S_out = S.view(B, T, N, N)                                # [B, T, N, N]
        return h.reshape(B, T, N * self.out_dim), S_out


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.v  = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h: torch.Tensor):
        score   = self.v(self.fc(h)).squeeze(-1)      # [B, T]
        weights = F.softmax(score, dim=-1)             # [B, T]
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # [B, H]
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
        return self.attn(self.drop(h))   # [B, H*2], [B, T]


class ST_GCLSTM(nn.Module):
    """
    Student model.

    forward() 返回字典，与 Teacher 对齐的蒸馏项：
        S_attn  : [B, T, N, N]   空间注意力矩阵（对齐 Teacher S_attn）
        attn_t  : [B, T]         时序注意力权重（对齐 Teacher attn_t）
        eeg_feat: [B, lstm_hidden*2]  特征向量（可选，用于特征蒸馏）
        logits  : [B, num_classes]
    """

    def __init__(
        self,
        num_nodes:   int   = 32,
        in_features: int   = 4,
        gcn_hidden:  int   = 64,
        gcn_out:     int   = 64,
        lstm_hidden: int   = 128,
        lstm_layers: int   = 2,
        fc_hidden:   int   = 256,
        num_classes: int   = 2,
        dropout:     float = 0.5,
    ):
        super().__init__()

        self.sgcn = SGCN(
            num_nodes=num_nodes,
            in_features=in_features,
            hidden_dim=gcn_hidden,
            out_dim=gcn_out,
            dropout=dropout,
        )

        self.abilstm = AttentionBiLSTM(
            input_size=num_nodes * gcn_out,
            lstm_hidden=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, pcc: torch.Tensor) -> dict:
        """
        x  : [B, T, N, F]
        pcc: [B, T, N, N]

        Returns dict with keys:
            logits   [B, C]
            eeg_feat [B, lstm_hidden*2]
            attn_t   [B, T]
            S_attn   [B, T, N, N]
        """
        gcn_out, S_attn      = self.sgcn(x, pcc)           # [B,T,N*gcn_out], [B,T,N,N]
        eeg_feat, attn_t     = self.abilstm(gcn_out)        # [B, H*2], [B, T]
        logits               = self.classifier(eeg_feat)    # [B, C]

        return {
            'logits'  : logits,
            'eeg_feat': eeg_feat,
            'attn_t'  : attn_t,
            'S_attn'  : S_attn,
        }


# =============================================================================
# 测试
# =============================================================================
if __name__ == '__main__':
    B, T, N, D = 8, 5, 30, 4

    model  = ST_GCLSTM(num_nodes=N, in_features=D)
    x      = torch.randn(B, T, N, D)
    pcc    = torch.randn(B, T, N, N)

    out = model(x, pcc)
    print(f"logits   : {out['logits'].shape}")    # [8, 2]
    print(f"eeg_feat : {out['eeg_feat'].shape}")  # [8, 256]
    print(f"attn_t   : {out['attn_t'].shape}")    # [8, 5]
    print(f"S_attn   : {out['S_attn'].shape}")    # [8, 5, 30, 30]

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params   : {total:,}")