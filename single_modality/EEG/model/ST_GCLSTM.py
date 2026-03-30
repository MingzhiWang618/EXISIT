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
        # return torch.sigmoid(S)             # [B*T, N, N]
        return F.softmax(S / self.temperature, dim=-1)
                                                   # [B*T, N, N]


class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 num_nodes: int, dropout: float = 0.5):
        super().__init__()
        self.W    = nn.Linear(in_features, out_features, bias=True)
        # self.bn = nn.BatchNorm1d(out_features)
        self.bn = nn.LayerNorm(out_features)
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
        out = self.bn(out)   # LayerNorm 直接作用于最后一维 out_features ✓
        out = F.relu(out)
        return self.drop(out)


class SGCN(nn.Module):
    """
    不再持有 A_base 参数。
    每个样本的 PCC [B, T, V, V] 直接作为静态先验传入 forward。
    """

    def __init__(self, num_nodes, in_features,
                 hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        # ── 去掉 A_base，不再有任何 PCC 相关参数 ─────────────────────────────
        self.spatial_attn = SpatialAttention(num_nodes, in_features)
        self.gcn1 = GraphConvLayer(in_features, hidden_dim, num_nodes, dropout)
        self.gcn2 = GraphConvLayer(hidden_dim,  out_dim,   num_nodes, dropout)
        self.num_nodes = num_nodes
        self.out_dim   = out_dim

    def forward(self, x: torch.Tensor, pcc: torch.Tensor) -> torch.Tensor:
        """
        x  : [B, T, N, F]
        pcc: [B, T, N, N]  每个样本自己的 PCC 连接矩阵
        """
        B, T, N, D = x.shape
        x_flat = x.reshape(B * T, N, D)         # [B*T, N, D]

        # ── 动态注意力 ────────────────────────────────────────────────────────
        S = self.spatial_attn(x_flat)            # [B*T, N, N]
        # ── 每样本 PCC 作为静态先验 ───────────────────────────────────────────
        # pcc [B, T, N, N] → 对称化 → [B*T, N, N]
        # relu 保证非负（负相关=无连接）
        A_base = F.relu(pcc)                             # [B, T, N, N]  值域 [0, 1]
        A_base = A_base.reshape(B * T, N, N)              # [B*T, N, N]

        # ── 融合 + 自环 ───────────────────────────────────────────────────────
        A = S * A_base                                     # [B*T, N, N]
        I = torch.eye(N, device=x.device).unsqueeze(0)    # [1,  N, N]
        A = A + I

        # ── 统计逻辑 ─────────────────────────────────────────────────────────
        
        # 恢复默认设置（可选，避免后续日志爆炸）
        torch.set_printoptions(profile="default")                                        # [B*T, N, N]
        # print(A)
        # ── GCN ──────────────────────────────────────────────────────────────
        h = self.gcn1(x_flat, A)                          # [B*T, N, hidden]
        h = self.gcn2(h,      A)                          # [B*T, N, out_dim]
        h = h.reshape(B, T, N * self.out_dim)             # [B, T, N*out_dim]
        return h


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.v  = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h: torch.Tensor):
        # score   = self.v(torch.tanh(self.fc(h))).squeeze(-1)
        score   = self.v(self.fc(h)).squeeze(-1)     # [B, T]
        weights = F.softmax(score, dim=-1)                       # [B, T]
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
        h    = self.drop(h)
        return self.attn(h)


class ST_GCLSTM(nn.Module):
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
        # ── 去掉 init_pcc，不再需要 ────────────────────────────────────────
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

    def forward(self, x: torch.Tensor, pcc: torch.Tensor):
        """
        x  : [B, T, N, F]
        pcc: [B, T, N, N]  来自 DataLoader，每个样本自己的 PCC
        """
        gcn_out     = self.sgcn(x, pcc)        # [B, T, N*gcn_out]
        ctx, attn_w = self.abilstm(gcn_out)    # [B, H*2], [B, T]
        logits      = self.classifier(ctx)     # [B, num_classes]
        return logits, attn_w


# =============================================================================
# 使用示例
# =============================================================================
if __name__ == '__main__':
    B, T, N, D = 8, 5, 30, 5

    model  = ST_GCLSTM(num_nodes=N, in_features=D)
    x      = torch.randn(B, T, N, D)
    pcc    = torch.randn(B, T, N, N)   # 每个样本自己的 PCC [B, T, N, N]

    logits, attn_w = model(x, pcc)
    print(f"logits : {logits.shape}")   # [8, 2]
    print(f"attn_w : {attn_w.shape}")  # [8, 5]