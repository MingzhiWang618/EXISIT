import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, List, Union


class AttentionPooling(nn.Module):
    """简单的注意力池化：对LSTM输出序列进行加权平均"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        lstm_outputs: [B, T, hidden_dim]  (已包含双向拼接)
        lengths: [B] 实际长度，用于mask
        """
        # 计算注意力分数
        attn_scores = self.attn_weights(lstm_outputs).squeeze(-1)  # [B, T]
        if lengths is not None:
            # 构造mask，将padding部分设为 -inf
            mask = torch.arange(lstm_outputs.size(1), device=lstm_outputs.device)[None, :] < lengths[:, None]
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T]
        # 加权求和
        pooled = torch.bmm(attn_weights.unsqueeze(1), lstm_outputs).squeeze(1)  # [B, hidden_dim]
        return pooled


class ModalityEncoder(nn.Module):
    """
    改进的单模态编码器，支持：
    - 双向LSTM/GRU
    - 变长序列处理（pack_padded）
    - 多种池化方式：最后隐藏状态(last)、平均池化(mean)、最大池化(max)、注意力池化(attention)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        rnn_type: str = 'lstm',       # 'lstm' 或 'gru'
        pooling: str = 'last',         # 'last', 'mean', 'max', 'attention'
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling
        rnn_dropout = dropout if num_layers > 1 else 0.0

        # 选择RNN类型
        rnn_cls = nn.LSTM if rnn_type.lower() == 'lstm' else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
        )

        # 输出维度 = hidden_dim * (2 if bidirectional else 1)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

        # 如果使用注意力池化，需要额外的线性层
        if pooling == 'attention':
            self.attention = AttentionPooling(self.out_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.out_dim)  # 可选的层归一化

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, F] 或 [B, F] (此时T=1)
        lengths: [B] 实际长度（如果为None，则认为没有padding）
        """
        # 处理2D输入（单个时间步）
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, F]
            if lengths is None:
                lengths = torch.ones(x.size(0), dtype=torch.long, device=x.device)

        # 打包变长序列（如果提供了lengths且所有序列长度相同则跳过，但保险起见仍用pack）
        if lengths is not None:
            # 确保长度在CPU上且按降序排列（pack_padded_sequence需要）
            lengths_cpu = lengths.cpu()
            # 按长度降序排序
            sorted_lengths, sort_idx = lengths_cpu.sort(descending=True)
            x = x[sort_idx]
            # 打包
            packed_x = pack_padded_sequence(x, sorted_lengths, batch_first=True, enforce_sorted=True)
            packed_output, hidden = self.rnn(packed_x)
            # 解包
            output, _ = pad_packed_sequence(packed_output, batch_first=True)  # [B, T, out_dim]
            # 恢复原始batch顺序
            inv_sort_idx = sort_idx.argsort()
            output = output[inv_sort_idx]
            if isinstance(hidden, tuple):  # LSTM返回 (h_n, c_n)
                hidden = hidden[0]  # 取h_n
            hidden = hidden[:, inv_sort_idx, :]  # [num_layers * num_directions, B, hidden_dim]
        else:
            output, hidden = self.rnn(x)
            if isinstance(hidden, tuple):
                hidden = hidden[0]

        # 根据池化方式聚合序列信息
        if self.pooling == 'last':
            # 取最后一个有效时间步的隐藏状态（即双向最后一层）
            # hidden 形状: [num_layers * num_directions, B, hidden_dim]
            # 双向时，最后一层的前向和后向拼接得到 out_dim
            if self.bidirectional:
                # 取最后一层的前向和后向
                forward_last = hidden[-2]  # 最后一层前向
                backward_last = hidden[-1]  # 最后一层后向
                feat = torch.cat([forward_last, backward_last], dim=-1)  # [B, out_dim]
            else:
                feat = hidden[-1]  # [B, hidden_dim]
        elif self.pooling == 'mean':
            # 对时间维度平均（需mask掉padding）
            if lengths is not None:
                mask = torch.arange(output.size(1), device=output.device)[None, :] < lengths[:, None]
                mask = mask.unsqueeze(-1).float()  # [B, T, 1]
                sum_out = (output * mask).sum(dim=1)  # [B, out_dim]
                feat = sum_out / lengths.float().unsqueeze(-1)
            else:
                feat = output.mean(dim=1)
        elif self.pooling == 'max':
            # 对时间维度最大池化（需mask）
            if lengths is not None:
                mask = torch.arange(output.size(1), device=output.device)[None, :] < lengths[:, None]
                # 将padding部分设为 -inf
                output_masked = output.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                feat = output_masked.max(dim=1)[0]
            else:
                feat = output.max(dim=1)[0]
        elif self.pooling == 'attention':
            feat = self.attention(output, lengths)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        # 层归一化与Dropout
        feat = self.layer_norm(feat)
        return self.dropout(feat)


class GatedFusion(nn.Module):
    """门控融合：对两个特征向量计算门控权重，加权融合"""
    def __init__(self, dim_a: int, dim_v: int, fused_dim: int):
        super().__init__()
        self.linear_a = nn.Linear(dim_a, fused_dim)
        self.linear_v = nn.Linear(dim_v, fused_dim)
        self.gate = nn.Linear(dim_a + dim_v, fused_dim)

    def forward(self, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        proj_a = self.linear_a(a)
        proj_v = self.linear_v(v)
        gate = torch.sigmoid(self.gate(torch.cat([a, v], dim=-1)))
        fused = gate * proj_a + (1 - gate) * proj_v
        return fused


class OuterProductFusion(nn.Module):
    """外积融合（双线性池化），降维至 fused_dim"""
    def __init__(self, dim_a: int, dim_v: int, fused_dim: int):
        super().__init__()
        # 双线性投影: 外积后接线性层
        self.bilinear = nn.Bilinear(dim_a, dim_v, fused_dim)

    def forward(self, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.bilinear(a, v)


class ImprovedAudioVisualBiLSTM(nn.Module):
    """
    改进的双模态LSTM模型，支持多种池化、融合和分类器配置。
    """
    def __init__(
        self,
        audio_input_dim: int,
        vision_input_dim: int,
        num_classes: int = 5,
        # 编码器参数
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        rnn_type: str = 'lstm',
        pooling: str = 'last',            # 编码器池化方式
        bidirectional: bool = True,
        # 融合参数
        fusion: str = 'concat',            # 'concat', 'gated', 'outer'
        fusion_output_dim: Optional[int] = None,  # 若None，concat时自动计算，门控/外积需指定
        # 分类器参数
        classifier_hidden_dims: List[int] = [128, 64],
        use_batchnorm: bool = True,
        use_layernorm: bool = False,      # 与BatchNorm互斥
    ):
        super().__init__()
        self.fusion = fusion

        # 音频编码器
        self.audio_encoder = ModalityEncoder(
            input_dim=audio_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            pooling=pooling,
            bidirectional=bidirectional,
        )
        # 视觉编码器
        self.vision_encoder = ModalityEncoder(
            input_dim=vision_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            pooling=pooling,
            bidirectional=bidirectional,
        )

        audio_out_dim = self.audio_encoder.out_dim
        vision_out_dim = self.vision_encoder.out_dim

        # 融合模块
        if fusion == 'concat':
            fused_dim = audio_out_dim + vision_out_dim
        elif fusion == 'gated':
            fused_dim = fusion_output_dim if fusion_output_dim is not None else audio_out_dim
            self.fusion_layer = GatedFusion(audio_out_dim, vision_out_dim, fused_dim)
        elif fusion == 'outer':
            fused_dim = fusion_output_dim if fusion_output_dim is not None else audio_out_dim
            self.fusion_layer = OuterProductFusion(audio_out_dim, vision_out_dim, fused_dim)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion}")

        # 分类器
        layers = []
        prev_dim = fused_dim
        for i, h_dim in enumerate(classifier_hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
        elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(
        self,
        audio: torch.Tensor,
        vision: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        vision_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        audio: [B, T_a, F_a] 或 [B, F_a]
        vision: [B, T_v, F_v] 或 [B, F_v]
        audio_lengths, vision_lengths: [B] 实际长度（如果为None，则无padding）
        """
        a_feat = self.audio_encoder(audio, audio_lengths)   # [B, audio_out_dim]
        v_feat = self.vision_encoder(vision, vision_lengths) # [B, vision_out_dim]

        # 融合
        if self.fusion == 'concat':
            fused = torch.cat([a_feat, v_feat], dim=-1)
        else:
            fused = self.fusion_layer(a_feat, v_feat)

        # 分类
        logits = self.classifier(fused)
        return logits


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 模拟数据
    batch_size = 4
    audio_len = 10
    vision_len = 8
    audio_dim = 20
    vision_dim = 30
    num_classes = 5

    audio = torch.randn(batch_size, audio_len, audio_dim)
    vision = torch.randn(batch_size, vision_len, vision_dim)
    audio_lengths = torch.tensor([10, 9, 8, 7])  # 实际长度
    vision_lengths = torch.tensor([8, 7, 6, 5])

    # 创建模型（使用注意力池化+门控融合）
    model = ImprovedAudioVisualBiLSTM(
        audio_input_dim=audio_dim,
        vision_input_dim=vision_dim,
        num_classes=num_classes,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        rnn_type='gru',
        pooling='attention',          # 使用注意力池化
        fusion='gated',                # 门控融合
        fusion_output_dim=128,
        classifier_hidden_dims=[64, 32],
        use_batchnorm=True,
    )

    logits = model(audio, vision, audio_lengths, vision_lengths)
    print(logits.shape)  # [4, 5]