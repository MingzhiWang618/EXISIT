import torch
import torch.nn as nn
import torch.nn.functional as F
# 這裡導入你原始 Teacher.py 中的組件
from .Teacher import ModalityEncoder, SGCN, AttentionBiLSTM, build_classifier

class AMBOKDFusion(nn.Module):
    """
    論文 Section IV-B：多頭自注意力融合模組
    將 EEG 和 AV 特徵對齊並融合
    """
    def __init__(self, eeg_dim, av_dim, align_dim=64, num_heads=2):
        super().__init__()
        self.fc_eeg = nn.Linear(eeg_dim, align_dim)
        self.fc_av = nn.Linear(av_dim, align_dim)
        self.concat_dim = align_dim * 2
        self.attn = nn.MultiheadAttention(embed_dim=self.concat_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, eeg_feat, av_feat):
        # 特徵投影對齊
        e_proj = F.relu(self.fc_eeg(eeg_feat))
        a_proj = F.relu(self.fc_av(av_feat))
        # 拼接並增加序列維度 [B, 1, D]
        fused = torch.cat([e_proj, a_proj], dim=-1).unsqueeze(1)
        # 自注意力計算
        attn_out, _ = self.attn(fused, fused, fused)
        return attn_out.squeeze(1)

class AMBOKDManager(nn.Module):
    """
    AMBOKD 損失管理類 (算法 1 核心實現)
    """
    def __init__(self, temperature=4.0, gamma=3.0):
        super().__init__()
        self.T = temperature
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 用於動態梯度調製的基線損失 (L_base)
        self.register_buffer('L_base_eeg', torch.zeros(1))
        self.register_buffer('L_base_av', torch.zeros(1))
        self.register_buffer('L_base_fused', torch.zeros(1))
        
        # 用於收集 Epoch 1 的數據
        self.epoch1_stats = {'eeg': [], 'av': [], 'fused': []}

    def _kl_div(self, s_logits, t_logits):
        """計算蒸餾損失"""
        p_s = F.log_softmax(s_logits / self.T, dim=-1)
        p_t = F.softmax(t_logits.detach() / self.T, dim=-1)
        return F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)

    def forward(self, preds, labels, epoch):
        # 提取各分支 Logits
        l_eeg, l_av, l_fused = preds['logits_eeg'], preds['logits_av'], preds['logits_fused']
        
        # 1. 計算標準交叉熵 (CE)
        ce_eeg = self.ce_loss(l_eeg, labels)
        ce_av = self.ce_loss(l_av, labels)
        ce_fused = self.ce_loss(l_fused, labels)

        # 2. 動態權重調製 (Dynamic Weights Modulation)
        # 計算 alpha 和 beta (使用 .item() 阻斷梯度，僅作為係數)
        with torch.no_grad():
            def get_w(s_ce, t_ce): return torch.clamp(s_ce / (t_ce + 1e-6), 0.1, 10.0)
            
            w_e_from_f = get_w(ce_eeg, ce_fused)
            w_e_from_a = get_w(ce_eeg, ce_av)
            
            w_a_from_f = get_w(ce_av, ce_fused)
            w_a_from_e = get_w(ce_av, ce_eeg)
            
            w_f_from_e = get_w(ce_fused, ce_eeg)
            w_f_from_a = get_w(ce_fused, ce_av)

        # 各分支 Total Loss (CE + 蒸餾)
        loss_eeg   = ce_eeg   + w_e_from_f * self._kl_div(l_eeg, l_fused)   + w_e_from_a * self._kl_div(l_eeg, l_av)
        loss_av    = ce_av    + w_a_from_f * self._kl_div(l_av, l_fused)    + w_a_from_e * self._kl_div(l_av, l_eeg)
        loss_fused = ce_fused + w_f_from_e * self._kl_div(l_fused, l_eeg)   + w_f_from_a * self._kl_div(l_fused, l_av)

        # 3. 動態梯度調製 (Dynamic Gradients Modulation)
        r_dg = {'eeg': 1.0, 'av': 1.0, 'fused': 1.0}
        
        if epoch == 1:
            # 收集第一輪的 CE 用於計算 L_base
            self.epoch1_stats['eeg'].append(ce_eeg.item())
            self.epoch1_stats['av'].append(ce_av.item())
            self.epoch1_stats['fused'].append(ce_fused.item())
        else:
            # Epoch 2 開始時計算一次 L_base
            if epoch == 2 and len(self.epoch1_stats['eeg']) > 0:
                self.L_base_eeg[0]   = sum(self.epoch1_stats['eeg']) / len(self.epoch1_stats['eeg'])
                self.L_base_av[0]    = sum(self.epoch1_stats['av']) / len(self.epoch1_stats['av'])
                self.L_base_fused[0] = sum(self.epoch1_stats['fused']) / len(self.epoch1_stats['fused'])
                self.epoch1_stats = {'eeg': [], 'av': [], 'fused': []} # 清空

            # 計算優化進度 R_S (Eq. 13)
            with torch.no_grad():
                rs_e = (self.L_base_eeg - ce_eeg) / (self.L_base_eeg + 1e-6)
                rs_a = (self.L_base_av - ce_av) / (self.L_base_av + 1e-6)
                rs_f = (self.L_base_fused - ce_fused) / (self.L_base_fused + 1e-6)
                
                # 計算梯度乘子 (Eq. 14)
                r_dg['eeg']   = torch.clamp(((rs_f + rs_a) / (2 * rs_e + 1e-6))**self.gamma, 0.1, 10.0)
                r_dg['av']    = torch.clamp(((rs_f + rs_e) / (2 * rs_a + 1e-6))**self.gamma, 0.1, 10.0)
                r_dg['fused'] = torch.clamp(((rs_e + rs_a) / (2 * rs_f + 1e-6))**self.gamma, 0.1, 10.0)

        # 最終加權總損失
        total_loss = (r_dg['eeg'] * loss_eeg + r_dg['av'] * loss_av + r_dg['fused'] * loss_fused) / 3.0
        
        return {
            'loss': total_loss,
            'ce_fused': ce_fused.detach(),
            'ce_eeg': ce_eeg.detach(),
            'ce_av': ce_av.detach()
        }

class AMBOKDModel(nn.Module):
    """
    整合後的 AMBOKD 模型：前向傳播 + 自動損失計算
    """
    def __init__(self, cfg):
        super().__init__()
        # 模態編碼器 (復用 Teacher.py 中的結構)
        self.audio_enc = ModalityEncoder(cfg.audio_dim, cfg.av_hidden, cfg.lstm_layers, cfg.dropout)
        self.vision_enc = ModalityEncoder(cfg.vision_dim, cfg.av_hidden, cfg.lstm_layers, cfg.dropout)
        
        self.sgcn = SGCN(cfg.num_nodes, cfg.in_features, cfg.gcn_hidden, cfg.gcn_out, cfg.dropout)
        self.abilstm = AttentionBiLSTM(cfg.num_nodes * cfg.gcn_out, cfg.lstm_hidden, cfg.lstm_layers, cfg.dropout)
        
        # 融合與分類
        eeg_dim = cfg.lstm_hidden * 2
        av_dim = self.audio_enc.out_dim + self.vision_enc.out_dim
        self.fusion = AMBOKDFusion(eeg_dim, av_dim)
        
        self.clf_eeg = build_classifier(eeg_dim, cfg.fc_hidden, cfg.num_classes, cfg.dropout)
        self.clf_av = build_classifier(av_dim, cfg.fc_hidden, cfg.num_classes, cfg.dropout)
        self.clf_fused = build_classifier(self.fusion.concat_dim, cfg.fc_hidden, cfg.num_classes, cfg.dropout)
        
        # 內置 AMBOKD 損失管理器
        self.criterion = AMBOKDManager(temperature=cfg.temperature)

    def forward(self, eeg, pcc, audio, vision, labels=None, epoch=None):
        # 1. 提取特徵
        feat_a = self.audio_enc(audio)
        feat_v = self.vision_enc(vision)
        feat_av = torch.cat([feat_a, feat_v], dim=-1)
        
        gcn_out, _ = self.sgcn(eeg, pcc)
        feat_eeg, _ = self.abilstm(gcn_out)
        
        feat_fused = self.fusion(feat_eeg, feat_av)
        
        # 2. 獲取 Logits
        preds = {
            'logits_eeg': self.clf_eeg(feat_eeg),
            'logits_av': self.clf_av(feat_av),
            'logits_fused': self.clf_fused(feat_fused)
        }
        
        # 3. 判斷是否計算 Loss
        if labels is not None and epoch is not None:
            output = self.criterion(preds, labels, epoch)
            output.update(preds) # 合併 logits 方便算 Accuracy
            return output
        
        return preds