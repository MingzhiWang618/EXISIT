"""Numerically stable CDD/EDD objectives with controlled gradient scale."""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillationConfig:
    alpha: float = 0.5
    cdd_ratio: float = 0.5
    edd_ratio: float = 0.5
    warmup_epochs: int = 10
    cdd_temperature: float = 2.0
    edd_temperature: float = 1.0
    ema_decay: float = 0.95
    confidence_floor: float = 0.1
    eps: float = 1e-6


def soften(probability, temperature, eps):
    return F.softmax(probability.clamp_min(eps).log() / temperature, dim=-1)


def kl_rows(student, teacher, eps):
    student = student.clamp_min(eps)
    teacher = teacher.clamp_min(eps)
    student = student / student.sum(-1, keepdim=True)
    teacher = teacher / teacher.sum(-1, keepdim=True)
    return F.kl_div(student.log(), teacher.detach(), reduction="none").sum(-1)


def confidence(probability, floor, eps):
    dimension = probability.shape[-1]
    entropy = -(probability.clamp_min(eps) * probability.clamp_min(eps).log()).sum(-1)
    return (1.0 - entropy / math.log(dimension)).clamp(floor, 1.0)


class StableDistillationLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or DistillationConfig()
        self.register_buffer("ema_cdd", torch.tensor(float("nan")))
        self.register_buffer("ema_edd", torch.tensor(float("nan")))

    def _update_ema(self, name, value):
        state = getattr(self, name)
        detached = value.detach()
        if torch.isnan(state):
            state.copy_(detached)
        else:
            state.mul_(self.config.ema_decay).add_(detached, alpha=1-self.config.ema_decay)

    def forward(self, student, teacher, labels, epoch):
        cfg = self.config
        ce = F.cross_entropy(student["logits"], labels)
        s_cdd = soften(student["S_attn"], cfg.cdd_temperature, cfg.eps)
        t_cdd = soften(teacher["S_attn"].detach(), cfg.cdd_temperature, cfg.eps)
        cdd_each = kl_rows(s_cdd, t_cdd, cfg.eps)
        cdd_weight = confidence(t_cdd, cfg.confidence_floor, cfg.eps)
        cdd = (cdd_each*cdd_weight).sum() / cdd_weight.sum().clamp_min(cfg.eps)
        s_edd = soften(student["attn_t"], cfg.edd_temperature, cfg.eps)
        t_edd = soften(teacher["attn_t"].detach(), cfg.edd_temperature, cfg.eps)
        edd_each = kl_rows(s_edd, t_edd, cfg.eps)
        edd_weight = confidence(t_edd, cfg.confidence_floor, cfg.eps)
        edd = (edd_each*edd_weight).sum() / edd_weight.sum().clamp_min(cfg.eps)
        if self.training:
            self._update_ema("ema_cdd", cdd)
            self._update_ema("ema_edd", edd)
        normalized_cdd = cdd / self.ema_cdd.clamp_min(cfg.eps).detach()
        normalized_edd = edd / self.ema_edd.clamp_min(cfg.eps).detach()
        strength = cfg.alpha * min(1.0, epoch / max(cfg.warmup_epochs, 1))
        distillation = strength * (cfg.cdd_ratio*normalized_cdd + cfg.edd_ratio*normalized_edd)
        total = ce + distillation
        return {"loss": total, "ce": ce, "cdd": cdd, "edd": edd,
                "normalized_cdd": normalized_cdd, "normalized_edd": normalized_edd,
                "distillation": distillation, "strength": total.new_tensor(strength)}
