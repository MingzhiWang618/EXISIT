import unittest
import torch
from exist_method.distillation import DistillationConfig, StableDistillationLoss


class StableDistillationTests(unittest.TestCase):
    def test_identical_distributions_have_zero_distillation(self):
        logits = torch.randn(3, 4, requires_grad=True)
        spatial = torch.softmax(torch.randn(3, 5, 8, 8), -1)
        temporal = torch.softmax(torch.randn(3, 5), -1)
        student = {"logits": logits, "S_attn": spatial, "attn_t": temporal}
        teacher = {"S_attn": spatial.clone(), "attn_t": temporal.clone()}
        out = StableDistillationLoss()(student, teacher, torch.tensor([0,1,2]), 1)
        self.assertAlmostEqual(out["cdd"].item(), 0.0, places=5)
        self.assertAlmostEqual(out["edd"].item(), 0.0, places=5)

    def test_gradients_are_finite(self):
        s_logits = torch.randn(4, 5, requires_grad=True)
        s_spatial_logits = torch.randn(4, 5, 30, 30, requires_grad=True)
        s_temporal_logits = torch.randn(4, 5, requires_grad=True)
        student = {"logits":s_logits, "S_attn":s_spatial_logits.softmax(-1),
                   "attn_t":s_temporal_logits.softmax(-1)}
        teacher = {"S_attn":torch.randn(4,5,30,30).softmax(-1),
                   "attn_t":torch.randn(4,5).softmax(-1)}
        loss = StableDistillationLoss(DistillationConfig())(student, teacher, torch.arange(4), 5)["loss"]
        loss.backward()
        for tensor in (s_logits, s_spatial_logits, s_temporal_logits):
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_warmup_strength(self):
        criterion = StableDistillationLoss(DistillationConfig(alpha=.5, warmup_epochs=10))
        s = {"logits":torch.randn(2,3), "S_attn":torch.ones(2,2,4,4)/4, "attn_t":torch.ones(2,2)/2}
        t = {"S_attn":torch.ones(2,2,4,4)/4, "attn_t":torch.ones(2,2)/2}
        self.assertAlmostEqual(criterion(s,t,torch.tensor([0,1]),1)["strength"].item(), .05, places=6)
        self.assertAlmostEqual(criterion(s,t,torch.tensor([0,1]),20)["strength"].item(), .5, places=6)

    def test_logit_distillation_contributes_to_loss(self):
        s = {"logits":torch.randn(2,3), "S_attn":torch.ones(2,2,4,4)/4,
             "attn_t":torch.ones(2,2)/2}
        t = {"logits":torch.randn(2,3), "S_attn":torch.ones(2,2,4,4)/4,
             "attn_t":torch.ones(2,2)/2}
        out = StableDistillationLoss(DistillationConfig(alpha=0, logit_weight=.5))(
            s,t,torch.tensor([0,1]),1)
        self.assertGreaterEqual(out["logit"].item(), 0)
        self.assertAlmostEqual(out["loss"].item(),
                               (out["ce"]+.5*out["logit"]).item(), places=6)


if __name__ == "__main__": unittest.main()
