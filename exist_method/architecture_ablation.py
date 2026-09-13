"""Runtime model switches used by controlled PCC and alpha-injection ablations."""
import torch
import torch.nn.functional as F


def configure_teacher(module, injection="column", use_pcc=True):
    if injection not in {"row", "element", "column"}:
        raise ValueError(f"invalid injection mode: {injection}")

    def spatial_forward(self, x, av_bt):
        q, k = self.Wq(x), self.Wk(x)
        scores = torch.bmm(q, k.transpose(1, 2))
        alpha = self.av_cross_attn(av_bt, x)
        if injection == "column":
            offset = alpha.unsqueeze(1)
        elif injection == "row":
            offset = alpha.unsqueeze(2)
        else:
            offset = alpha.unsqueeze(2) * alpha.unsqueeze(1)
        return F.softmax((scores + offset) / self.temperature, dim=-1)

    def sgcn_forward(self, x, pcc, av_ctx):
        batch, windows, nodes, features = x.shape
        flat = x.reshape(batch * windows, nodes, features)
        av_bt = av_ctx.unsqueeze(1).expand(batch, windows, -1).reshape(batch * windows, -1)
        attention = self.spatial_attn(flat, av_bt)
        adjacency = attention
        if use_pcc:
            adjacency = adjacency * F.relu(pcc).reshape(batch * windows, nodes, nodes)
        adjacency = adjacency + torch.eye(nodes, device=x.device).unsqueeze(0)
        hidden = self.gcn2(self.gcn1(flat, adjacency), adjacency)
        return hidden.reshape(batch, windows, nodes * self.out_dim), attention.view(batch, windows, nodes, nodes)

    module.SpatialAttention.forward = spatial_forward
    module.SGCN.forward = sgcn_forward
    return module.TeacherModel


def configure_student(module, use_pcc=True):
    def sgcn_forward(self, x, pcc):
        batch, windows, nodes, features = x.shape
        flat = x.reshape(batch * windows, nodes, features)
        attention = self.spatial_attn(flat)
        adjacency = attention
        if use_pcc:
            adjacency = adjacency * F.relu(pcc).reshape(batch * windows, nodes, nodes)
        adjacency = adjacency + torch.eye(nodes, device=x.device)
        hidden = self.gcn2(self.gcn1(flat, adjacency), adjacency)
        return hidden.reshape(batch, windows, nodes * self.out_dim), attention.view(batch, windows, nodes, nodes)

    module.SGCN.forward = sgcn_forward
    return module.ST_GCLSTM
