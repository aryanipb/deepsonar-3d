"""Pure-PyTorch replacements for the CUDA extensions used by the official PoinTr code
(pointnet2_ops, chamfer_dist) so the unmodified AdaPoinTr weights run on CPU and GPU."""
from types import SimpleNamespace

import torch
import torch.nn as nn


def print_log(msg, logger=None):
    pass


class _Registry:
    def register_module(self):
        return lambda cls: cls


MODELS = _Registry()


def furthest_point_sample(xyz, npoint):
    """xyz: B N 3 -> int32 idx B npoint (deterministic, starts at index 0 like the CUDA op)."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device, dtype=xyz.dtype)
    farthest = torch.zeros(B, dtype=torch.long, device=xyz.device)
    batch = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)
        dist = torch.minimum(dist, ((xyz - centroid) ** 2).sum(-1))
        farthest = dist.argmax(-1)
    return idx.int()


def gather_operation(features, idx):
    """features: B C N, idx: B S -> B C S"""
    idx = idx.long()
    return torch.gather(features, 2, idx.unsqueeze(1).expand(-1, features.size(1), -1))


def three_nn(unknown, known):
    """unknown: B n 3, known: B m 3 -> (dist B n 3, idx B n 3); dist is euclidean like the CUDA op."""
    d = torch.cdist(unknown, known)
    dist, idx = torch.topk(d, 3, dim=-1, largest=False)
    return dist, idx.int()


def three_interpolate(features, idx, weight):
    """features: B C m, idx/weight: B n 3 -> B C n"""
    B, C, _ = features.shape
    n = idx.size(1)
    g = gather_operation(features, idx.reshape(B, -1)).reshape(B, C, n, 3)
    return (g * weight.unsqueeze(1)).sum(-1)


pointnet2_utils = SimpleNamespace(
    furthest_point_sample=furthest_point_sample,
    gather_operation=gather_operation,
    three_nn=three_nn,
    three_interpolate=three_interpolate,
)


def _fps(data, number):
    idx = furthest_point_sample(data, number)
    return gather_operation(data.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()


def _jitter_points(pc, std=0.01, clip=0.05):
    return pc + torch.randn_like(pc).mul_(std).clamp_(-clip, clip)


misc = SimpleNamespace(fps=_fps, jitter_points=_jitter_points)


def _nn_idx(a, b, chunk=2048):
    """Index of nearest neighbour in b for every point of a, computed in chunks without autograd."""
    with torch.no_grad():
        return torch.cat([torch.cdist(a[:, i:i + chunk], b).argmin(-1) for i in range(0, a.size(1), chunk)], 1)


def chamfer_sqdist(p1, p2):
    """Per-point squared NN distances, matching chamfer_dist (which returns squared distances).
    Memory-light: NN search is chunked, gradients flow only through the matched pairs (as in the CUDA op)."""
    i12 = _nn_idx(p1, p2)
    i21 = _nn_idx(p2, p1)
    d1 = ((p1 - torch.gather(p2, 1, i12.unsqueeze(-1).expand(-1, -1, 3))) ** 2).sum(-1)
    d2 = ((p2 - torch.gather(p1, 1, i21.unsqueeze(-1).expand(-1, -1, 3))) ** 2).sum(-1)
    return d1, d2


class ChamferDistanceL1(nn.Module):
    """Same definition as extensions/chamfer_dist ChamferDistanceL1: (mean sqrt d1 + mean sqrt d2) / 2."""

    def forward(self, xyz1, xyz2):
        d1, d2 = chamfer_sqdist(xyz1, xyz2)
        return (torch.sqrt(d1.clamp_min(1e-12)).mean() + torch.sqrt(d2.clamp_min(1e-12)).mean()) / 2


trunc_normal_ = nn.init.trunc_normal_


class DropPath(nn.Module):
    """Stochastic depth, equivalent to timm.layers.DropPath."""

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep
