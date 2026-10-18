import math
from typing import Optional
import torch
import numpy as np
from torch import nn
from math import pi
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super(BesselBasisLayer, self).__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        freq_init = np.pi * torch.arange(1, num_radial + 1)
        self.freq = nn.Parameter(freq_init)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class ExpNormalSmearing(nn.Module):
    def __init__(self, num_rbf=128, cutoff_lower=0.0, cutoff_upper=5.0, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff()
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(-dist) - self.means) ** 2)
# class ExpNormalSmearing(nn.Module):
#     def __init__(self, num_rbf=128, cutoff_lower=0.0, cutoff_upper=5.0, trainable=False):
#         super(ExpNormalSmearing, self).__init__()
#         self.num_rbf = num_rbf
#
#         # 最简单的实现：使用固定的正弦余弦编码
#         self.register_buffer('freqs', torch.arange(1, num_rbf + 1).float())
#
#     def forward(self, dist):
#         dist = dist.unsqueeze(-1)  # (num_edges, 1)
#         # 简单的正弦余弦编码
#         encoding = torch.sin(dist * self.freqs) + torch.cos(dist * self.freqs)
#         return encoding


def Pair_distance(pos, edge_index):
    row, col = edge_index
    dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
    return dist


def weight_down(x):
    '''
    x up , return W down
    for molecular cutoff = 5.0
    '''
    W =  0.5 * (torch.cos(x * math.pi / 5.0) + 1.0)
    W = W * (x < 5.0).float()    # distance > 5.0 == 0
    return W


def weight_up(x):
    '''
    X up, return W up
    '''
    W = 0.5 * (-torch.cos(x * math.pi / 4.0) + 1.0)
    return W

