import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Embedding
from torch_geometric.nn import MessagePassing, radius_graph
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from math import pi as PI

from gtdiff.utils.chem import BOND_TYPES
from ..common import MeanReadout, SumReadout, MultiLayerPerceptron
import torch.nn as nn
from torch.nn import Sequential, Linear
from torch.nn.functional import softmax
from .utils import ExpNormalSmearing, CosineCutoff, BesselBasisLayer
import math
from math import pi as PI


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=128):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class AsymmetricSineCosineSmearing(Module):

    def __init__(self, num_basis=50):
        super().__init__()
        num_basis_k = num_basis // 2
        num_basis_l = num_basis - num_basis_k
        self.register_buffer('freq_k', torch.arange(1, num_basis_k + 1).float())
        self.register_buffer('freq_l', torch.arange(1, num_basis_l + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0) + self.freq_l.size(0)

    def forward(self, angle):
        # If we don't incorporate `cos`, the embedding of 0-deg and 180-deg will be the
        #  same, which is undesirable.
        s = torch.sin(angle.view(-1, 1) * self.freq_k.view(1, -1))  # (num_angles, num_basis_k)
        c = torch.cos(angle.view(-1, 1) * self.freq_l.view(1, -1))  # (num_angles, num_basis_l)
        return torch.cat([s, c], dim=-1)
    
class SymmetricCosineSmearing(Module):

    def __init__(self, num_basis=50):
        super().__init__()
        self.register_buffer('freq_k', torch.arange(1, num_basis+1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0)

    def forward(self, angle):
        return torch.cos(angle.view(-1, 1) * self.freq_k.view(1, -1))   # (num_angles, num_basis)


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))  # Learnable parameter
        self.shift = torch.log(torch.tensor(2.0))

    def forward(self, x):
        # Manually apply the beta scaling and softplus
        scaled_x = self.beta * x
        return F.softplus(scaled_x) - self.shift

class DistanceWeightingNetwork(nn.Module):
    def __init__(self, hidden_dim=32):
        super(DistanceWeightingNetwork, self).__init__()
        self.layer1 = Linear(1, hidden_dim)
        self.layer2 = Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, edge_length):
        edge_length = edge_length.unsqueeze(-1)  # Ensure edge_length is properly shaped for linear layers
        x = self.activation(self.layer1(edge_length))
        weights = torch.sigmoid(self.layer2(x))  # Use sigmoid to ensure output weights are between 0 and 1
        return weights.squeeze(-1)  # Remove the last dimension to match the expected shape

class AttentionModule(nn.Module):
    def __init__(self, feature_size):
        super(AttentionModule, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(feature_size))

    def forward(self, nn_output, weights):
        context = torch.sigmoid(self.attention_weights)  # Contextual modulation
        return nn_output * context + weights * (1 - context)

class GraphTransformerLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, seq, cutoff, smooth, n_heads, hidden_dim=32):
        """
        初始化 CFConv 模块
        :param in_channels: 输入特征维度
        :param out_channels: 输出特征维度
        :param num_filters: 注意力头总特征维度
        :param seq: 用于边特征的 MLP
        :param cutoff: 截断距离
        :param smooth: 是否平滑截断函数
        :param n_heads: 注意力头数
        :param alpha: 一个可学习的参数（可能与 pd 相关）
        """
        super(GraphTransformerLayer, self).__init__(aggr='add')  # 聚合方式为加和
        self.n_heads = n_heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.d_heads = num_filters // n_heads  # 每个注意力头的维度
        # self.alpha = nn.Parameter(torch.tensor([1.0]), requires_grad=True)  # 可学习的参数
        # self.distance_weighting = DistanceWeightingNetwork(hidden_dim)
        # 定义多头注意力的 Q, K, V 线性变换
        self.lin_q = nn.Linear(in_channels, num_filters, bias=False)
        self.lin_k = nn.Linear(in_channels, num_filters, bias=False)
        self.lin_v = nn.Linear(in_channels, num_filters, bias=False)

        # 用于 pd 的处理
        self.lin_dk = nn.Linear(in_channels, num_filters, bias=False)
        self.lin_dv = nn.Linear(in_channels, num_filters, bias=False)


        # 输出线性变换
        self.lin_out = nn.Linear(num_filters, out_channels)

        # 用于处理边特征的序列模型（MLP）
        self.seq = seq
        self.cutoff = cutoff
        self.cosine = CosineCutoff()
        self.smooth = smooth
        self.silu = torch.nn.SiLU()  # 激活函数
        self.distance_expansion = ExpNormalSmearing()

        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化参数
        """
        torch.nn.init.xavier_uniform_(self.lin_q.weight)
        torch.nn.init.xavier_uniform_(self.lin_k.weight)
        torch.nn.init.xavier_uniform_(self.lin_v.weight)
        torch.nn.init.xavier_uniform_(self.lin_dk.weight)
        torch.nn.init.xavier_uniform_(self.lin_dv.weight)
        torch.nn.init.xavier_uniform_(self.lin_out.weight)
        if self.lin_out.bias is not None:
            self.lin_out.bias.data.fill_(0)



    def forward(self, x, edge_index, edge_length, edge_attr):
        """
        前向传播
        :param x: 节点特征 (num_nodes, in_channels)
        :param edge_index: 边索引 (2, num_edges)
        :param edge_length: 边的长度 (num_edges,)
        :param edge_attr: 边特征 (num_edges, edge_attr_dim)
        :param pd: 如果存在，则为边特征的一部分（比如 pd = edge_length）
        """
        # 计算截断权重（C）
        # learnable_weights = self.distance_weighting(edge_length)
        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)
        else:
            C = (edge_length <= self.cutoff).float()

        # 归一化边特征
        W = self.seq(edge_attr) * C.view(-1, 1)
        W = W / (W.norm(dim=-1, keepdim=True) + 1e-6)  # L2 归一化

        # 计算多头注意力的 Q, K, V
        q = self.lin_q(x).view(-1, self.n_heads, self.d_heads)
        k = self.lin_k(x).view(-1, self.n_heads, self.d_heads)
        v = self.lin_v(x).view(-1, self.n_heads, self.d_heads)

        # 传播计算
        x = self.propagate(edge_index, q=q, k=k, v=v, W=W, edge_length=edge_length, size=(x.size(0), x.size(0)))
        x = x.reshape(-1, self.num_filters)  # 合并多头结果

        x = self.lin_out(x)  # 输出线性变换
        return x

    def message(self, q, k, v, edge_index, W, edge_length):
        """
        消息传递阶段
        :param q: 查询向量 Q
        :param k: 键向量 K
        :param v: 值向量 V
        :param edge_index: 边索引
        :param W: 边权重
        :param edge_length: 边长度
        :param pd: 边特征，用于额外的处理
        """
        # 索引边上的 q, k, v
        q_i = q[edge_index[0]]
        k_j = k[edge_index[1]]
        v_j = v[edge_index[1]]

        # 如果 pd（边长度）存在，进行处理

        f_pd = self.distance_expansion(edge_length)  # 假设这个函数用来处理 pd
        f_pd =f_pd.squeeze(1)

        f_pd_k = self.silu(self.lin_dk(f_pd)).reshape(-1, self.n_heads, self.d_heads)
        f_pd_v = self.silu(self.lin_dv(f_pd)).reshape(-1, self.n_heads, self.d_heads)


        attn = ((q_i * k_j * f_pd_k) / math.sqrt(self.d_heads)).sum(-1)
        attn = attn - attn.max(dim=-1, keepdim=True)[0]  # 数值稳定
        attn_scores = torch.softmax(attn, dim=-1)  # 归一化
        v = v_j * f_pd_v

        attn_scores = attn_scores.unsqueeze(-1)  # (num_edges, n_heads, 1)
        W = W.view(-1, self.n_heads, self.d_heads)  # 调整 W 的形状
        message_weights = attn_scores * W  # 广播相乘
        message_weights = message_weights / (message_weights.norm(dim=-1, keepdim=True) + 1e-6)  # 归一化

        # 消息加权
        message = v * message_weights
        message = message.view(message.size(0), -1)  # 展平多头维度
        return message




class TransformerInteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth,n_heads):
        super(TransformerInteractionBlock, self).__init__()
        mlp1 = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),

            nn.Linear(num_filters, num_filters),
        )
        mlp2 = nn.Sequential(
            nn.Linear(num_gaussians, num_filters // 2),  # Note: Adjusted the size here
            ShiftedSoftplus(),

            nn.Linear(num_filters // 2, num_filters // 2),
        )

        self.Layer1 = GraphTransformerLayer(hidden_channels, hidden_channels, num_filters, mlp1, cutoff, smooth,n_heads)
        self.Layer2 = GraphTransformerLayer(hidden_channels, hidden_channels, num_filters//2, mlp2, cutoff, smooth, n_heads)
        self.Lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(256, hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):

        x1 = self.Layer1(x, edge_index, edge_length, edge_attr)
        x2 = self.Layer2(x, edge_index, edge_length, edge_attr)


        x= torch.cat([x1, x2], dim=-1)
        x = self.act(x)

        x = self.lin(x)
        weights = self.Lin(x)
        x = x * weights

        return x


class GraphTransformerEncoder(nn.Module):

    def __init__(self, hidden_channels=128, num_filters=128,
                num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False,n_heads=8):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff

        self.embedding = nn.Embedding(100, hidden_channels, max_norm=10.0)

        self.interactions = nn.ModuleList()

        for _ in range(num_interactions):
            block = TransformerInteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth, n_heads)
            self.interactions.append(block)


    def forward(self, z, edge_index, edge_length, edge_attr, embed_node=True):
        # pd=edge_length
        if embed_node:
            assert z.dim() == 1 and z.dtype == torch.long
            h = self.embedding(z)
        else:
            h = z


        for interaction in self.interactions:

            h = h + interaction(h, edge_index, edge_length, edge_attr)

        return h


