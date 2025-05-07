
###################NO_TEARS##############################

import math
import torch
import torch.nn as nn
from nns.case.mat_gru import MatGRU
from nns.case.dag_gen_block import DGBlockV1, DAGGumbelSigmoid
from nns.case.sem_block import MaskGCNSEM
from nns.case.feat_blocks import feat_embed_block
from utils.graph_utils import spatial_norm_tensor

# ── REMOVED NOTEARS post‐processing import ──
# from nns.post_processing_blocks import PostProcessingBlocV2

# ── NEW: import the DAGMA projector ──
from dagma import DAGMA


class DagGenGRUg4s2v1(nn.Module):
    """
    Causal DAG generator with DAGMA replacing NOTEARS.
    """
    def __init__(self,
                 num_nodes, in_feats_dim, out_feats_dim,
                 hidden_dim, num_layers=2, num_heads=4,
                 feats_layers=3, dist_adj=None, agg_feats='ori',
                 node_norm=False, use_norm=False,
#                use_pp=False, step_pri=0.01, step_dual=0.01,
                 # ── REPLACED NOTEARS args with DAGMA args ──
                 use_dagma=True, dagma_mu=1.0, dagma_alpha=0.1, dagma_steps=10,
                 reg_sp_intra=2e-3, num_intra_pp_iters=1000, **kwargs):
        super(DagGenGRUg4s2v1, self).__init__()
        GRU_FOLD = 4

        self.num_nodes    = num_nodes
        self.in_feats_dim = in_feats_dim
        self.out_feats_dim= out_feats_dim
        self.hidden_dim   = hidden_dim
        self.num_layers   = num_layers
        self.num_heads    = num_heads
        self.use_norm     = use_norm

        assert hidden_dim % num_heads == 0

        # 1) Feature embedding block
        self.embed = feat_embed_block(
            num_nodes, in_feats_dim, hidden_dim,
            num_layers=feats_layers,
            dist_adj=dist_adj, agg_feats=agg_feats,
            node_norm=node_norm
        )

        # 2) Q/K projections for inter/intra attention
        self.W_inter_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_inter_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_intra_q= nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_intra_k= nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 3) GRUs over flattened N^2 “pixel” scores
        self.graph_gru_inter = MatGRU(num_nodes**2, num_heads, num_heads*GRU_FOLD, num_layers)
        self.graph_gru_intra = MatGRU(num_nodes**2, num_heads, num_heads*GRU_FOLD, num_layers)

        # 4) MLP heads generating logits per edge
        self.graph_gen_inter = DGBlockV1(num_heads*GRU_FOLD, hidden_dim, 1)
        self.graph_gen_intra = DGBlockV1(num_heads*GRU_FOLD, hidden_dim, 1)

        # 5) Gumbel‐Sigmoid for near-binary edges
        self.gumbel_sigmoid = DAGGumbelSigmoid()

        # ── NEW: DAGMA setup ──
        self.use_dagma   = use_dagma
        self.dagma_mu    = float(dagma_mu)
        self.dagma_alpha = float(dagma_alpha)
        self.dagma_steps = int(dagma_steps)
        # instantiate the DAGMA central-path projector
        self.dagma_solver = DAGMA(mu=self.dagma_mu,
                                  alpha=self.dagma_alpha,
                                  steps=self.dagma_steps)

        # 6) SEM encoder for structure-learning reconstruction loss
        self.sem_encoder = MaskGCNSEM(
            num_nodes, hidden_dim, out_feats_dim, hidden_dim
        )

    def split_heads(self, x):
        # x: (T, B, N, H) → (T, B, heads, N, H//heads)
        T, B, N, _ = x.shape
        x = x.reshape(T, B, N, self.num_heads, -1)
        return x.permute(0, 1, 3, 2, 4)

    @staticmethod
    def flat_pixels(x):
        # x: (T, B, C, N, N) → (T, B, N^2, C)
        T, B, C, N, _ = x.shape
        return x.reshape(T, B, C, -1).transpose(-1, -2)

    @staticmethod
    def unflat_pixels(x):
        # x: (T, B, N^2, C) → (T, B, C, N, N)
        T, B, N2, C = x.shape
        N = int(math.sqrt(N2))
        x = x.reshape(T, B, N, N, C)
        return x.permute(0, 1, 4, 2, 3)

    def forward(self, x_with_pre, gen_graph_only=False):
        # x_with_pre: (T+1, B, N, D)
        # 1) Embed features
        x_with_pre = self.embed(x_with_pre)  # (T+1, B, N, H)

        # 2) Split into “previous” vs. “current”
        pre_x = x_with_pre[:-1, ...]  # (T, B, N, H)
        x     = x_with_pre[1:, ...]   # (T, B, N, H)

        # 3) Compute multi-head Q/K for inter & intra
        inter_q = self.split_heads(self.W_inter_q(x))
        inter_k = self.split_heads(self.W_inter_k(pre_x))
        intra_q = self.split_heads(self.W_intra_q(x))
        intra_k = self.split_heads(self.W_intra_k(pre_x))

        # 4) Scores = Q·Kᵀ → flatten to “pixels”
        pre_feats = self.flat_pixels(torch.matmul(inter_q, inter_k.transpose(-1, -2)))
        cur_feats = self.flat_pixels(torch.matmul(intra_q, intra_k.transpose(-1, -2)))

        # 5) Temporal GRU aggregation
        h_inter, _ = self.graph_gru_inter(pre_feats)
        h_intra, _ = self.graph_gru_intra(cur_feats)

        # 6) MLP → logits → unflatten → (T, B, N, N)
        inter_graph = self.unflat_pixels(self.graph_gen_inter(h_inter)).squeeze(2)
        intra_graph = self.unflat_pixels(self.graph_gen_intra(h_intra)).squeeze(2)

        # 7) Gumbel‐Sigmoid thresholding
        inter_graph = self.gumbel_sigmoid(inter_graph)
        intra_graph = self.gumbel_sigmoid(intra_graph, mask=True)

        # ── REPLACED NOTEARS post‐processing with DAGMA central‐path solver ──
        if self.use_dagma:
            T, B, N, _ = intra_graph.shape
            # flatten batch/time and move to CPU numpy
            flat = intra_graph.reshape(-1, N, N).detach().cpu().numpy()
            # project each adjacency via DAGMA
            proj = self.dagma_solver.project(flat)  # returns (T*B, N, N) numpy array
            # restore tensor shape
            intra_graph = (
                torch.from_numpy(proj)
                     .to(x_with_pre.device)
                     .view(T, B, N, N)
            )

        # 8) Stack inter & intra: (T, B, 2, N, N)
        graphs = torch.stack([inter_graph, intra_graph], dim=2)

        if gen_graph_only:
            return graphs, None

        # 9) SEM reconstruction for structure‐learning loss
        if self.use_norm:
            normed = spatial_norm_tensor(graphs, add_self_loops=False)
            reconst = self.sem_encoder(x_with_pre, normed)
        else:
            reconst = self.sem_encoder(x_with_pre, graphs)

        return graphs, reconst


# import time
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter


# def matrix_poly(matrix, d, p):
#     # return (I + (1/d) * matrix)^p
#     x = torch.eye(d).to(matrix.device) + torch.div(matrix, d)
#     return torch.matrix_power(x, p)


# def constraint2(x, max_loop=30):
#     # normalize by num_nodes
#     return matrix_poly(x * x, x.shape[-1], min(max_loop, x.shape[-1])).diagonal(dim1=-2, dim2=-1).sum(dim=-1) / x.shape[-1] - 1


# class PostProcessingBlock(nn.Module):
#     def __init__(self, step_pri=0.01, step_dual=0.01, reg_sp=2e-3, num_iters=1000):
#         super(PostProcessingBlock, self).__init__()
#         self.reg_sp = reg_sp
#         self.num_iters = num_iters
#         self.thresholder = nn.Threshold(0.5, 0)
#         self.relu = nn.ReLU()

#         # for 20 nodes
#         if step_pri == 'auto':
#             self.step_pri = Parameter(torch.FloatTensor([0.01]))
#         else:  # float
#             self.step_pri = step_pri

#         if step_dual == 'auto':
#             self.step_dual = Parameter(torch.FloatTensor([0.01]))
#         else:  # float
#             self.step_dual = step_dual

#     def forward(self, adj):
#         # adj: (B, N, N)
#         # x: (B, N, N)
#         alpha = torch.zeros(adj.shape[0]).to(adj.device)
#         x, scores = adj.repeat(2, 1, 1, 1)
#         scores = self.thresholder(scores)

#         # h_total_time = 0
#         # alpha_total_time = 0
#         # update_x_total_time = 0
#         # grad_total_time = 0

#         for _ in range(self.num_iters):
#             # begin = time.time()
#             # primal update
#             grad = - scores + 2 * alpha.reshape(scores.shape[0], 1, 1) * x * \
#                    matrix_poly(x * x, x.shape[-1], x.shape[-1]-1).transpose(-1, -2) / x.shape[-1]
#             # grad_time = time.time()
#             # grad_total_time += grad_time - begin
#             # print(x.max())
#             # print(x[0, 0, :])
#             # assert torch.isnan(alpha).sum().item() == 0
#             # assert torch.isnan(matrix_poly(x * x, x.shape[-1])).sum().item() == 0
#             # assert torch.isnan(grad).sum().item() == 0
#             til_x = x - self.step_pri * grad
#             new_x = self.relu(til_x.abs() - self.reg_sp * self.step_pri)
#             x = 1 - F.relu(1 - new_x)  # min(A, 1)
#             # update_x_time = time.time()
#             # update_x_total_time += update_x_time - grad_time
#             # assert torch.isnan(x).sum().item() == 0

#             # dual update
#             h_newx = constraint2(x)
#             # h_time = time.time()
#             # h_total_time += h_time - update_x_time
#             # assert torch.isnan(h_newx).sum().item() == 0
#             alpha = alpha + self.step_dual * h_newx
#             # alpha_time = time.time()
#             # alpha_total_time += alpha_time - h_time

#         # print(f"M2 - grad: {grad_total_time: .2f}s")
#         # print(f"M2 - update_x: {update_x_total_time: .2f}s")
#         # print(f"M2 - h: {h_total_time: .2f}s")
#         # print(f"M2 - alpha: {alpha_total_time: .2f}s")
#         x = self.thresholder(x)
#         return x


# class PostProcessingBlocV2(nn.Module):
#     def __init__(self, step_pri=0.01, step_dual=0.01, reg_sp=2e-3, num_iters=1000, threshold=0.5):
#         super(PostProcessingBlocV2, self).__init__()
#         self.reg_sp = reg_sp
#         self.num_iters = num_iters
#         self.thresholder = nn.Threshold(threshold, 0)
#         self.relu = nn.ReLU()

#         # for 20 nodes
#         if step_pri == 'auto':
#             self.step_pri = Parameter(torch.FloatTensor([0.01]))
#         else:  # float
#             self.step_pri = step_pri

#         if step_dual == 'auto':
#             self.step_dual = Parameter(torch.FloatTensor([0.01]))
#         else:  # float
#             self.step_dual = step_dual

#     def forward(self, adj, mask=False):
#         # adj: (B, N, N)
#         # x: (B, N, N)
#         if mask:
#             diagonal_mask = (1. - torch.eye(adj.shape[-1])).to(adj.device)  # (N, N)
#             adj = diagonal_mask * adj

#         alpha = torch.zeros(adj.shape[0]).to(adj.device)
#         x, scores = adj.repeat(2, 1, 1, 1)
#         scores = self.thresholder(scores)

#         # h_total_time = 0
#         # alpha_total_time = 0
#         # update_x_total_time = 0
#         # grad_total_time = 0

#         identity_x = torch.eye(x.shape[-1]).to(x.device)
#         B = identity_x + torch.div(x * x, x.shape[-1])  # I + (1/N)*(x*x)
#         x_poly = torch.matrix_power(B, x.shape[-1] - 1)  # [I + (1/N)*(x*x)]^(N-1)
#         for _ in range(self.num_iters):
#             # begin = time.time()
#             # primal update
#             grad = - scores + 2 * alpha.reshape(scores.shape[0], 1, 1) * x * \
#                    x_poly.transpose(-1, -2) / x.shape[-1]
#             # grad_time = time.time()
#             # grad_total_time += grad_time - begin

#             til_x = x - self.step_pri * grad
#             new_x = self.relu(til_x.abs() - self.reg_sp * self.step_pri)
#             x = 1 - F.relu(1 - new_x)  # min(A, 1)
#             # update_x_time = time.time()
#             # update_x_total_time += update_x_time - grad_time

#             # dual update
#             B = identity_x + torch.div(x * x, x.shape[-1])
#             x_poly = torch.matrix_power(B, x.shape[-1] - 1)  # [I + (1/N)*(x*x)]^(N-1)
#             h_newx = torch.bmm(x_poly, B).diagonal(dim1=-2, dim2=-1).sum(dim=-1) / x.shape[-1] - 1
#             # h_time = time.time()
#             # h_total_time += h_time - update_x_time

#             alpha = alpha + self.step_dual * h_newx
#             # alpha_time = time.time()
#             # alpha_total_time += alpha_time - h_time

#         # print(f"PP V2")
#         # print(f"M2 - grad: {grad_total_time: .2f}s")
#         # print(f"M2 - update_x: {update_x_total_time: .2f}s")
#         # print(f"M2 - h: {h_total_time: .2f}s")
#         # print(f"M2 - alpha: {alpha_total_time: .2f}s")
#         x = self.thresholder(x)
#         return x


# class SPPostProcessingBlock(nn.Module):
#     def __init__(self, step_size=0.01, reg_sp=2e-3, num_iters=1000):
#         super(SPPostProcessingBlock, self).__init__()
#         self.num_iters = num_iters
#         self.thresholder = nn.Threshold(0.5, 0)
#         self.relu = nn.ReLU()

#         # for 20 nodes
#         if step_size == 'auto':
#             self.step_size = Parameter(torch.FloatTensor([0.01]))
#         else:  # float
#             self.step_size = step_size

#         if reg_sp == 'auto':
#             self.reg_sp = Parameter(torch.FloatTensor([0.01]))
#         else:  # float
#             self.reg_sp = reg_sp

#     def forward(self, adj):
#         # adj: (B, N, N)
#         # x: (B, N, N)
#         x, scores = adj.repeat(2, 1, 1, 1)
#         scores = self.thresholder(scores)

#         for _ in range(self.num_iters):
#             # primal update
#             grad = - scores
#             til_x = x - self.step_size * grad
#             new_x = F.relu(til_x.abs() - self.reg_sp * self.step_size)  # proximal update
#             x = 1 - F.relu(1 - new_x)  # min(x, 1)

#         x = self.thresholder(x)
#         return x
