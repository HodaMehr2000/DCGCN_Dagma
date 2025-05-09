import math
import torch
import numpy as np
import torch.nn as nn
from nns.case.mat_gru import MatGRU
from nns.case.dag_gen_block import DGBlockV1, DAGGumbelSigmoid
from nns.case.sem_block import MaskGCNSEM
from nns.case.feat_blocks import feat_embed_block
from utils.graph_utils import spatial_norm_tensor

# ──  DAGMA library
try:
    import dagma as DAGMA_lib
    _has_lib = True
except ImportError:
    _has_lib = False

#  (Hand‐written DAGMA) 

class HandwrittenDagmaProjector:
    
    def __init__(self,
                 alpha=1.0, mu0=1.0, mu_factor=0.5,
                 outer_steps=3, inner_steps=20, lr=1e-2):
        self.alpha       = alpha
        self.mu          = mu0
        self.mu_factor   = mu_factor
        self.outer_steps = outer_steps
        self.inner_steps = inner_steps
        self.lr          = lr

    def _grad_h(self, A):
        d = A.shape[-1]
        M = self.alpha * np.eye(d) + np.abs(A)
        invM = np.linalg.inv(M)
        return - invM * np.sign(A)

    def project(self, W, grad_Q_fn):
        Wp = W.copy()
        for _ in range(self.outer_steps):
            for __ in range(self.inner_steps):
                gQ = grad_Q_fn(Wp)              # ∇Q
                gH = self._grad_h(Wp)           # ∇H
                Wp -= self.lr * (self.mu * gQ + gH)
                Wp = np.clip(Wp, 0, None)
            self.mu *= self.mu_factor
        return Wp

class DagGenGRUg4s2v1(nn.Module):
    """
    Causal DAG generator with optionally library‐DAGMA or handwritten projector.
    """
    def __init__(self,
                 num_nodes, in_feats_dim, out_feats_dim,
                 hidden_dim, num_layers=2, num_heads=4,
                 feats_layers=3, dist_adj=None, agg_feats='ori',
                 node_norm=False, use_norm=False,
                 #  دو فلگ جدید 
                 use_dagma=True, use_lib_dagma=True,
                 dagma_mu=1.0, dagma_alpha=1.0, dagma_steps=10,
                 reg_sp_intra=2e-3, num_intra_pp_iters=1000, **kwargs):
        super().__init__()
        GRU_FOLD = 4

        self.num_nodes    = num_nodes
        self.hidden_dim   = hidden_dim
        self.num_heads    = num_heads
        self.use_norm     = use_norm
        self.use_dagma    = use_dagma
        self.use_lib_dagma = use_lib_dagma and _has_lib

        # 1) Feature embed
        self.embed = feat_embed_block(
            num_nodes, in_feats_dim, hidden_dim,
            num_layers=feats_layers, dist_adj=dist_adj,
            agg_feats=agg_feats, node_norm=node_norm
        )

        # 2) Q/K projections
        self.W_inter_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_inter_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_intra_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_intra_k = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 3) MatGRUs
        self.graph_gru_inter = MatGRU(num_nodes**2, num_heads, num_heads*GRU_FOLD, num_layers)
        self.graph_gru_intra = MatGRU(num_nodes**2, num_heads, num_heads*GRU_FOLD, num_layers)

        # 4) DGBlock logits
        self.graph_gen_inter = DGBlockV1(num_heads*GRU_FOLD, hidden_dim, 1)
        self.graph_gen_intra = DGBlockV1(num_heads*GRU_FOLD, hidden_dim, 1)

        # 5) Gumbel‐Sigmoid
        self.gumbel_sigmoid = DAGGumbelSigmoid()

        # 6) DAGMA projector
        if use_dagma:
            if self.use_lib_dagma:
                self.dagma_solver = DAGMA_lib(mu=dagma_mu,
                                              alpha=dagma_alpha,
                                              steps=dagma_steps)
            else:
                self.dagma_solver = HandwrittenDagmaProjector(
                    alpha=dagma_alpha, mu0=dagma_mu,
                    outer_steps=dagma_steps, inner_steps=50,
                    lr=1e-2
                )

        # 7) SEM encoder
        self.sem_encoder = MaskGCNSEM(
            num_nodes, hidden_dim, out_feats_dim, hidden_dim
        )

    def split_heads(self, x):
        T, B, N, H = x.shape
        x = x.view(T, B, N, self.num_heads, H//self.num_heads)
        return x.permute(0, 1, 3, 2, 4)

    @staticmethod
    def flat_pixels(x):
        T, B, C, N, _ = x.shape
        return x.reshape(T, B, C, N*N).permute(0, 1, 3, 2)

    @staticmethod
    def unflat_pixels(x):
        T, B, N2, C = x.shape
        N = int(math.sqrt(N2))
        x = x.view(T, B, N, N, C)
        return x.permute(0, 1, 4, 2, 3)

    def forward(self, x_with_pre, gen_graph_only=False):
        # 1) embed
        x_with_pre = self.embed(x_with_pre)
        pre_x, x = x_with_pre[:-1], x_with_pre[1:]

        # 2) Q/K + split
        iq = self.split_heads(self.W_inter_q(x))
        ik = self.split_heads(self.W_inter_k(pre_x))
        uq = self.split_heads(self.W_intra_q(x))
        uk = self.split_heads(self.W_intra_k(pre_x))

        # 3) attention scores
        pi = torch.matmul(iq, ik.transpose(-1,-2))
        cu = torch.matmul(uq, uk.transpose(-1,-2))

        # 4) flatten
        pf = self.flat_pixels(pi)
        cf = self.flat_pixels(cu)

        # 5) GRU
        hi, _ = self.graph_gru_inter(pf)
        hu, _ = self.graph_gru_intra(cf)

        # 6) logits + unflat
        gi = self.unflat_pixels(self.graph_gen_inter(hi)).squeeze(2)
        gu = self.unflat_pixels(self.graph_gen_intra(hu)).squeeze(2)

        # 7) Gumbel
        gi = self.gumbel_sigmoid(gi)
        gu = self.gumbel_sigmoid(gu, mask=True)

        # 8) DAGMA post‐process
        if self.use_dagma:
            T, B, N, _ = gu.shape
            flat_np = gu.detach().cpu().numpy().reshape(-1, N, N)
            if self.use_lib_dagma:
                proj_np = self.dagma_solver.project(flat_np)
            else:
                #  Q - handwritten
                def grad_Q_fn(W_np):
                    # stub:  ∇Q - loss 
                    return np.zeros_like(W_np)
                proj_np = self.dagma_solver.project(flat_np, grad_Q_fn)
            gu = torch.from_numpy(proj_np).view(T, B, N, N).to(x_with_pre.device)

        # 9) stack
        graphs = torch.stack([gi, gu], dim=2)

        if gen_graph_only:
            return graphs, None

        # 10) SEM loss
        if self.use_norm:
            graphs = spatial_norm_tensor(graphs, add_self_loops=False)
        reconst = self.sem_encoder(x_with_pre, graphs)

        return graphs, reconst
