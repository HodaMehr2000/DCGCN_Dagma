import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter



def constraint2(x, max_loop=30):
    """
    DAGMA acyclicity test:
      h(W) = -logdet(s I - W^2) + N log(s)
    It returns a (B,) tensor that is zero iff each W in the batch is a DAG.
    """
    B, N, _ = x.shape
    # choose barrier parameter s>0 (must exceed spectral radius of x^2; 1.0 if data is scaled)
    s = 1.0
    # build batched identity
    I = torch.eye(N, device=x.device).unsqueeze(0).expand(B, N, N)
    # compute S = sI - W^2
    S = s * I - x * x
    # compute log-determinant per batch
    # use slogdet for numerical stability
    sign, logdet = torch.slogdet(S)
    # if any sign<=0, the matrix is singular or invalid—push h large
    h = -logdet + N * math.log(s)
    # (optionally) clamp h to be non-negative
    return h.clamp_min(0.0)


def dagma_acyclicity_and_grad(x, s=1.0):
    """
    Compute DAGMA acyclicity constraint and its gradient for batch of adjacency matrices x.
    Returns:
      h: tensor of shape (B,) with -logdet(sI - x^2) + N*log(s)
      grad: tensor of shape (B,N,N) with gradient w.r.t. x: 2 * inv(sI - x^2) * x
    """
    B, N, _ = x.shape
    # Identity batch
    I = torch.eye(N, device=x.device).unsqueeze(0).expand(B, N, N)
    S = s * I - x * x
    # acyclicity value per instance
    h = -torch.logdet(S) + N * math.log(s)
    # gradient per instance
    invS = torch.inverse(S)
    grad = 2.0 * invS * x
    return h, grad


class DAGMAPostProcessingBlock(nn.Module):
    """
    Post-processing block using DAGMA acyclicity constraint.
    Similar API to original PostProcessingBlock.
    """
    def __init__(self, step_pri=0.01, step_dual=0.01, reg_sp=2e-3,
                 num_iters=1000, s=1.0, threshold=0.5):
        super(DAGMAPostProcessingBlock, self).__init__()
        self.reg_sp = reg_sp
        self.num_iters = num_iters
        self.s = s
        self.thresholder = nn.Threshold(threshold, 0.)
        self.relu = nn.ReLU()

        # primary and dual step sizes
        self.step_pri = Parameter(torch.tensor(step_pri)) if step_pri == 'auto' else step_pri
        self.step_dual = Parameter(torch.tensor(step_dual)) if step_dual == 'auto' else step_dual

    def forward(self, adj, mask=False):
        """
        adj: (B, N, N) raw scores
        mask: if True, zero out diagonal before processing
        returns: (B, N, N) thresholded adjacency
        """
        if mask:
            diag_mask = 1.0 - torch.eye(adj.shape[-1], device=adj.device)
            adj = adj * diag_mask

        B, N, _ = adj.shape
        # initialize dual variables
        alpha = torch.zeros(B, device=adj.device)
        # initialize x and score copies
        x, scores = adj.repeat(2, 1, 1, 1)
        scores = self.thresholder(scores)

        for _ in range(self.num_iters):
            # compute constraint and its gradient
            h, grad_h = dagma_acyclicity_and_grad(x, self.s)
            # primal gradient includes score and constraint gradient
            grad = -scores + alpha.view(B,1,1) * grad_h
            # gradient descent with sparsity prox
            til_x = x - self.step_pri * grad
            new_x = self.relu(til_x.abs() - self.reg_sp * self.step_pri)
            x = 1.0 - F.relu(1.0 - new_x)
            # dual ascent on constraint
            alpha = alpha + self.step_dual * h

        return self.thresholder(x)


class SPPostProcessingBlock(nn.Module):
    """
    Simplified post-processing block without acyclicity.
    """
    def __init__(self, step_size=0.01, reg_sp=2e-3, num_iters=1000, threshold=0.5):
        super(SPPostProcessingBlock, self).__init__()
        self.num_iters = num_iters
        self.thresholder = nn.Threshold(threshold, 0.)
        self.relu = nn.ReLU()
        self.step_size = Parameter(torch.tensor(step_size)) if step_size == 'auto' else step_size
        self.reg_sp   = Parameter(torch.tensor(reg_sp))   if reg_sp   == 'auto' else reg_sp

    def forward(self, adj):
        x, scores = adj.repeat(2,1,1,1)
        scores = self.thresholder(scores)
        for _ in range(self.num_iters):
            grad = -scores
            til_x = x - self.step_size * grad
            new_x = F.relu(til_x.abs() - self.reg_sp * self.step_size)
            x = 1.0 - F.relu(1.0 - new_x)
        return self.thresholder(x)



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
