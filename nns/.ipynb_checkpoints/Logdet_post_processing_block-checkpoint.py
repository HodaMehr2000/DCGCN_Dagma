import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


def constraint_logdet(x, s=1.1):
    """
    Acyclicity constraint based on log-determinant:
    logdet(sI - W◦W) - d * log(s) ≈ 0  ⇨ DAG
    """
    B, N, _ = x.shape
    W2 = x * x  # (B, N, N)
    I = torch.eye(N).to(x.device).unsqueeze(0).expand(B, N, N)
    S = s * I - W2
    logdet = torch.linalg.slogdet(S)[1]  # log|det|
    return logdet - N * math.log(s)


class LogDetPostProcessingBlock(nn.Module):
    def __init__(self, step_pri=0.01, step_dual=0.01, reg_sp=2e-3, num_iters=1000, threshold=0.5, logdet_s=1.1):
        super(LogDetPostProcessingBlock, self).__init__()
        self.reg_sp = reg_sp
        self.num_iters = num_iters
        self.logdet_s = logdet_s
        self.thresholder = nn.Threshold(threshold, 0)
        self.relu = nn.ReLU()

        # step sizes
        self.step_pri = Parameter(torch.FloatTensor([step_pri])) if step_pri == 'auto' else step_pri
        self.step_dual = Parameter(torch.FloatTensor([step_dual])) if step_dual == 'auto' else step_dual

    def forward(self, adj):
        # adj: (B, N, N)
        alpha = torch.zeros(adj.shape[0]).to(adj.device)
        x, scores = adj.repeat(2, 1, 1, 1)  # duplicate for optimization
        scores = self.thresholder(scores)

        for _ in range(self.num_iters):
            # primal update
            grad = -scores  # simplified (no ∇Q)
            til_x = x - self.step_pri * grad
            new_x = self.relu(til_x.abs() - self.reg_sp * self.step_pri)
            x = 1 - F.relu(1 - new_x)  # clip to [0, 1]

            # dual update using logdet constraint
            h_newx = constraint_logdet(x, s=self.logdet_s)
            alpha = alpha + self.step_dual * h_newx

        return self.thresholder(x)

class SPPostProcessingBlock(nn.Module):
    def __init__(self, step_size=0.01, reg_sp=2e-3, num_iters=1000):
        super(SPPostProcessingBlock, self).__init__()
        self.num_iters = num_iters
        self.thresholder = nn.Threshold(0.5, 0)
        self.relu = nn.ReLU()

        # for 20 nodes
        if step_size == 'auto':
            self.step_size = Parameter(torch.FloatTensor([0.01]))
        else:  # float
            self.step_size = step_size

        if reg_sp == 'auto':
            self.reg_sp = Parameter(torch.FloatTensor([0.01]))
        else:  # float
            self.reg_sp = reg_sp

    def forward(self, adj):
        # adj: (B, N, N)
        # x: (B, N, N)
        x, scores = adj.repeat(2, 1, 1, 1)
        scores = self.thresholder(scores)

        for _ in range(self.num_iters):
            # primal update
            grad = - scores
            til_x = x - self.step_size * grad
            new_x = F.relu(til_x.abs() - self.reg_sp * self.step_size)  # proximal update
            x = 1 - F.relu(1 - new_x)  # min(x, 1)

        x = self.thresholder(x)
        return x