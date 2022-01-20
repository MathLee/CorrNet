

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



# class FLoss(torch.nn.Module):
#     def __init__(self, beta=0.3, log_like=False):
#         super(FLoss, self).__init__()
#         self.beta = beta
#         self.log_like = log_like
#
#     def forward(self, prediction, target):
#         EPS = 1e-10
#         floss = 0.0
#         N = prediction.shape[0]
#         for i in range(0, N):
#             TP = (prediction[i, :, :, :] * target[i, :, :, :])
#             H = self.beta * target[i, :, :, :] + prediction[i, :, :, :]
#             fm = (1 + self.beta) * TP / (H + EPS)
#             if self.log_like:
#                 floss = floss - torch.log(fm)
#             else:
#                 floss = floss + (1 - fm)
#
#         return floss / N

class FLoss(torch.nn.Module):
    def __init__(self, beta=0.3, log_like=False):
        super(FLoss, self).__init__()
        self.beta = beta
        self.log_like = log_like

    def forward(self, prediction, target):
        EPS = 1e-10
        N = prediction.size(0)
        TP = (prediction * target).view(N, -1).sum(dim=1)
        H = self.beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
        fmeasure = (1 + self.beta) * TP / (H + EPS)
        if self.log_like:
            floss = -torch.log(fmeasure)
        else:
            floss  = (1 - fmeasure)
        return floss.mean()