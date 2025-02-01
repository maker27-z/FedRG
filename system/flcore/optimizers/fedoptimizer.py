import torch
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(CustomLoss, self).__init__()
        self.temperature = temperature

    def forward(self, vl, Pj, Nj):
        """
        vl: tensor of shape (N, d), where N is the number of samples and d is the feature dimension
        Pj: tensor of shape (M, d), positive samples
        Nj: tensor of shape (K, d), negative samples

        Returns:
        Custom loss value
        """
        # for vplus in Pj:
        # if len(Nj) > 0:
        #     for nj in Nj:
        #         neg += F.mse_loss(vl, nj)
        # denominator = neg
        # if denominator == 0.0:
        #     loss = numerator
        # else:
        #     loss = numerator/ denominator

        # loss /= len(Pj)

        vl = F.normalize(vl, p=2, dim=-1).squeeze()
        Pj = F.normalize(Pj, p=2, dim=-1).squeeze()
        Nj = [F.normalize(nj, p=2, dim=-1).squeeze() for nj in Nj]

        numerator = torch.exp(torch.dot(vl, Pj) / self.temperature)

        neg = 0.0
        for nj in Nj:
            neg += torch.exp(torch.dot(vl, nj) / self.temperature)

        # Adding a small value to the denominator to avoid division by zero
        loss = -torch.log(numerator / (numerator + neg + 1e-8))

        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group['lr'])


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, server_cs, client_cs):
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cs, client_cs):
                p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_model, device):
        group = None
        weight_update = local_model.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                localweight = localweight.to(device)
                # approximate local model
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)

        return group['params']


# class APFLOptimizer(Optimizer):
#     def __init__(self, params, lr):
#         defaults = dict(lr=lr)
#         super(APFLOptimizer, self).__init__(params, defaults)
#
#     def step(self, beta=1, n_k=1):
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 d_p = beta * n_k * p.grad.data
#                 p.data.add_(-group['lr'], d_p)


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])
