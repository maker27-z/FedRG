import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphConstructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphConstructor, self).__init__()

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.nnodes = nnodes

        if static_feat is not None:
            tensor_shape = static_feat[0].shape
            xd = tensor_shape[0]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)



    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)  #��ȡ�ڵ����������
            nodevec2 = self.emb2(idx)
        else:
            # nodevec1 = self.static_feat[idx, :]
            nodevec1 = torch.stack([self.static_feat[i] for i in idx])
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha*a))

        return adj

    def eval(self, idx, full=False):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            # nodevec1 = self.static_feat[idx, :]
            nodevec1 = torch.stack([self.static_feat[i] for i in idx])
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))-torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha*a))

        #save the big k numbers
        # if not full:
        #     mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        #     mask.fill_(float('0'))
        #     s1, t1 = adj.topk(self.k, 1)
        #     mask.scatter_(1, t1, s1.fill_(1))
        #     adj = adj*mask

        return adj
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x