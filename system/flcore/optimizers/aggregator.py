import torch
from flcore.optimizers.GraphConstructor import GraphConstructor
from utils.data_utils import normalize_adj
import torch.nn.functional as F


def rep_aggregate(args, param_metrix, a_weights):
    subgraph_size = min(len(param_metrix), args.num_clients)

    #GNN
    matrix = torch.ones(subgraph_size, subgraph_size)
    row_sums = matrix.sum(dim=1, keepdim=True)
    a_weights = matrix / row_sums

    A = generate_adj(args, param_metrix, subgraph_size, a_weights)

    A = torch.tensor(normalize_adj(A.detach().numpy()))

    new_param_metrix = torch.tensor([sublist.tolist() for sublist in param_metrix])
    aggregated_param = torch.mm(A, new_param_metrix)
    aggregated_rep = [torch.tensor(param).to(args.device) for param in aggregated_param]

    return aggregated_rep


def generate_adj(args, param_metrix, subgraph_size, a_weights):
    graph_len = len(param_metrix)
    dist_metrix = F.normalize(a_weights).to(args.device)

    gc = GraphConstructor(graph_len, subgraph_size, args.node_dim,
                          args.device, args.adjalpha, param_metrix).to(args.device)
    idx = torch.arange(graph_len).to(args.device)
    optimizer = torch.optim.SGD(gc.parameters(), lr=args.gnn_learning_rate, weight_decay=args.gnn_weight_decay)

    for e in range(args.gc_epoch):
        optimizer.zero_grad()
        adj = gc(idx)
        adj = F.normalize(adj)

        loss = F.mse_loss(adj, dist_metrix)
        loss.backward()
        optimizer.step()

    adj = gc.eval(idx).to("cpu")

    return adj
