import torch
from torch.nn import Sequential, Linear, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn as nn

class GCN_with_Sep(nn.Module):
    def __init__(self, features, hidden, classes, dropout=5e-5, K=4):
        super(GCN_with_Sep, self).__init__()
        self.embed = nn.Sequential(nn.Linear(features, hidden),
                                   nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 ** (K + 1) - 1), classes)
        self.K = K

    def forward(self, data):
        x, adj = data.x, data.adj
        hidden_reps = []
        adj_2 = torch.pow(adj, 2)
        x = self.embed(x)
        hidden_reps.append(x)
        for _ in range(self.K):
            r1 = adj.matmul(x)
            r2 = adj_2.matmul(x)
            x = torch.cat([r1, r2], dim=-1)
            hidden_reps.append(x)
        hf = self.dropout(torch.cat(hidden_reps, dim=-1))
        return F.log_softmax(self.fc(hf), dim=1)


class DualGNN(torch.nn.Module): # 传入edge_index
    # the model decomposing the normalized adj matrix
    def __init__(self, input_dim, hidden_dim, output_dim, MLPA=0.001, A_dim=1, K=2):
        super().__init__()

        self.net1 = GCN_with_Sep(input_dim, hidden_dim, output_dim, K=K)
        self.net2 = GCN_with_Sep(input_dim, hidden_dim, output_dim, K=K)

        self.offset_mlp = Sequential(
            Linear(input_dim, hidden_dim*2),
            ReLU(),
            Linear(hidden_dim*2, output_dim)
        )

        self.mlp_A = Sequential(
            Linear(A_dim, hidden_dim*2),
            ReLU(),
            Linear(hidden_dim*2, output_dim)
        )

        self.MLPA = MLPA

        self._cached_edge_index = None
        self.add_self_loops = True

    def forward(self, data, output_emb=False):
        x, edge_index, adj = data.x, data.edge_index, data.adj

        cache = self._cached_edge_index
        if cache is None:
            edge_weight = None
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(0), self.add_self_loops)
            self._cached_edge_index = (edge_index, edge_weight)
        else:
            edge_index, edge_weight = cache[0], cache[1]


        x1 = self.net1(data)
        x2 = self.net2(data)

        x_offset = self.offset_mlp(x)
        x = x1 + x2 + 0.001 * x_offset

        if self.MLPA != 0:
            x += self.MLPA * self.mlp_A(data.A)

        if output_emb:
            return x
        else:
            return F.log_softmax(x, dim=1)
