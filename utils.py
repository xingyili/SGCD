import math
import random
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
import numpy as np
import os
import h5py


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix, degree
import scipy.sparse as sp

from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm.auto import tqdm
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_ppi(dataset: str = 'CPDB', essential_gene=False, health_gene=False, PATH='PPI_data/'):
    print(dataset)
    ppi_PATH = os.path.join(PATH, f'{dataset}_multiomics.h5')
    ppi_essential_PATH = os.path.join(PATH, f'{dataset}_essential_test01_multiomics.h5')
    if health_gene:
        return get_health(dataset, PATH)  # get_health没实现
    elif essential_gene:
        f = h5py.File(ppi_essential_PATH, 'r')
    else:
        f = h5py.File(ppi_PATH, 'r')
    src, dst = np.nonzero(f['network'][:])
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    x = torch.from_numpy(f['features'][:]).float()
    y = torch.from_numpy(
        np.logical_or(np.logical_or(f['y_test'][:], f['y_val'][:]), f['y_train'][:])).int()
    train_mask = torch.from_numpy(f['mask_train'][:])
    val_mask = torch.from_numpy(f['mask_val'][:])
    test_mask = torch.from_numpy(f['mask_test'][:])
    name = f['gene_names'][:]
    name = np.array([x.decode("utf-8") for x in name[:, 1]], dtype=str)

    g = Data(x=x, edge_index=edge_index, y=y)
    g.train_mask = train_mask
    g.val_mask = val_mask
    g.test_mask = test_mask
    # 通过train_mask test_mask val_mask得到idx_train, idx_test, idx_val
    g.idx_train = torch.nonzero(g.train_mask).squeeze()
    g.idx_test = torch.nonzero(g.test_mask).squeeze()
    g.idx_val = torch.nonzero(g.val_mask).squeeze()
    g.name = name
    edge_weight = torch.ones(edge_index.shape[1])
    g.adj = torch.sparse_coo_tensor(edge_index, edge_weight)
    g.num_classes = f['y_test'][:].shape[1]

    return g


def remove_edges(edge_index, edges_to_remove):
    edges_to_remove = torch.cat(
        [edges_to_remove, edges_to_remove.flip(0)], dim=1)
    edges_to_remove = edges_to_remove.to(edge_index)

    # it's not intuitive to remove edges from a graph represented as `edge_index`
    edge_weight_remove = torch.zeros(edges_to_remove.size(1)) - 1e5
    edge_weight = torch.cat(
        [torch.ones(edge_index.size(1)), edge_weight_remove], dim=0)
    edge_index = torch.cat([edge_index, edges_to_remove], dim=1).cpu().numpy()
    adj_matrix = sp.csr_matrix(
        (edge_weight.cpu().numpy(), (edge_index[0], edge_index[1])))
    adj_matrix.data[adj_matrix.data < 0] = 0.
    adj_matrix.eliminate_zeros()
    edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
    return edge_index


def edge_index_to_sparse_tensor_adj(edge_index):
    sparse_adj_adj = to_scipy_sparse_matrix(edge_index)
    values = sparse_adj_adj.data
    indices = np.vstack((sparse_adj_adj.row, sparse_adj_adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = sparse_adj_adj.shape
    sparse_adj_adj_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return sparse_adj_adj_tensor


import torch
from torch_sparse import SparseTensor
from torch_scatter import scatter

from torch_scatter import scatter_add


def gcn_norm_edge_weighted(edge_index, edge_weight, num_nodes, device):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),),
                                 device=edge_index.device)
    edge_weight = edge_weight.view(-1)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    d1_adj = torch.diag(deg).to_sparse().to(device)
    d1_adj = torch.pow(d1_adj, -0.5)
    a1 = edge_index_to_sparse_tensor_adj(edge_index).to(device)

    return torch.sparse.mm(torch.sparse.mm(d1_adj, a1), d1_adj)


def gcn_norm(edge_index, num_nodes, device):
    a1 = edge_index_to_sparse_tensor_adj(edge_index).to(device)
    d1_adj = torch.diag(degree(edge_index[0], num_nodes=num_nodes)).to_sparse().to(device)
    d1_adj = torch.pow(d1_adj, -0.5)

    return torch.sparse.mm(torch.sparse.mm(d1_adj, a1), d1_adj)


def prepare_norm_adjs(edge_index, num_nodes, layer_num=2, device=None):
    temp_loop_edge_index, _ = add_self_loops(edge_index)
    sparse_adj_tensor = edge_index_to_sparse_tensor_adj(temp_loop_edge_index)

    k_hop_adjs = []
    k_hop_edge_index = []
    k_hop_adjs.append(sparse_adj_tensor)

    for i in range(layer_num - 1):
        temp_adj_adj = torch.sparse.mm(k_hop_adjs[i], sparse_adj_tensor)

        k_hop_adjs.append(temp_adj_adj)
        k_hop_edge_index.append(temp_adj_adj._indices())

    for i in range(layer_num - 1):
        k_hop_edge_index[i], _ = remove_self_loops(k_hop_edge_index[i])
        if i == 0:
            k_hop_edge_index[i] = remove_edges(k_hop_edge_index[i], edge_index)
        else:
            k_hop_edge_index[i] = remove_edges(k_hop_edge_index[i], k_hop_edge_index[i - 1])

    norm_adjs = []

    norm_adjs.append(gcn_norm(edge_index, num_nodes, device))
    norm_adjs.append(gcn_norm(k_hop_edge_index[0], num_nodes, device))

    return norm_adjs


def prepare_norm_adjs_edge_weighted(edge_index, edge_weight, num_nodes, layer_num=2, device=None):
    temp_loop_edge_index, _ = add_self_loops(edge_index)
    sparse_adj_tensor = edge_index_to_sparse_tensor_adj(temp_loop_edge_index)

    k_hop_adjs = []
    k_hop_edge_index = []
    k_hop_adjs.append(sparse_adj_tensor)

    for i in range(layer_num - 1):
        temp_adj_adj = torch.sparse.mm(k_hop_adjs[i], sparse_adj_tensor)

        k_hop_adjs.append(temp_adj_adj)
        k_hop_edge_index.append(temp_adj_adj._indices())

    for i in range(layer_num - 1):
        k_hop_edge_index[i], _ = remove_self_loops(k_hop_edge_index[i])
        if i == 0:
            k_hop_edge_index[i] = remove_edges(k_hop_edge_index[i], edge_index)
        else:
            k_hop_edge_index[i] = remove_edges(k_hop_edge_index[i], k_hop_edge_index[i - 1])

    norm_adjs = []

    norm_adjs.append(gcn_norm_edge_weighted(edge_index, edge_weight, num_nodes, device))
    norm_adjs.append(gcn_norm_edge_weighted(k_hop_edge_index[0], edge_weight, num_nodes, device))

    return norm_adjs


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


from sklearn.metrics import roc_curve, auc, precision_recall_curve


def compute_auc(y_true, y_score):
    # 计算fpr, tpr
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # 计算auc
    auc_score = auc(fpr, tpr)
    return auc_score


def compute_auprc(y_true, y_score):
    # 计算precision, recall
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # 计算auprc
    auprc_score = auc(recall, precision)
    return auprc_score

from sklearn.metrics import f1_score, accuracy_score
def compute_f1_acc(y_true, y_pred):
    # 计算f1
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return f1, acc

def normalize(matrix):
    """对称归一化稀疏矩阵"""
    matrix = matrix.to_dense() + torch.eye(matrix.size(0), device=matrix.device)  # A+I
    row_sum = matrix.sum(1)
    d_inv_sqrt = row_sum.pow(-0.5).flatten()  # (D + I)^-0.5
    d_inv_sqrt[d_inv_sqrt == float('inf')] = 0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)  # 构建成对角矩阵
    normalized_matrix = d_mat_inv_sqrt.matmul(matrix).matmul(d_mat_inv_sqrt)

    # 将结果转换为稀疏张量
    indices = normalized_matrix.nonzero().t()
    values = normalized_matrix[indices[0], indices[1]]
    normalized_sparse_matrix = torch.sparse_coo_tensor(indices, values, size=normalized_matrix.size())

    return normalized_sparse_matrix