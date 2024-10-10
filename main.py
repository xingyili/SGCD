#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/4/23 19:32
# @Author  : Jimmy
# @FileName: main.py
# @Usage: say something
import sys, argparse
from models import *
from utils import *
import json
import os
from datetime import datetime
import torch_geometric

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='STRINGdb',
                    choices=['CPDB', 'STRINGdb', 'MULTINET', 'PCNet', 'IRefIndex', 'IRefIndex_2015'],
                    help="The dataset to be used.")
parser.add_argument('--seed', type=int, default=1234,
                    help="random seed")
parser.add_argument('--hidden_unit', type=int, default=64,
                    help="Hidden dim")
parser.add_argument('--wd', type=float, default=5e-5,
                    help="Weight decay.")
parser.add_argument('--lr', type=float, default=0.014866883823555476,
                    help="Learning rate.")
parser.add_argument('--MLPA', type=float, default=0.020384170281443927,
                    help="The ratio of signals from the MLP(A) part.")
parser.add_argument('--device', type=str, default='1',
                    choices=['cpu', '0', '1', '2', '3'],
                    help="The GPU device to be used.")
parser.add_argument('--run', type=int, default=1,
                    help="The # of test runs.")
parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--logs', default=False, help='Save the results to a log file')
args = parser.parse_args()

set_seed(args.seed)
if args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:' + args.device
torch.autograd.set_detect_anomaly(True)


""" Experimental settings """
hidden_unit = args.hidden_unit
data = get_ppi(dataset=args.dataset)
num_nodes = data.y.shape[0]
all_idx = np.array(range(num_nodes))
num_features = data.x.shape[1]
num_classes = data.y.max().item() + 1
data.adj = normalize(data.adj) #归一化
data.A = torch_geometric.utils.to_dense_adj(data.edge_index)[0]
data = data.to(device)

print("Device: {}".format(args.device))
print("Dataset: {}".format(args.dataset))
print("wd: {}".format(args.wd))
print("lr: {}".format(args.lr))
print("hidden_unit: {}".format(hidden_unit))
print("MLP_A: {}".format(args.MLPA))
print("max_epochs: {}".format(args.max_epochs))

kfold = 5

all_mask = (data.train_mask | data.val_mask | data.test_mask).cpu().numpy()
y = data.y.squeeze()[all_mask.squeeze()].cpu().numpy()
idx_list = np.arange(all_mask.shape[0])[all_mask.squeeze()]

skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
# 用np.random.permutation打乱idx_list
# kf = KFold(n_splits=kfold, shuffle=True, random_state=0)
train_mask_set = []
test_masks_set = []

best_test_auprc_list = []
best_test_auc_list = []

# 根据五折交叉验证的划分结果生成train_mask和test_mask
for train_index, test_index in skf.split(idx_list, y): #划分训练集和测试集
    train_mask = np.full_like(all_mask, False)  # 初始化与all_mask相同大小的train_mask
    test_mask = np.full_like(all_mask, False)  # 初始化与all_mask相同大小的test_mask

    # 将训练集索引位置设置为True
    train_mask[idx_list[train_index]] = True
    # 将测试集索引位置设置为True
    test_mask[idx_list[test_index]] = True

    train_mask_set.append(train_mask)
    test_masks_set.append(test_mask)



hidden_unit = args.hidden_unit
num_features = data.x.shape[1]
num_classes = data.y.max().item() + 1

test_auprc_list = []
test_auc_list = []

# 下面根据train_mask和test_mask进行训练和测试
for train_mask, test_mask in zip(train_mask_set, test_masks_set):
    model = DualGNN(num_features, hidden_unit, num_classes, MLPA=args.MLPA,
                         A_dim=data.A.shape[0], K=args.K).to(data.edge_index.device)

    #model = H2GCN_ver2(num_features, hidden_unit, num_classes, K=2).to(data.edge_index.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_idx = train_mask.nonzero()[0]
    test_idx = test_mask.nonzero()[0]


    for epoch in tqdm(range(1, args.max_epochs + 1)):
        # 训练模型
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[train_idx], data.y[train_idx].squeeze().long())
        loss.backward()
        optimizer.step()
        # 测试模型

    model.eval()
    out = model(data)
    y_score = F.softmax(out, dim=1)[:, 1]
    auprcs = []
    aucs = []
    for idx in (train_idx, test_idx):
        auprc = compute_auprc(data.y[idx].cpu().detach().numpy(), y_score[idx].cpu().detach().numpy())
        auc = compute_auc(data.y[idx].cpu().detach().numpy(), y_score[idx].cpu().detach().numpy())
        auprcs.append(auprc)
        aucs.append(auc)
    train_auprc, test_auprc = auprcs
    train_auc, test_auc = aucs


    print(f"test_auc: {test_auc}, test_auprc: {test_auprc}")
    test_auprc_list.append(test_auprc)
    test_auc_list.append(test_auc)

test_auprc_list = np.array(test_auprc_list)
test_auc_list = np.array(test_auc_list)

mean_test_auc = test_auc_list.mean()
std_test_auc = test_auc_list.std()
mean_test_auprc = test_auprc_list.mean()
std_test_auprc = test_auprc_list.std()

print(f"AUC mean: {mean_test_auc}, AUPRC mean: {mean_test_auprc}")
print(f"AUC std: {std_test_auc}, AUPRC std: {std_test_auprc}")

if args.logs == True:
    # Create a dictionary containing the args and results
    write_log = {
        'args': {
            'dataset': args.dataset,
            'seed': args.seed,
            'hidden_unit': args.hidden_unit,
            'wd': args.wd,
            'lr': args.lr,
            'MLPA': args.MLPA,
            'device': args.device,
            'run': args.run,
            'max_epochs': args.max_epochs
        },
        'test_auc_mean': mean_test_auc,
        'test_auc_std': std_test_auc,
        'test_auprc_mean': mean_test_auprc,
        'test_auprc_std': std_test_auprc
    }

    # Define the file path and name based on args.dataset and current time
    file_name = f"{args.dataset}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
    file_path = os.path.join('metric_result', file_name)

    # Check if the metric_result directory exists, and create it if not
    if not os.path.exists('metric_result'):
        os.makedirs('metric_result')

    # Save the data dictionary to a YAML file
    with open(file_path, 'w') as file:
        json.dump(write_log, file, indent=4)
