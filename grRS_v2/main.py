import random
import numpy as np
import torch
import scipy.sparse as sp
import os
import argparse
from sklearn.preprocessing import MinMaxScaler
from utils import csr2torch, recall_at_k, ndcg_at_k, hit_at_k, normalize_sparse_adjacency_matrix, normalize_sparse_adjacency_matrix_, filter
from dataset import Dataset  

# from dataset_weeplaces import DatasetWeeplaces
# from dataset_scaling import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_directory = os.getcwd() 

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,    default="Mafengwo", # "CAMRa2011" or "Mafengwo"
    help="Either CAMRa2011, agree_data or weeplaces.",
)

parser.add_argument(
    "--verbose",
    type=int,
    default=1,
    help="Whether to print the results or not. 1 prints the results, 0 does not.",
)
parser.add_argument("--user_alpha", type=float, default=1, help="weight of P_2^T @ R_2")
parser.add_argument("--group_alpha", type=float, default=1, help="weight of P_2^T @ R_2")
parser.add_argument("--alpha", type=float, default=0.5, help="For normalization of R")
parser.add_argument("--power", type=float, default=1, help="For normalization of P")
parser.add_argument("--filter_pair", type=str, default="filter_1D_1D", help="pair filter of user and group")



args = parser.parse_args()  

if args.verbose:
    print(f"Device: {device}")

# load dataset
dataset = args.dataset
path = current_directory + f'/data/{dataset}/'
data = Dataset(path) 
R_tr_g, R_ts_g, R_tr_u, R_ts_u, C, g_neg, u_neg = data.getDataset()

# shape
train_n_groups = R_tr_g.shape[0]
train_group_n_items = R_tr_g.shape[1]
train_n_users = R_tr_u.shape[0]
train_user_n_items = R_tr_u.shape[1]

# if args.verbose:
#     print(f"number of tr_groups: {train_n_groups}")
#     print(f"number of tr_groups_items: {train_group_n_items}")
#     print(f"number of tr_users: {train_n_users}")
#     print(f"number of tr_users_items: {train_user_n_items}")

# R_tilde 구하기
train_group_mceg_norm = normalize_sparse_adjacency_matrix(R_tr_g.to_dense(), args.alpha) 
R_tr_u_star = R_tr_u.to_dense() * C.unsqueeze(1)  # group-user consistency calculation
train_user_mceg_norm = normalize_sparse_adjacency_matrix(R_tr_u_star, args.alpha) 

# R = R_tr.to_dense()
new_R_tr_g = R_tr_g.to_dense() 
new_R_tr_u = R_tr_u.to_dense() 

# P_tilde = R^T @ R
train_group_P = train_group_mceg_norm.T @ train_group_mceg_norm 
train_user_P = train_user_mceg_norm.T @ train_user_mceg_norm

#  P_bar = P_tilde◦s
train_group_P.data **= args.power
train_user_P.data **= args.power
new_P = filter(train_user_P.data, train_group_P.data, args.user_alpha, args.group_alpha, args.filter_pair) # filter_?D_?D


# to device
train_group_P = train_group_P.to(device=device).float()
new_R_tr_g = new_R_tr_g.to(device=device).float()
train_user_P = train_user_P.to(device=device).float()
new_R_tr_u = new_R_tr_u.to(device=device).float()

new_P = new_P.to(device=device).float()

# Our model
train_group_results = new_R_tr_g @ (train_group_P) # train_group_P: [7057, 7057], new_R_tr_g: [290, 7057] 
train_user_results = new_R_tr_u @ (train_user_P) # 유저만 
new_t_group_results = new_R_tr_g @ (new_P) # 여기 수정함 

# Now get the results
inf_m = -99999 
# group_gt_mat = R_ts_g.to_dense()
# user_gt_mat = R_ts_u.to_dense()
# group_results = train_group_results.cpu() + (inf_m) * R_tr_g.to_dense() 
# user_results = train_user_results.cpu() + (inf_m) * R_tr_u.to_dense()
# group_gt_mat = group_gt_mat.cpu().detach().numpy()
# user_gt_mat = user_gt_mat.cpu().detach().numpy()
# group_results = group_results.cpu().detach().numpy()
# user_results = user_results.cpu().detach().numpy()

new_group_gt_mat = R_ts_g.to_dense()
new_group_results = new_t_group_results.cpu() + (inf_m) * R_tr_g.to_dense() 
new_group_gt_mat = new_group_gt_mat.cpu().detach().numpy()
new_group_results = new_group_results.cpu().detach().numpy()

# print(f"alpha: {a}, p: {p} ")
# print(f"Recall@K: {recall_at_k(group_gt_mat, group_results, g_neg, k=10):.4f}")
# print(f"NDCG@K: {ndcg_at_k(group_gt_mat, group_results, g_neg, k=10):.4f}")
# print(f"Recall@K: {recall_at_k(user_gt_mat, user_results, u_neg, k=10):.4f}")
# print(f"NDCG@K: {ndcg_at_k(user_gt_mat, user_results, u_neg, k=10):.4f}")

#print(f"NEW MODEL Recall@K: {recall_at_k(new_group_gt_mat, new_group_results, g_neg, k=10):.4f}")
print(f"NEW MODEL Hit@K: {hit_at_k(new_group_gt_mat, new_group_results, g_neg, k=5):.4f}")
print(f"NEW MODEL NDCG@K: {ndcg_at_k(new_group_gt_mat, new_group_results, g_neg, k=5):.4f}")






#python main.py --verbose=1 --user_alpha=1  --group_alpha=50 --alpha=0.6 --power=0.7 --filter_pair filter_1D_2D




