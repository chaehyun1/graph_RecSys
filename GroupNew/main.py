import random
import numpy as np
import torch
import scipy.sparse as sp
import os
import argparse
from utils import csr2torch, recall_at_k, ndcg_at_k, normalize_sparse_adjacency_matrix, inference_new
from dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_directory = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="CAMRa2011",
    help="Either gowalla, yelp, amazon, or ml-1m",
)
parser.add_argument(
    "--verbose",
    type=int,
    default=1,
    help="Whether to print the results or not. 1 prints the results, 0 does not.",
)
parser.add_argument("--alpha", type=float, default=0.6, help="For normalization of R")
parser.add_argument("--power", type=float, default=0.9, help="For normalization of P")


random.seed(2022)
np.random.seed(2022)


if __name__ == "__main__":
    args = parser.parse_args()  

    if args.verbose:
        print(f"Device: {device}")

    # load dataset
    dataset = args.dataset
    path = f'data/{dataset}/'
    data = Dataset(path)
    R_tr_g, R_ts_g, R_tr_u, R_ts_u = data.getDataset()

    # shape
    train_n_groups = R_tr_g.shape[0]
    train_group_n_items = R_tr_g.shape[1]
    test_n_groups = R_ts_g.shape[0]
    test_group_n_items = R_ts_g.shape[1]
    train_n_users = R_tr_u.shape[0]
    train_user_n_items = R_tr_u.shape[1]
    test_n_users = R_ts_u.shape[0]
    test_user_n_items = R_ts_u.shape[1]

    if args.verbose:
        print(f"number of tr_groups: {train_n_groups}")
        print(f"number of tr_groups_items: {train_group_n_items}")
        print(f"number of ts_groups: {test_n_groups}")
        print(f"number of ts_groups_items: {test_group_n_items}")
        print(f"number of tr_users: {train_n_users}")
        print(f"number of tr_users_items: {train_user_n_items}")
        print(f"number of ts_users: {test_n_users}")
        print(f"number of ts_users_items: {test_user_n_items}")

    # number of overall ratings
    train_group_n_inters = torch.nonzero(R_tr_g._values()).cpu().size(0) + torch.nonzero(R_tr_g[0]._values()).cpu().size(0) 
    test_group_n_inters = torch.nonzero(R_ts_g._values()).cpu().size(0) + torch.nonzero(R_ts_g[0]._values()).cpu().size(0)
    train_user_n_inters = torch.nonzero(R_tr_u._values()).cpu().size(0) + torch.nonzero(R_tr_u[0]._values()).cpu().size(0)
    test_user_n_inters = torch.nonzero(R_ts_u._values()).cpu().size(0) + torch.nonzero(R_ts_u[0]._values()).cpu().size(0)

    print()
    if args.verbose:
        print(f"number of overall ratings(tr_group): {train_group_n_inters}")
        print(f"number of overall ratings(ts_group): {test_group_n_inters}")
        print(f"number of overall ratings(tr_user): {train_user_n_inters}")
        print(f"number of overall ratings(ts_user): {test_user_n_inters}")

    # normalize R
    train_group_mceg_norm = normalize_sparse_adjacency_matrix(R_tr_g.to_dense(), args.alpha)
    test_group_mceg_norm = normalize_sparse_adjacency_matrix(R_ts_g.to_dense(), args.alpha)
    train_user_mceg_norm = normalize_sparse_adjacency_matrix(R_tr_u.to_dense(), args.alpha)
    test_user_mceg_norm = normalize_sparse_adjacency_matrix(R_ts_u.to_dense(), args.alpha)

    # R = R_tr.to_dense()
    new_R_tr_g = R_tr_g.to_dense()
    new_R_ts_g = R_ts_g.to_dense()
    new_R_tr_u = R_tr_u.to_dense()
    new_R_ts_u = R_ts_u.to_dense()

    # P = R^T @ R
    train_group_P = train_group_mceg_norm.T @ train_group_mceg_norm
    test_group_P = test_group_mceg_norm.T @ test_group_mceg_norm
    train_user_P = train_user_mceg_norm.T @ train_user_mceg_norm
    test_user_P = test_user_mceg_norm.T @ test_user_mceg_norm

    # P = P^p
    train_group_P.data **= args.power
    test_group_P.data **= args.power
    train_user_P.data **= args.power
    test_user_P.data **= args.power

    # to device
    train_group_P = train_group_P.to(device=device).float()
    new_R_tr_g = new_R_tr_g.to(device=device).float()
    test_group_P = test_group_P.to(device=device).float()
    new_R_ts_g = new_R_ts_g.to(device=device).float()
    train_user_P = train_user_P.to(device=device).float()
    new_R_tr_u = new_R_tr_u.to(device=device).float()
    test_user_P = test_user_P.to(device=device).float()
    new_R_ts_u = new_R_ts_u.to(device=device).float()

    # Our model
    train_group_results = new_R_tr_g @ (train_group_P) # train_group_P: [7057, 7057], new_R_tr_g: [290, 7057] -> 문제 발생: NaN
    test_group_results = new_R_ts_g @ (test_group_P)
    train_user_results = new_R_tr_u @ (train_user_P)
    test_user_results = new_R_ts_u @ (test_user_P)
    new_model =  new_R_tr_u @ inference_new(train_user_P, train_group_P, 1, 0, "cpu")

    # Now get the results
    train_group_gt_mat = R_tr_g.to_dense()
    test_group_gt_mat = R_ts_g.to_dense()
    train_user_gt_mat = R_tr_u.to_dense()
    test_user_gt_mat = R_ts_u.to_dense()
    train_group_results = train_group_results + (-99999) * R_tr_g.to_dense() 
    test_group_results = test_group_results + (-99999) * R_ts_g.to_dense()
    train_user_results = train_user_results + (-99999) * R_tr_u.to_dense()
    test_user_results = test_user_results + (-99999) * R_ts_u.to_dense()
    train_group_gt_mat = train_group_gt_mat.cpu().detach().numpy()
    test_group_gt_mat = test_group_gt_mat.cpu().detach().numpy()
    train_user_gt_mat = train_user_gt_mat.cpu().detach().numpy()
    test_user_gt_mat = test_user_gt_mat.cpu().detach().numpy()
    train_group_results = train_group_results.cpu().detach().numpy()
    test_group_results = test_group_results.cpu().detach().numpy()
    train_user_results = train_user_results.cpu().detach().numpy()
    test_user_results = test_user_results.cpu().detach().numpy()

    new_model_gt_mat = R_tr_u.to_dense()
    new_model_results = new_model + (-99999) * R_tr_u.to_dense()
    new_model_gt_mat = new_model_gt_mat.cpu().detach().numpy() 
    new_model_results = new_model_results.cpu().detach().numpy()
    print(f"NEW MODEL Recall@20: {recall_at_k(new_model_gt_mat, new_model_results, k=20):.4f}")
    print(f"NEW MODEL NDCG@20: {ndcg_at_k(new_model_gt_mat, new_model_results, k=20):.4f}")

    # print(f"alpha: {a}, p: {p} ")
    print(f"Recall@20: {recall_at_k(train_group_gt_mat, train_group_results, k=20):.4f}")
    print(f"NDCG@20: {ndcg_at_k(train_group_gt_mat, train_group_results, k=20):.4f}")
    print(f"Recall@20: {recall_at_k(test_group_gt_mat, test_group_results, k=20):.4f}")
    print(f"NDCG@20: {ndcg_at_k(test_group_gt_mat, test_group_results, k=20):.4f}")
    print(f"Recall@20: {recall_at_k(train_user_gt_mat, train_user_results, k=20):.4f}")
    print(f"NDCG@20: {ndcg_at_k(train_user_gt_mat, train_user_results, k=20):.4f}")
    print(f"Recall@20: {recall_at_k(test_user_gt_mat, test_user_results, k=20):.4f}")
    print(f"NDCG@20: {ndcg_at_k(test_user_gt_mat, test_user_results, k=20):.4f}")
