import pandas as pd
import scipy.sparse as sp
import numpy as np
import torch
import os


class Dataset:
    def __init__(self, path):
        self.path = path
        self.group_train_file = path + '/groupRatingTrain.txt'
        self.group_test_file = path + '/groupRatingTest.txt'
        self.train_file = path + '/userRatingTrain.txt'
        self.test_file = path + '/userRatingTest.txt'

    def datasetToDataFrame(self, file_train, user=True):
        data = []
        with open(file_train, 'r') as file:
            for line in file:
                line = line.strip().split()  
                data.append([int(x) for x in line])  

        if user:
            df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])
        else:
            df = pd.DataFrame(data, columns=['group_id', 'item_id', 'rating'])
        return df 
    
    def getDataset(self):
        df_group_train = self.datasetToDataFrame(self.group_train_file, False)
        df_group_test = self.datasetToDataFrame(self.group_test_file, False)
        df_user_train = self.datasetToDataFrame(self.train_file)
        df_user_test = self.datasetToDataFrame(self.test_file)

        # 문제점: group에서의 item이 user에서의 item에 없는 경우 발생
        # 해결 
        unique_group_item = df_group_train['item_id'].unique()
        unique_user_item = df_user_train['item_id'].unique()
        group_not_in_user = set(unique_group_item) - set(unique_user_item) # 두 집합의 차집합을 계산하여 각 데이터프레임에서 겹치지 않는 ID 찾기
        user_not_in_group = set(unique_user_item) - set(unique_group_item)
        df_group_train = df_group_train[~df_group_train['item_id'].isin(group_not_in_user)]
        df_user_train = df_user_train[~df_user_train['item_id'].isin(user_not_in_group)]

        # unique number of groups, items, users
        train_groups = sorted(df_group_train['group_id'].unique())
        train_group_items = sorted(df_group_train['item_id'].unique())
        test_groups = sorted(df_group_test['group_id'].unique())
        test_group_items = sorted(df_group_test['item_id'].unique())
        train_users = sorted(df_user_train['user_id'].unique())
        train_user_items = sorted(df_user_train['item_id'].unique())
        test_users = sorted(df_user_test['user_id'].unique())
        test_user_items = sorted(df_user_test['item_id'].unique())

        # dictionary for group, item, user
        train_group_to_index = {group: index for index, group in enumerate(train_groups)}
        train_group_item_to_index = {item: index for index, item in enumerate(train_group_items)}   
        test_group_user_to_index = {group: index for index, group in enumerate(test_groups)}
        test_group_item_to_index = {item: index for index, item in enumerate(test_group_items)} 
        train_user_to_index = {user: index for index, user in enumerate(train_users)}
        train_user_item_to_index = {item: index for index, item in enumerate(train_user_items)} 
        test_user_to_index = {user: index for index, user in enumerate(test_users)}
        test_user_item_to_index = {item: index for index, item in enumerate(test_user_items)} 

        # number of groups, items, users
        train_num_groups = len(train_groups)
        train_group_num_items = len(train_group_items) # 7686
        test_num_groups = len(test_groups)
        test_group_num_items = len(test_group_items)
        train_num_users = len(train_users)
        train_user_num_items = len(train_user_items) # 7057
        test_num_users = len(test_users)
        test_user_num_items = len(test_user_items)

        # create sparse matrix (id, item, rating)
        train_group_sparse_matrix = sp.lil_matrix((train_num_groups, train_group_num_items))
        test_group_sparse_matrix = sp.lil_matrix((test_num_groups, test_group_num_items))
        train_user_sparse_matrix = sp.lil_matrix((train_num_users, train_user_num_items))
        test_user_sparse_matrix = sp.lil_matrix((test_num_users, test_user_num_items))

        # fill sparse matrix
        for _, row in df_group_train.iterrows():
            group_index = train_group_to_index[row['group_id']]
            item_index = train_group_item_to_index[row['item_id']]
            train_group_sparse_matrix[group_index, item_index] = row['rating']

        for _, row in df_group_test.iterrows():
            group_index = test_group_user_to_index[row['group_id']]
            item_index = test_group_item_to_index[row['item_id']]
            test_group_sparse_matrix[group_index, item_index] = row['rating']

        for _, row in df_user_train.iterrows():
            user_index = train_user_to_index[row['user_id']]
            item_index = train_user_item_to_index[row['item_id']]
            train_user_sparse_matrix[user_index, item_index] = row['rating']

        for _, row in df_user_test.iterrows():
            user_index = test_user_to_index[row['user_id']]
            item_index = test_user_item_to_index[row['item_id']]
            test_user_sparse_matrix[user_index, item_index] = row['rating']

        # 희소 행렬을 csr_matrix로 변환
        train_group_sparse_matrix = train_group_sparse_matrix.tocsr()
        test_group_sparse_matrix = test_group_sparse_matrix.tocsr()
        train_user_sparse_matrix = train_user_sparse_matrix.tocsr()
        test_user_sparse_matrix = test_user_sparse_matrix.tocsr()

        train_group_csr_matrix = train_group_sparse_matrix
        test_group_csr_matrix = test_group_sparse_matrix
        train_user_csr_matrix = train_user_sparse_matrix
        test_user_csr_matrix = test_user_sparse_matrix

        # Convert CSR matrix to COO format (Coordinate List)
        train_group_coo_matrix = train_group_csr_matrix.tocoo()
        test_group_coo_matrix = test_group_csr_matrix.tocoo()
        train_user_coo_matrix = train_user_csr_matrix.tocoo()
        test_user_coo_matrix = test_user_csr_matrix.tocoo()

        # Convert COO matrix to PyTorch sparse tensor
        train_group_data = torch.FloatTensor(train_group_coo_matrix.data)
        test_group_data = torch.FloatTensor(test_group_coo_matrix.data)
        train_user_data = torch.FloatTensor(train_user_coo_matrix.data)
        test_user_data = torch.FloatTensor(test_user_coo_matrix.data)

        # Convert COO matrix to PyTorch LongTensor
        train_group_indices = torch.LongTensor(np.vstack((train_group_coo_matrix.row, train_group_coo_matrix.col)))
        test_group_indices = torch.LongTensor(np.vstack((test_group_coo_matrix.row, test_group_coo_matrix.col)))
        train_user_indices = torch.LongTensor(np.vstack((train_user_coo_matrix.row, train_user_coo_matrix.col)))
        test_user_indices = torch.LongTensor(np.vstack((test_user_coo_matrix.row, test_user_coo_matrix.col)))

        # Convert COO matrix to PyTorch sparse tensor
        train_group_result = torch.sparse_coo_tensor(train_group_indices, train_group_data, torch.Size(train_group_coo_matrix.shape))
        test_group_result = torch.sparse_coo_tensor(test_group_indices, test_group_data, torch.Size(test_group_coo_matrix.shape))
        train_user_result = torch.sparse_coo_tensor(train_user_indices, train_user_data, torch.Size(train_user_coo_matrix.shape))
        test_user_result = torch.sparse_coo_tensor(test_user_indices, test_user_data, torch.Size(test_user_coo_matrix.shape))

        return train_group_result, test_group_result, train_user_result, test_user_result

