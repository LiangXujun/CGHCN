# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 22:31:30 2023

@author: liangxj
"""

#%%
import math
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import inv
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

import os
os.chdir(os.path.realpath(os.path.join(__file__, '..')))
from prepareData import prepare_data

np.random.seed(123)

#%%
def calculate_evaluation_metrics(pred_scores, true_labels):
    auc = roc_auc_score(true_labels, pred_scores)
    average_precision = average_precision_score(true_labels, pred_scores)

    pred_scores_mat = np.mat([pred_scores])
    true_labels_mat = np.mat([true_labels])
    sorted_predict_score = np.array(sorted(list(set(np.array(pred_scores_mat).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[
        (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(pred_scores_mat, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    TP = predict_score_matrix * true_labels_mat.T
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = true_labels_mat.sum() - TP
    TN = len(true_labels_mat.T) - TP - FP - FN
    tpr = TP / (TP + FN)

    f1_score_list = 2 * TP / (len(true_labels_mat.T) + TP - TN)
    accuracy_list = (TP + TN) / len(true_labels_mat.T)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index, 0]
    accuracy = accuracy_list[max_index, 0]
    return np.array([auc, average_precision, f1_score, accuracy])

#%%
def gen_knn_hg(X, n_neighbors, is_prob = False, with_feature=False):
    X = np.array(X)
    n_nodes = X.shape[0]
    n_edges = n_nodes

    m_dist = pairwise_distances(X)

    # top n_neighbors+1
    m_neighbors = np.argpartition(m_dist, kth=n_neighbors+1, axis=1)
    m_neighbors_val = np.take_along_axis(m_dist, m_neighbors, axis=1)

    m_neighbors = m_neighbors[:, :n_neighbors+1]
    m_neighbors_val = m_neighbors_val[:, :n_neighbors+1]

    # check
    for i in range(n_nodes):
        if not np.any(m_neighbors[i, :] == i):
            m_neighbors[i, -1] = i
            m_neighbors_val[i, -1] = 0.

    node_idx = m_neighbors.reshape(-1)
    edge_idx = np.tile(np.arange(n_edges).reshape(-1, 1), (1, n_neighbors+1)).reshape(-1)

    if not is_prob:
        values = np.ones(node_idx.shape[0])
    else:
        avg_dist = np.mean(m_dist)
        m_neighbors_val = m_neighbors_val.reshape(-1)
        values = np.exp(-np.power(m_neighbors_val, 2.) / np.power(avg_dist, 2.))

    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))
    # w = np.ones(n_edges)
    return H/n_edges


def generate_G_from_H(H):
    DV = np.sum(H, axis=1)
    DE = np.sum(H, axis=0)
    DV[DV==0] = np.inf
    DE[DE==0] = np.inf
    invDE = np.diag(np.power(DE, -1))
    DV2 = np.diag(np.power(DV, -0.5))
    HT = H.T
    G = DV2 @ H @ invDE @ HT @ DV2
    return G

#%%
class Config(object):
    def __init__(self):
        self.data_path = './dataset/dataset2/'
        self.k_neig = 10
        self.lam = 20
        self.mu = 1
        self.nfold = 5
        self.n_rep = 10

opt = Config()        
dataset = prepare_data(opt)
num_users, num_items = dataset['md_p'].shape
pos_idx = dataset['pos_idx']
neg_idx = dataset['neg_idx']
num_pos = len(pos_idx[0])

#%%
metric_tab = np.zeros((opt.nfold, opt.n_rep, 4))
for ir in range(opt.n_rep):
    kf = KFold(n_splits = opt.nfold, shuffle = True)
    for ik, (train, test) in enumerate(kf.split(range(num_users))):
        Rtrain = dataset['md_p'].copy() 
        Rtrain[test,] = 0
        train_pos = np.where(Rtrain)
        train_neg = np.where(Rtrain==0)
        idx = np.random.choice(len(train_neg[0]), len(train_pos[0]), replace = False)
        train_neg =train_neg[0][idx], train_neg[1][idx]
        
        Xtrain = np.r_[np.c_[dataset['mm']['data'][train_pos[0]], dataset['dd']['data'][train_pos[1]]],
                       np.c_[dataset['mm']['data'][train_neg[0]], dataset['dd']['data'][train_neg[1]]]]
        n = len(train_pos[0])
        y_train = np.c_[np.r_[np.ones((n, 1)), np.zeros((n, 1))], np.r_[np.zeros((n, 1)), np.ones((n, 1))]]
        
        H = np.array(gen_knn_hg(Xtrain, opt.k_neig).todense())
        G = generate_G_from_H(H)
        Dt = np.eye(n*2) - G
        P = opt.lam*inv(Xtrain.T@G@Xtrain + opt.lam*Xtrain.T@Xtrain + opt.mu*np.eye(Xtrain.shape[1]))@Xtrain.T@y_train
        
        R_tmp = np.ones_like(Rtrain)
        R_tmp[train,] = 0
        test_idx = np.where(R_tmp)
        Xtest = np.c_[dataset['mm']['data'][test_idx[0]], dataset['dd']['data'][test_idx[1]]]
        scores = Xtest@P
        y_score = scores[:,0] - scores[:,1]
        
        metrics = calculate_evaluation_metrics(y_score, dataset['md_p'][test_idx])
        metric_tab[ik,ir,] = metrics
        print(metrics)
