import csv
import torch
import random
import numpy as np

#%%
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return np.array(md_data)
        # return torch.FloatTensor(md_data)


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return np.array(md_data)
        # return torch.FloatTensor(md_data)


def construct_H_with_KNN(dis_mat, k_neig, is_probH = True, m_prob = 1):
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx,center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec).squeeze())
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig-1] = center_idx
        
        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx,center_idx] = np.exp(-dis_vec[node_idx]**2/(m_prob*avg_dis)**2)
            else:
                H[node_idx,center_idx] = 1.0
    return H


def get_KNN_graph(sim_mat, k_neig):
    graph = np.zeros(sim_mat.shape)
    for i in range(sim_mat.shape[0]):
        idx = np.argsort(sim_mat[i,])[::-1]
        for j in idx[0:k_neig+1]:
            if j == i:
                continue
            else:
                graph[i,j] = 1
                graph[j,i] = 1
    # graph = graph + np.eye(graph.shape[0])
    return graph


def prepare_data(opt):
    dataset = dict()
    dataset['md_p'] = read_csv(opt.data_path + '\\m-d.csv')
    dataset['md_true'] = read_csv(opt.data_path + '\\m-d.csv')

    dataset['pos_idx'] = np.where(dataset['md_p'] != 0)
    dataset['neg_idx'] = np.where(dataset['md_p'] == 0)

    dd_matrix = read_csv(opt.data_path + '\\d-d.csv')
    dd_edge_index = np.where(dd_matrix)
    dataset['dd'] = {'data': dd_matrix, 'edge_index': dd_edge_index}
    dataset['dd_g'] =  get_KNN_graph(dd_matrix, opt.k_neig)
    dataset['dd_hg'] = construct_H_with_KNN(1 - dd_matrix, opt.k_neig, is_probH = False)
    
    Sd = dataset['dd_g']
    Bd = Sd*Sd.T
    Ud = Sd - Bd
    dataset['Bd'] = Bd
    dataset['Ud'] = Ud

    mm_matrix = read_csv(opt.data_path + '\\m-m.csv')
    mm_edge_index = np.where(mm_matrix)
    dataset['mm'] = {'data': mm_matrix, 'edge_index': mm_edge_index}
    dataset['mm_g'] =  get_KNN_graph(mm_matrix, opt.k_neig)
    dataset['mm_hg'] = construct_H_with_KNN(1 - mm_matrix, opt.k_neig, is_probH = False)
    
    Sm = dataset['mm_g']
    Bm = Sm*Sm.T
    Um = Sm - Bm
    dataset['Bm'] = Bm
    dataset['Um'] = Um
    
    return dataset
