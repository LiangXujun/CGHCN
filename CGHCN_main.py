#%%
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

import math
import numpy as np

from copy import deepcopy
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

import os
os.chdir(os.path.realpath(os.path.join(__file__, '..')))
from prepareData import prepare_data

np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
class Config(object):
    def __init__(self):
        self.data_path = './datasets/dataset1'
        self.nfold = 5
        self.n_rep = 10 
        self.k_neig = 10
        self.emb_dim = 64
        self.hid_dim = 64
        self.dropout = 0.0
        self.num_epoches = 500

def calculate_evaluation_metrics(pred_mat, pos_edges, neg_edges):
    pos_pred_socres = pred_mat[pos_edges[0], pos_edges[1]]
    neg_pred_socres = pred_mat[neg_edges[0], neg_edges[1]]
    pred_scores = np.hstack((pos_pred_socres, neg_pred_socres))
    true_labels = np.hstack((np.ones(pos_pred_socres.shape[0]), np.zeros(neg_pred_socres.shape[0])))

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

def impute_zeros(inMat,inSim,k=10):
	mat = deepcopy(inMat)
	sim = deepcopy(inSim)
	(row,col) = mat.shape
 	# np.fill_diagonal(mat,0)
	indexZero = np.where(~mat.any(axis=1))[0]
	numIndexZeros = len(indexZero)

	np.fill_diagonal(sim,0)
	if numIndexZeros > 0:
		sim[:,indexZero] = 0
	for i in indexZero:
		currSimForZeros = sim[i,:]
		indexRank = np.argsort(currSimForZeros)

		indexNeig = indexRank[-k:]
		simCurr = currSimForZeros[indexNeig]

		mat_known = mat[indexNeig, :]
		
		if sum(simCurr) >0:  
			mat[i,: ] = np.dot(simCurr ,mat_known) / sum(simCurr)
	return mat


def generate_G_from_H(H):
    DV = np.sum(H, axis=1)
    DE = np.sum(H, axis=0)
    DV[DV==0] = np.inf
    DE[DE==0] = np.inf
    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    H = np.mat(H)
    HT = H.T
    G = DV2 @ H @ invDE @ HT @ DV2
    return G


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        z = G.matmul(x) + x
        return z


class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.1):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)

    def forward(self, x, G):
        x = F.dropout(x, self.dropout)
        x = self.hgc1(x, G)
        return x


class Net(nn.Module):
    def __init__(self, opt, num_users, num_items):
        super(Net, self).__init__()
        self.dropout = opt.dropout
        self.user_emb = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_users, opt.emb_dim)))
        self.item_emb = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_items, opt.emb_dim)))
        self.user_encoder_s = GraphConv(opt.emb_dim, opt.hid_dim)
        self.item_encoder_s = GraphConv(opt.emb_dim, opt.hid_dim)
        self.user_encoder = HGNN_embedding(opt.hid_dim, opt.hid_dim, 0)
        self.item_encoder = HGNN_embedding(opt.hid_dim, opt.hid_dim, 0)
        # self.decoder = BiDecoder(args.hid_dim, args.bdc_dropout)

    def forward(self, Gs, Gh):
        user_Gs = Gs['user']
        item_Gs = Gs['item']
        user_Gh = Gh['user']
        item_Gh = Gh['item']
        
        user_x = F.dropout(self.user_emb, 0)
        item_x = F.dropout(self.item_emb, 0)
        
        user_z1 = self.user_encoder_s(user_x, user_Gs)
        item_z1 = self.item_encoder_s(item_x, item_Gs)
        
        user_z1 = F.dropout(user_z1, self.dropout)
        item_z1 = F.dropout(item_z1, self.dropout)
	    
        user_z = self.user_encoder(user_z1, user_Gh)
        item_z = self.item_encoder(item_z1, item_Gh)
        # user_z = torch.concat([user_x, user_z1, user_z2], 1)
        # item_z = torch.concat([item_x, item_z1, item_z2], 1)
        pred_ratings = torch.mm(user_z, item_z.t())
        return pred_ratings

#%%
opt = Config()

dataset = prepare_data(opt)
num_users, num_items = dataset['md_p'].shape


graph_d = np.array(np.where(dataset['dd_g']))
graph_m = np.array(np.where(dataset['mm_g']))

Gd_s = torch.LongTensor(graph_d).to(device)
Gm_s = torch.LongTensor(graph_m).to(device)
Gs = {'user':Gm_s, 'item':Gd_s}

metric_tab = np.zeros((opt.nfold, opt.n_rep, 4))

for ir in range(opt.n_rep):
    kf = KFold(n_splits = opt.nfold, shuffle = True)
    for ik, (train, test) in enumerate(kf.split(range(num_users))):
        Htrain = dataset['md_p'].copy() 
        Htrain[test,] = 0
        Htrain = impute_zeros(Htrain, dataset['mm']['data'])
        
        Hm = np.minimum(np.c_[Htrain, Htrain@(Htrain.T@Htrain)], 1)
        Hd = np.minimum(np.c_[Htrain.T, Htrain.T@(Htrain@Htrain.T)], 1)
        
        Gm = generate_G_from_H(Hm)
        Gd = generate_G_from_H(Hd)
        
        Htrain = torch.tensor(Htrain, dtype = torch.float).to(device)
        Gh = {'user':torch.tensor(Gm, dtype = torch.float).to(device), 'item':torch.tensor(Gd, dtype = torch.float).to(device)}
        model = Net(opt, num_users, num_items).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-4)
        for epoch in tqdm(range(opt.num_epoches)):
            scores = model(Gs, Gh)
            loss = F.binary_cross_entropy_with_logits(scores, Htrain)
            # loss = (1 - opt.alpha)*loss_sum[train,].sum() + opt.alpha*loss_sum[test,].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        y_score = model(Gs, Gh).detach().cpu().numpy()
        test = test[dataset['md_p'][test,].sum(1)!=0]
        metrics = calculate_evaluation_metrics(y_score[test,], np.where(dataset['md_p'][test,]==1), np.where(dataset['md_p'][test,]==0))
	metrics_tab[ik,ir,] = metrics

#%%
np.savez('/result/CGHCN_DS1_local_result.npz', )
