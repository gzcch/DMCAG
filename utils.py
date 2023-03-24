from __future__ import division, print_function
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score
import scipy.io as scio
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

nmi = normalized_mutual_info_score
vmeasure = v_measure_score
ari = adjusted_rand_score

class multiViewDataset2(Dataset):

    def __init__(self, dataName,viewNumber,method,pretrain):
        dataPath = './data/' + dataName + '.mat'
        matData = scio.loadmat(dataPath)
        self.data=[]
        self.viewNumber = viewNumber
        for viewIndex in range(viewNumber):
            temp=matData['X'+str(viewIndex+1)].astype(np.float32)
            if self.viewNumber>=2:
                temp=min_max_scaler.fit_transform(temp)
            self.data.append(temp)
        Y = matData['Y'][0]
        self.labels = Y
        self.pretrain=pretrain


    def __getitem__(self, index):
        data_tensor=[]
        for viewIndex in range(self.viewNumber):
            m=self.data[viewIndex][index]
            data_tensor.append(torch.from_numpy(self.data[viewIndex][index]))
        label= self.labels[index]
        label_tensor=torch.tensor(label)
        index_tensor = torch.tensor(index)

        return data_tensor, label_tensor, index_tensor


    def __len__(self):
        return len(self.labels)

class imagedataset(Dataset):

    def __init__(self, dataName,viewNumber,method,pretrain):
        dataPath = './data/' + dataName + '.mat'
        matData = scio.loadmat(dataPath)
        self.data=[]
        self.viewNumber = viewNumber
        for viewIndex in range(viewNumber):
            temp=matData['X'+str(viewIndex+1)].astype(np.float32)
            if self.viewNumber>=6:
                temp=min_max_scaler.fit_transform(temp)
            self.data.append(temp)
        Y = matData['Y'][0]
        self.labels = Y
        self.pretrain=pretrain


    def __getitem__(self, index):
        data_tensor=[]
        for viewIndex in range(self.viewNumber):
            m=self.data[viewIndex][index]
            data_tensor.append(torch.from_numpy(self.data[viewIndex][index]))
        label= self.labels[index]
        label_tensor=torch.tensor(label)
        index_tensor = torch.tensor(index)

        return data_tensor, label_tensor, index_tensor


    def __len__(self):
        return len(self.labels)


class WKLDiv(torch.nn.Module):
    def __init__(self):
        super(WKLDiv, self).__init__()

    def forward(self, q_logit, p, w):
        p_logit=torch.log(p + 1e-12)
        kl = torch.sum(p * (p_logit- q_logit)*w, 1)
        return torch.mean(kl)


#######################################################
# Evaluate Critiron
#######################################################

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

