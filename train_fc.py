from __future__ import print_function, division
import argparse
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
import cvxopt
from sklearn import preprocessing
import warnings
from time import time
import tqdm
warnings.filterwarnings('ignore')
min_max_scaler = preprocessing.MinMaxScaler()
normalize = preprocessing.Normalizer()

min_max_scaler = preprocessing.MinMaxScaler()
normalize = preprocessing.Normalizer()

from utils import cluster_acc, WKLDiv, multiViewDataset2
import os

cpu_num = 10
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, n_z):
        super(ClusteringLayer, self).__init__()
        self.centroids = Parameter(torch.Tensor(n_clusters, n_z),requires_grad=True)

    def forward(self, x):
        q = 1.0 / (1 + torch.sum(
            torch.pow(x.unsqueeze(1) - self.centroids, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        return q


class SingleViewModel(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, pretrain):
        super(SingleViewModel, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            Linear(n_input, n_enc_1),
            nn.ReLU(inplace=True),
            Linear(n_enc_1, n_enc_2),
            nn.ReLU(inplace=True),
            Linear(n_enc_2, n_enc_3),
            nn.ReLU(inplace=True),
            Linear(n_enc_3, n_z),
        )

        for m in self.encoder:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(m.bias, 0)

        self.decoder = nn.Sequential(
            Linear(n_z, n_dec_1),
            nn.ReLU(inplace=True),
            Linear(n_dec_1, n_dec_2),
            nn.ReLU(inplace=True),
            Linear(n_dec_2, n_dec_3),
            nn.ReLU(inplace=True),
            Linear(n_dec_3, n_input)
        )

        for m in self.decoder:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(m.bias, 0)

        self.pretrain = pretrain
        self.clusteringLayer = ClusteringLayer(args.n_clusters, n_z)
        self.A_weight=torch.nn.Parameter(torch.Tensor( n_z,args.arch ),requires_grad=False)
        self.c_weight = torch.nn.Parameter(torch.Tensor(args.arch, args.batch_size), requires_grad=False)

    def computeA(self, x, mode):
        if mode == 'cos':
            a = F.normalize(x, p=2, dim=1)
            b = F.normalize(x.T, p=2, dim=0)
            A = torch.mm(a, b)
            A = (A + 1) / 2
        if mode == 'kernel':
            x = torch.nn.functional.normalize(x, p=1.0, dim=1)
            a = x.unsqueeze(1)
            A = torch.exp(-torch.sum(((a - x.unsqueeze(0)) ** 2) * 1000, dim=2))

        if mode == 'knn':
            dis2 = (-2 * x.mm(x.t())) + torch.sum(torch.square(x), axis=1, keepdim=True) + torch.sum(
                torch.square(x.t()), axis=0, keepdim=True)
            A = torch.zeros(dis2.shape).cuda()
            A[(torch.arange(len(dis2)).unsqueeze(1), torch.topk(dis2, args.k, largest=False).indices)] = 1
            A = A.detach()
        if mode=='sigmod':

            A=1/(1+torch.exp(-torch.mm(x,x.T)))

        return A

    def computegcn(self, x, Af):
        D = torch.sum(Af, dim=1)
        D = 1 / D
        D = torch.sqrt(D)
        D = torch.diag(D)
        ATi = torch.matmul(torch.matmul(D, Af), D)
        z_refine = torch.matmul(ATi, torch.matmul(ATi, x))

        return z_refine, ATi

    def forward(self, x):
        z = self.encoder(x)
        # z = torch.nn.functional.normalize(z, p=1.0, dim=1)

        if self.pretrain:
            x_bar = self.decoder(z)
            return x_bar, z
        z_self=torch.matmul(self.A_weight,self.c_weight)

        z_self=z_self.T
        x_bar=self.decoder(z)
        q=self.clusteringLayer(z)

        #q = self.clusteringLayer(z)

        return z_self, z, x_bar,self.c_weight, q


class MultiViewModel(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 pretrain,
                 save_path):
        super(MultiViewModel, self).__init__()
        self.pretrain = pretrain
        self.save_path = save_path
        self.n_clusters = n_clusters
        self.viewNumber = args.viewNumber
        self.Al_weight=torch.nn.Parameter(torch.Tensor( args.viewNumber* n_z,args.arch ),requires_grad=False)
        self.cl_weight = torch.nn.Parameter(torch.Tensor(args.arch, args.batch_size), requires_grad=False)

        aes = []

        for viewIndex in range(self.viewNumber):
            aes.append(SingleViewModel(
                n_enc_1=n_enc_1,
                n_enc_2=n_enc_2,
                n_enc_3=n_enc_3,
                n_dec_1=n_dec_1,
                n_dec_2=n_dec_2,
                n_dec_3=n_dec_3,
                n_input=n_input[viewIndex],
                n_z=n_z,
                pretrain=self.pretrain))

        self.aes = nn.ModuleList(aes)

    def forward(self, x):
        outputs = []
        for viewIndex in range(self.viewNumber):
            outputs.append(self.aes[viewIndex](x[viewIndex]))
        outputs.append(self.Al_weight)
        outputs.append(self.cl_weight)
        return outputs


def computegcn(z, Af):
    D = torch.sum(Af, dim=1)
    D = 1 / D
    D = torch.sqrt(D)
    D = torch.diag(D)
    ATi = torch.matmul(torch.matmul(D, Af), D)
    z_refine = torch.matmul(ATi, torch.matmul(ATi, z))

    return z_refine, ATi


def make_qp(x, centroids):
    q = 1.0 / (1.0*0.001 + torch.sum(torch.pow(x.unsqueeze(1) - torch.tensor(centroids).cuda(), 2), 2))
    q = (q.t() / torch.sum(q, 1)).t()
    p = target_distribution(q)
    return q, p


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def graph_fusion(graphlist, wlist):
    torch.cuda.empty_cache()
    with torch.no_grad():
        fusiongraph =torch.sum(torch.mul(graphlist, wlist.T.unsqueeze(dim=1)), dim=0)/ (torch.sum(wlist))
    for v in range(graphlist.shape[0]):
        w = 1 / (2 * (torch.norm((graphlist[v] - fusiongraph), p='fro')))
        wlist.T[v] = w

    return fusiongraph, wlist

import sys
def calculate_c(Z, A , gamma=1):
    sys.stdout = open(os.devnull, 'w')


    c_min = torch.rand(A.shape[1], Z.shape[1]).cuda()
    H_0=2*gamma*torch.eye(A.shape[1]).cuda()+2*torch.mm(A.T,A)
    H=(H_0+H_0.T)/2
    H=H.cpu().detach().data.numpy()
    r=A.shape[1]
    for i in range(Z.shape[1]):
        f=-2*torch.matmul(Z[:,i].T,A)
        f=f.T
        f=f.cpu().detach().data.numpy()


        if i==0:
            c_min=quadprog(H,f, L=-np.eye(r), k=np.zeros((r,1),dtype=float),
                           Aeq=np.ones((1,r),dtype=float),beq=1)
            c_min=torch.tensor(c_min)
        else:
            tem= quadprog(H,f, L=-np.eye(r), k=np.zeros((r,1),dtype=float),
                           Aeq=np.ones((1,r),dtype=float),beq=1)
            c_min=torch.cat((c_min,torch.tensor(tem)),dim=1)
    sys.stdout = sys.__stdout__
    #for iter in range(1000):
        #temp=(torch.mm(A.T,Z)/torch.mm(torch.mm(A.T,A)+0*gamma*torch.eye(A.shape[1]).cuda(),c_min))
        #c_min = c_min * temp
        #temp = (torch.mm(A.T, Z)- torch.mm(torch.mm(A.T, A) + 0 * gamma * torch.eye(A.shape[1]).cuda(), c_min))/(A.shape[1]*Z.shape[1])
        #c_min=c_min+0.001*temp
        #if torch.gt(up_c,0)==1:
            #c_min=up_c
        #else:
            #break
    #temp= torch.matmul(A.T,A)+gamma*torch.eye(A.shape[1]).cuda()
    #c_min=torch.matmul(torch.matmul(torch.linalg.inv(temp),A.T),Z)
    #c_min=torch.abs(c_min)
    c_min=torch.clamp(c_min,0,1)
    return (c_min.cuda()).float()
def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    n_var = H.shape[1]
    H = np.float64(H)
    f= np.float64(f)
    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if L is not None or k is not None:
        assert(k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert(Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')

    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq)

    return np.array(sol['x'])

def cacluate_U(C):
    #(evals, evecs) = torch.eig(C, eigenvectors=True)
    U, _, _ = torch.linalg.svd(C.T)
    #U = U[:, 0:args.n_clusters]
    U = U[:, 0:args.n_clusters]
    return U

def pretrain_aes():
    save_path = args.save_path
    viewNumber = args.viewNumber
    model = MultiViewModel(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.arch,
        pretrain=True,
        save_path=save_path).cuda()
    #if args.noise==0:
    #model.load_state_dict(torch.load(args.save_path))
    dataset = multiViewDataset2(args.dataset, args.viewNumber, args.method, True)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=args.lr)
    print('Start Pretraining.')
    for epoch in tqdm.tqdm(range(1000)):
        for batch_idx, (x, _, _) in enumerate(dataLoader):
            loss = 0.
            for viewIndex in range(viewNumber):
                x[viewIndex] = x[viewIndex].cuda()
        output = model(x)
        for viewIndex in range(viewNumber):
            loss = loss + F.mse_loss(output[viewIndex][0], x[viewIndex])
            #loss = loss +  nn.KLDivLoss(reduction='mean')(torch.tensor(0.001).log(), output[viewIndex][1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
           print("mseloss loss", loss)



    for batch_idx, (x, y, _) in enumerate(dataLoader):
        for viewIndex in range(viewNumber):
            x[viewIndex] = x[viewIndex].cuda()
        output = model(x)
        y = y.data.cpu().numpy()
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
    kmeans_arch = KMeans(n_clusters=args.arch, n_init=100)
    for viewIndex in range(args.viewNumber):
        z_v = output[viewIndex][1]

        kmeans_arch.fit_predict(z_v.cpu().detach().data.numpy())
        a = torch.tensor(kmeans_arch.cluster_centers_).cuda()
        model.aes[viewIndex].A_weight.data =a.T
        cmin=calculate_c(z_v.T,a.T,gamma=args.gamma)
        model.aes[viewIndex].c_weight.data=cmin
        f_temp=cacluate_U(cmin)
        kmeans.fit_predict(z_v.cpu().detach().data.numpy())
        model.aes[viewIndex].clusteringLayer.centroids.data = \
            torch.tensor(kmeans.cluster_centers_).cuda()

        if (viewIndex) == 0:
            z_all = z_v
            f_all = f_temp

        else:
            z_all = torch.cat((z_all, z_v), dim=1)
            f_all = torch.cat((f_all, f_temp), dim=1)
    kmeans_arch.fit_predict(z_all.cpu().detach().data.numpy())

    a_all = torch.tensor(kmeans_arch.cluster_centers_).cuda()
    c_allmin = calculate_c(z_all.T, a_all.T,gamma=args.gamma)
    model.cl_weight.data = c_allmin
    model.Al_weight.data =a_all.T
    torch.save(model.state_dict(), args.save_path)
"""
    f_temp = cacluate_U(torch.clamp(model.cl_weight.data, 0, 1))
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
    kmeans.fit_predict(f_temp.cpu().detach().data.numpy())
    y_pred = kmeans.labels_
    #y_pred = (np.argmax(qpred.cpu().detach().data.numpy(), axis=1))
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    print('z_all f concat fineTuning:Acc {:.4f}'.format(acc),
          ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))


    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
    kmeans.fit_predict(f_all.cpu().detach().data.numpy())
    y_pred = kmeans.labels_
    #y_pred = (np.argmax(qpred.cpu().detach().data.numpy(), axis=1))
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    print('f_all f concat fineTuning:Acc {:.4f}'.format(acc),
          ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))



"""



def mask_correlated_samples(N):
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(N//2):
        mask[i, N//2 + i] = 0
        mask[N//2 + i, i] = 0
    mask = mask.bool()
    return mask

def embeddingcontras(h_i ,h_j):
    loss_function=nn.CrossEntropyLoss(reduction="sum")
    N = 2 * args.batch_size
    h = torch.cat((h_i, h_j), dim=0)
    sim = torch.matmul(h, h.T) / 0.5
    sim_i_j = torch.diag(sim, args.batch_size)
    sim_j_i = torch.diag(sim, -args.batch_size)
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(N // 2):
        mask[i, N // 2 + i] = 0
        mask[N // 2 + i, i] = 0
    mask = mask.bool()
    negative_samples = sim[mask].reshape(N, -1)
    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    loss = loss_function(logits, labels)
    loss /= N
    return loss


def labelcontras( q_i, q_j):
    p_i = q_i.sum(0).view(-1)
    p_i /= p_i.sum()
    ne_i = torch.log(torch.tensor(p_i.size(0))) + (p_i * torch.log(p_i)).sum()
    p_j = q_j.sum(0).view(-1)
    p_j /= p_j.sum()
    ne_j = torch.log(torch.tensor(p_j.size(0))) + (p_j * torch.log(p_j)).sum()
    entropy = ne_i + ne_j

    q_i = q_i.t()
    q_j = q_j.t()
    N = 2 * args.n_clusters
    q = torch.cat((q_i, q_j), dim=0)

    sim = nn.CosineSimilarity(dim=2)(q.unsqueeze(1), q.unsqueeze(0)) /1
    sim_i_j = torch.diag(sim, args.n_clusters)
    sim_j_i = torch.diag(sim, -args.n_clusters)

    positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N)
    negative_clusters = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_clusters.device).long()
    logits = torch.cat((positive_clusters, negative_clusters), dim=1)
    loss = nn.CrossEntropyLoss(reduction="sum")(logits, labels)
    loss /= N


    return loss + entropy

def fineTuning():
    model = MultiViewModel(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.arch,
        pretrain=False,
        save_path=args.save_path).cuda()
    model.load_state_dict(torch.load(args.save_path))

    dataset = multiViewDataset2(args.dataset, args.viewNumber, args.method, True)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)
    # intalize graph gf,z_all
    for batch_idx, (x, y, idx) in enumerate(dataLoader):
        for viewIndex in range(args.viewNumber):
            x[viewIndex] = x[viewIndex].cuda()
        output = model(x)

        for viewIndex in range(args.viewNumber):

            z_temp = output[viewIndex][1]

            if (viewIndex) == 0:
                z_all = z_temp
            else:
                z_all = torch.cat((z_all, z_temp), dim=1)

    loss_function = nn.KLDivLoss(reduction='mean')


    print('Start Self-supervised Learning.')
    for epoch in tqdm.tqdm(range(1000)):
        qlist=list()

        for batch_idx, (x, y, idx) in enumerate(dataLoader):
            mseloss = 0.
            view_loss=0.
            kl_loss=0.
            for viewIndex in range(args.viewNumber):
                x[viewIndex] = x[viewIndex].cuda()
            output = model(x)


        for viewIndex in range(args.viewNumber):
            mseloss = mseloss + torch.nn.functional.mse_loss(output[viewIndex][2], x[viewIndex])
            z_temp=output[viewIndex][1]
            q_temp=output[viewIndex][4]
            qlist.append(q_temp)
            torch.max(q_temp)
            single_viewloss=torch.nn.functional.mse_loss(output[viewIndex][0], output[viewIndex][1])\
                      +args.gamma*torch.norm(model.aes[viewIndex].c_weight.data, p='fro')
            view_loss=view_loss+(single_viewloss/args.batch_size)

            if (viewIndex) == 0:
                z_all = z_temp

            else:
                z_all = torch.cat((z_all, z_temp), dim=1)

        if epoch==0:
            for viewIndex in range(args.viewNumber):
                c_temp = torch.clamp(model.aes[viewIndex].c_weight.data, 0, 1)
                f_temp = cacluate_U(c_temp)
                if (viewIndex) == 0:
                    f_all = f_temp
                else:
                    f_all = torch.cat((f_all, f_temp), dim=1)

            kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
            kmeans.fit_predict(f_all.cpu().detach().data.numpy())



            _, p = make_qp(torch.tensor(f_all).cuda(), kmeans.cluster_centers_)
            _,ind = torch.sort(p, dim=1)

        for viewIndex in range(args.viewNumber):
            kl_loss = kl_loss + loss_function(qlist[viewIndex].log(), p)
        view_loss=0.1*view_loss
        loss =1*mseloss+view_loss+kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch %500==0:
            print('mse_loss: {:.4f}'.format(mseloss),
                  ',kl_loss{:.4f}'.format(kl_loss))

    kmeans_arch = KMeans(n_clusters=args.arch, n_init=100)
    print('Start Contrastive Learning.')
    for epoch in tqdm.tqdm(range(100)):
        for batch_idx, (x, _, _) in enumerate(dataLoader):
            loss = 0.
            for viewIndex in range(args.viewNumber):
                x[viewIndex] = x[viewIndex].cuda()
            output = model(x)
            for viewIndex in range(args.viewNumber):
                loss = loss + F.mse_loss(output[viewIndex][2], x[viewIndex])
            for viewIndexa in range(args.viewNumber):
                for viewIndexb in range(viewIndexa):
                    #loss=loss+embeddingcontras(output[viewIndexa][1],output[viewIndexb][1])
                    loss = loss + labelcontras(output[viewIndexa][4], output[viewIndexb][4])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch %10==0:
            print(loss)

    for batch_idx, (x, y, _) in enumerate(dataLoader):
        y = y.data.cpu().numpy()
        for viewIndex in range(args.viewNumber):
            x[viewIndex] = x[viewIndex].cuda()
    output=model(x)
    qpred=0
    for viewIndex in range(args.viewNumber):
        z_temp = output[viewIndex][1]
        q=output[viewIndex][4]
        kmeans_arch.fit_predict(z_temp.cpu().detach().data.numpy())
        qpred += q
        a = torch.tensor(kmeans_arch.cluster_centers_).cuda()
        c_temp = calculate_c(z_temp.T,a.T,gamma=args.gamma)
        f_temp = cacluate_U(c_temp)
        if (viewIndex) == 0:
            z_all = z_temp
            f_all = f_temp

        else:
            z_all = torch.cat((z_all, z_temp), dim=1)
            f_all = torch.cat((f_all, f_temp), dim=1)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
    kmeans.fit_predict(f_all.cpu().detach().data.numpy())
    q_all,p_all= make_qp(f_all,kmeans.cluster_centers_)
    for i in range(q_all.shape[0]):
        q_all[i] = q_all[i][ind[i]]

    y_pred = kmeans.labels_
    # y_pred = (np.argmax(qpred.cpu().detach().data.numpy(), axis=1))
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    print('Acc {:.4f}'.format(acc),
          ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dataset', type=str, default='BDGP')
    parser.add_argument('--arch', type=int, default=50)
    parser.add_argument('--gamma', type=int, default=5)

    # parser.add_argument('--update_interval', default=1000, type=int)
    # parser.add_argument('--tol', default=0.0002, type=float)
    # parser.add_argument('--AR', default=0.95, type=float)
    args = parser.parse_args()
    # args.cuda =
    # print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.dataset = 'HW'
    args.method = 'HW'
    args.noise= 0
    args.arch= 50
    args.gamma= 1


    if args.dataset == 'BDGP':
        args.n_input = [1750, 79]
        args.viewNumber = 2
        args.instanceNumber = 2500
        args.batch_size = 2500
        args.n_clusters = 5
        args.save_path = './data/BDGP.pkl'
        args.noise=1
        args.arch=50
        args.gamma=10

    if args.dataset == 'HW':
        args.n_input = [216, 76, 64, 6, 240, 47]
        args.viewNumber = 6
        args.instanceNumber = 2000
        args.batch_size = 2000
        args.n_clusters = 10
        args.save_path = './data/HW.pkl'
        args.arch=50
        args.gamma=0.1

    if args.dataset == 'Fmnist-MV':
        args.n_input = [784, 784 , 784]
        args.viewNumber = 3
        args.instanceNumber =10000
        args.batch_size = 10000
        args.n_clusters = 10
        args.arch=50
        args.gamma=1
        args.save_path = './data/Fmnist-MV.pkl'
        setup_seed(200)





    if args.dataset == 'UCI-3V':
        args.n_input = [240, 76 , 64]
        args.viewNumber = 3
        args.instanceNumber = 2000
        args.batch_size = 2000
        args.n_clusters = 10
        args.save_path = './data/UCI_3V.pkl'
        args.arch=10
        args.gamma=1



    print(args)

    start =time()

    t0 = time()

    pretrain_aes()
    fineTuning()
    t1 = time()
    print(t1 - t0)
