import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.linalg.decomp import eigh

a = 1
b = 0
c = 1

class LDALoss(nn.Module):
    """
    MagLDA loss:
    Args:
        r: A batch of features.
        classes: Class labels for each example.
        true_classes: True Class labels for each example.
        clusters: Cluster labels for each example.
        cluster_classes: Class label for each cluster.
        n_clusters: Total number of clusters.

    Returns:
        cost: The mean MagLDA loss for the batch.
    """
    def __init__(self, n_components=None):
        super(LDALoss, self).__init__()
        self.r = None
        self.classes = None
        self.true_classes = None
        self.clusters = None
        self.cluster_classes = None
        self.n_clusters = None
        self.scalings_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_components = n_components

    def forward(self, r, classes, true_classes, m, d, classes_mean):

        def _compute_scatter_subclass_within(cluster_means, cluster_examples):
            nfeatures = self.r.size()[-1]
            cluster_means = cluster_means #(num_clusters, emb_dim)
            classes_mean = self.classes_mean #(num_classes, emd_dim)
            N = self.r.size()[0]
            SSW1 = Variable(torch.zeros(nfeatures, nfeatures).type(GPU_FLOAT_DTYPE))
            SSW2 = Variable(torch.zeros(nfeatures, nfeatures).type(GPU_FLOAT_DTYPE))
            for i, x in enumerate(cluster_examples):
                c = self.true_cluster_classes[i] #has to be true class
                cls_mean = torch.from_numpy(classes_mean[c])
                cls_mean = Variable(cls_mean.type(GPU_FLOAT_DTYPE)) 
                cls_mean_ = cls_mean.resize(nfeatures, 1)
                cluster_mean_ = cluster_means[i].resize(nfeatures, 1)
                SSW2 += torch.mm((cls_mean_ - cluster_mean_),(cls_mean_ - cluster_mean_).t())
                for row in x:
                    row_ = row.resize(nfeatures, 1)
                    SSW1 += torch.mm((row_ - cluster_mean_), (row_ - cluster_mean_).t())
            Sw = a * SSW1 + b * SSW2
            return Sw
        
        def _compute_scatter_subclass_between(cluster_means):
            nfeatures = self.r.size()[-1]
            Sb = Variable(torch.zeros(nfeatures, nfeatures).type(GPU_FLOAT_DTYPE))
            n_features = self.r.size()[-1]
            cluster_means = cluster_means
            for i in xrange(cluster_means.size()[0]-1): 
                for j in xrange(i+1, cluster_means.size()[0]): 
                    if self.cluster_classes[i] != self.cluster_classes[j]:
                        Sb += c * torch.mm((cluster_means[i].resize(nfeatures,1) - cluster_means[j].resize(nfeatures,1)), (cluster_means[i].resize(nfeatures,1) - cluster_means[j].resize(nfeatures,1)).t())
            return Sb


        GPU_INT_DTYPE = torch.cuda.IntTensor
        GPU_LONG_DTYPE = torch.cuda.LongTensor
        GPU_FLOAT_DTYPE = torch.cuda.FloatTensor
        self.r = r
        #print self.r [torch.cuda.FloatTensor of size 64x2 (GPU 0)]
        self.classes_mean = classes_mean
        self.classes = torch.from_numpy(classes).type(GPU_LONG_DTYPE)
        self.true_classes = torch.from_numpy(true_classes).type(GPU_LONG_DTYPE)
        self.clusters, _ = torch.sort(torch.arange(0, float(m)).repeat(d))
        self.clusters = self.clusters.type(GPU_INT_DTYPE)
        self.cluster_classes = self.classes[0:m*d:d]
        self.true_cluster_classes = self.true_classes[0:m*d:d]
        #print ('(Loss) self.cluster_classes: ', self.cluster_classes)
        print ('(Loss) self.true_cluster_classes: ', self.true_cluster_classes)
        self.n_clusters = m

        # Take cluster means within the batch
        cluster_examples = dynamic_partition(self.r, self.n_clusters)
        

        cluster_means = torch.stack([torch.mean(x, dim=0) for x in cluster_examples])
        #print cluster_means: (self.n_clusters, 2)
    
        Sw = _compute_scatter_subclass_within(cluster_means, cluster_examples)
        Sw+=Variable(torch.eye(Sw.size()[0]).type(GPU_FLOAT_DTYPE),requires_grad=False)*1e-3
        Sb = _compute_scatter_subclass_between(cluster_means)

        #evals, evecs = eigh(Sb.data.cpu().numpy(), Sw.data.cpu().numpy())
        #evals = torch.from_numpy(evals).type(GPU_FLOAT_DTYPE)
        #evecs = torch.from_numpy(evecs).type(GPU_FLOAT_DTYPE)
        #evals, evecs = torch.eig(torch.inverse(Sw)*Sb, eigenvectors=True)
        #evals = evals[:,0]
        #evals_sorted, sorted_ind = torch.sort(evals)
        #evecs_sorted = evecs[:, sorted_ind]
        #print evals_sorted
        #print evecs_sorted

        U, S, V  = torch.svd(torch.inverse(Sw)*Sb)
        
        print S
        top_k_evals = S[:self.n_components]
        print top_k_evals
        print top_k_evals.max() / top_k_evals.min()

        #maximize variance between classes <-> (k smallest eigenvalues below threshold)
        #thresh = torch.min(top_k_evals) + 10.0
        thresh = torch.mean(top_k_evals) 

         
        top_k_evals_ = top_k_evals[(top_k_evals <= thresh)]
        cost = -torch.mean(top_k_evals_)

        return cost


def dynamic_partition(X, n_clusters):
    """Partitions the data into the number of cluster bins"""
    cluster_bin = torch.chunk(X, n_clusters)
    return cluster_bin





