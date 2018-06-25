import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.init as init
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
import torchvision.models as models
from models.lenet import LeNet
from lda_loss.lda_tools import *
from lda_loss.lda_loss import LDALoss
from lda_loss.utils import plot_embedding, plot_smooth
from utils.train_settings import parse_settings
from utils.sampler import SubsetSequentialSampler
from utils.average_meter import AverageMeter
from visualizer.visualizer import VisdomLinePlotter
from datasets.load_dataset import load_dataset

args = parse_settings()

def run_lda_loss(args):
    m = args.m
    d = args.d
    k = args.k

    global plotter
    plotter = VisdomLinePlotter(env_name=args.name)

    trainloader, testloader, trainset, testset, n_train = load_dataset(args)

    emb_dim = 128
    n_epochs = 15
    epoch_steps = len(trainloader)
    print ('epoch_steps: ', epoch_steps)
    n_steps = epoch_steps * 15
    #Refreshing clueters after each epoch
    cluster_refresh_interval = epoch_steps

    if args.mnist:
        model = (LeNet(emb_dim)).cuda()
    if args.cifar10:
        model = (VGG(depth=16, num_classes=emb_dim))
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    minibatch_maglda_loss = LDALoss(n_components=9)

    images = getattr(trainset, 'train_data')
    #print images [torch.ByteTensor of size 60000x28x28]
    labels = getattr(trainset, 'train_labels')
    #print labels [torch.LongTensor of size 60000]

    # Get initial embedding
    initial_reps, classes_mean = compute_reps(model, trainset, labels, 400)
    #print initial_reps.shape

    if args.cifar10:
        labels = np.array(labels, dtype=np.float32)

    # Create batcher
    batch_builder = ClusterBatchBuilder(labels, k, m, d)
    batch_builder.update_clusters(initial_reps) #inital centroids, self.assignments (each example belongs to which cluster) and self.cluster_assignments (each cluster (key) corresponds to which entries (value)) have been computed. 

    batch_losses = []
        
    """batch_example_inds: Batch size of mxd, comes from 1 self.cluster and m-1 nearest imposter clusters. for each cluster m, generate the number of d entries and return batch_example_inds. 
        batch_class_inds: For each example, generate its class index (but this class index is not the true class index, only guarantee that different class has different class inds. e.g. d=3, mxd examples [22,22,22,30,30,30,32,32,32,1,1,1,11,11,11] -> cluster index [22, 30, 32, 1, 11] -> true class index[2, 4, 4, 0, 1] -> generated class index [0, 1, 1, 2, 3] -> [0,0,0,1,1,1,1,1,1,2,2,2,3,3,3]) -> batch_class_inds. Because only different classes are considered during losses computation, it doesn't consider which true class it is. """
    batch_example_inds, batch_class_inds, batch_class_true_inds = batch_builder.gen_batch()
    print len(batch_example_inds)
    trainloader.sampler.batch_indices = batch_example_inds #reassign the self.batch_indices for SubsetSequentialSampler.
    _ = model.train()
    losses = AverageMeter()

    for i in xrange(n_steps):
        #trainloader is controlled by a subset sampler -> only generate one batch
        for (img, target) in trainloader:
            img = Variable(img).cuda() #sampled by mxd
            target = Variable(target).cuda()
            optimizer.zero_grad()
            output, features = model(img)
	    batch_loss  = minibatch_maglda_loss(
                           output,			
                           batch_class_inds, 
                           batch_class_true_inds,
			   m, d, classes_mean)
            batch_loss.backward()
            optimizer.step()

            # Update loss index
            #batch_builder.update_losses(batch_example_inds, batch_example_losses) #when self.example_losses (60000, ) and cluster_losses (80, ) is None, create them. Then update self.example_losses and cluster_losses based on the batch (mxd) feedback.

	    batch_losses.append(batch_loss.data[0])

            print ('iteration: %i, batch_loss: %.3f' % (i, batch_loss))

	    if i % cluster_refresh_interval == 0:
	        print("Refreshing Clusters and Classes Mean")
                #refresh k-means and classes mean in new feature space
		reps, classes_mean = compute_reps(model, trainset, labels, 400)
                #refresh centroids, self.assignments and self.cluster_assignments.
		batch_builder.update_clusters(reps)

	    #if i % 2000 == 0:
            #    print ("Plot Embedding")
            #	 n_plot = 500
	    #	plot_embedding(compute_reps(model, trainset, 400)[:n_plot], labels[:n_plot], name=i)

            batch_example_inds, batch_class_inds, batch_class_true_inds = batch_builder.gen_batch()
            trainloader.sampler.batch_indices = batch_example_inds

            losses.update(batch_loss, 1)

            # Log the average training loss history
            if args.visdom:
	        plotter.plot('loss', 'train', i, losses.avg.data[0])

    # Plot loss curve
    plot_smooth(batch_losses, "batch-losses")

if __name__ == '__main__':
    run_lda_loss(args)
