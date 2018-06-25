import argparse
import os
import shutil

def parse_settings():

	# Training settings
	parser = argparse.ArgumentParser(description='Losses')
	print(parser)
	parser.add_argument('--k', type=int, default=2, help='number of clusters')
	parser.add_argument('--m', type=int, default=8, help='self cluster plus (m-1) hardest imposter clusters')
	parser.add_argument('--d', type=int, default=100, help='number of samplesfrom each cluster')
	parser.add_argument('--epochs', type=int, default=50,
	                    help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=1e-4,
	                    help='learning rate (default: 0.01)')
	parser.add_argument('--lda_loss', action='store_true', default=False,
						help='Enables the MagLDA loss for representation learning')
	parser.add_argument('--mnist', action='store_true', default=False,
						help='Use the mnist dataset')
	parser.add_argument('--cifar10', action='store_true', default=False,
						help='Use the CIFAR-10 dataset')
	parser.add_argument('--visdom', dest='visdom', action='store_true', default=False, help='Use visdom to track and plot')
	parser.add_argument('--name', default='MagLDA', type=str,
						help='name of experiment')
	return parser.parse_args()

def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = directory + filename
    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')
