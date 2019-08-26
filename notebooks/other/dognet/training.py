import sys
from skimage.transform import resize
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tnrange
import random


def create_generator(data, labels, device = torch.device("cpu"), s=0 , size=(460, 460),batch_size= 10,transform=False):
    """
    Generates training patches
    :param data: list of images
    :param labels: list of labels
    :return: function generator
    """
    def f(index = np.arange(batch_size), batch_size=batch_size, size=size, s=s,transform = transform):
        d = np.zeros((batch_size, data[0].shape[0], int(size[0]/2), int(size[1]/2)))
        l = np.zeros((batch_size, 2, int(size[0]/2), int(size[1]/2)))
        if transform:
            rot = np.random.randint(0, 1, batch_size)
            flip = np.random.randint(0, 1, batch_size)
            intensity = np.random.uniform(0.5,1.5,batch_size)
        else:
            rot = np.zeros(batch_size)
            flip = np.zeros(batch_size)
            intensity = np.ones(batch_size)
        for i in range(batch_size):
            dd = np.zeros((3,int(size[0]/2), int(size[1]/2)))
            for k in range(3):
                dd[k] = intensity[i]*resize(data[index[i]][k],(size[0]/2,size[1]/2), order=1, preserve_range=True)
            ll = resize(labels[index[i]],(size[0]/2,size[0]/2), order=1, preserve_range=True)

            if rot[i] > 0:
                dd = np.rot90(dd)
                ll = np.rot90(ll)
            if flip[i] > 0:
                dd = np.flipud(dd)
                ll = np.flipud(ll)
            d[i] = dd
            l[i, 0] = ll
            l[i, 1] = 1-ll
            
        d, l = torch.from_numpy(d).float().to(device), torch.from_numpy(l).float().to(device)
        return d, l

    return f

def create_test_generator(data, labels, device = torch.device("cpu"), s=0 , size=(460, 460)):
    n = len(data)
    
    def f(index = np.arange(n), n=n, size=size, s=s):
        d = np.zeros((n, data[0].shape[0], int(size[0]/2), int(size[1]/2)))
        l = np.zeros((n, 2, int(size[0]/2), int(size[1]/2)))
        for i in range(n):
            for k in range(3):
                d[i,k] = resize(data[i][k],(int(size[0]/2), int(size[1]/2)), order=1, preserve_range=True)
            l[i,0] = resize(labels[i],(int(size[0]/2), int(size[1]/2)), order=1, preserve_range=True)
            l[i,1] = 1-resize(labels[i],(int(size[0]/2), int(size[1]/2)), order=1, preserve_range=True)

        d, l = torch.from_numpy(d).float().to(device), torch.from_numpy(l).float().to(device)
        return d, l
    return f
    
def create_generator_3d(data, labels, s=0 , size=(64, 64), n=10, depth=1):
    """
    Generates training patches
    :param data: list of images
    :param labels: list of labels
    :return: function generator
    """

    def f(n=n, size=size, s=s):

        d = np.zeros((n, data[0].shape[0],1+2*depth, size[0], size[1]))
        l = np.zeros((n, 1, size[0] - 2 * s, size[1] - 2 * s))
        rot = np.random.randint(0, 1, n)
        flip = np.random.randint(0, 1, n)
        for i in range(n):
            index = np.random.randint(depth, len(data)-depth)
            x = np.random.randint(0, data[index].shape[1] - size[0])
            y = np.random.randint(0, data[index].shape[2] - size[1])
            dd = np.stack([data[a][:, x:x + size[0], y:y + size[1]].copy() for a in range(index-depth,index+depth+1) ],1) 
            ll = labels[index][x + s:x + size[0] - s, y + s:y + size[1] - s].copy()

            if rot[i] > 0:
                dd = np.rot90(dd)
                ll = np.rot90(ll)
            if flip[i] > 0:
                dd = np.flipud(dd)
                ll = np.flipud(ll)
            d[i] = dd
            l[i, 0] = ll
        return d, l

    return f

def update_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def print_percent(percent):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('=' * percent, 5 * percent))
    sys.stdout.flush()


class focal_loss():
    def __init__(self,alpha=0.9,gamma = 5):
        self.alpha = alpha
        self.gamma = gamma
        
    def loss(self,y_pred,y_tar):
        epsilon = 1e-5
        y_pred_clipped = torch.clamp(y_pred,epsilon,1.0-epsilon)
        y_pred_t = y_tar*y_pred_clipped + (1-y_tar)*(1-y_pred_clipped)
        alpha_t = 2*(self.alpha*y_tar+(1-self.alpha)*(1-y_tar))
        fl = -torch.mean(alpha_t*((1-y_pred_t)**self.gamma)*torch.log(y_pred_t))
       
        return fl

def soft_dice_loss(y_pred, y_true, epsilon=1e-6):
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_pred: b x N x X x Y Network output, must sum to 1 over c channel (such as after softmax) 
        y_true: b x N x X x Y  One hot encoding of ground truth       
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

    # skip the batch and class axis for calculating Dice score

    numerator = 2. * torch.sum(y_pred * y_true)
    denominator = torch.sum(y_pred.pow(2) + y_true)
    
    return 1 - torch.mean(numerator / (denominator + epsilon))  # average over classes and batch

def create_weight(pos_weight):
    def weighted_binary_cross_entropy(sigmoid_x, targets, size_average=True, reduce=True):
        """
        Args:
            sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
            targets: true value, one-hot-like vector of size [N,C]
            pos_weight: Weight for postive sample
        """
        if not (targets.size() == sigmoid_x.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

        loss = -  pos_weight*targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()



        if not reduce:
            return loss
        elif size_average:
            return loss.mean()
        else:
            return loss.sum()
    return weighted_binary_cross_entropy

def train_routine(detector,
                  generator,
                  testgenerator, #
                  n_train_samples,
                  n_test_samples,
                  n_epoch=5000,
                  batch_size=10,
                  loss="focal",
                  lr=0.01,
                  alpha = 0.5,
                  gamma = 1,
                  device = torch.device("cpu"),
                  margin=10,
                  decay_schedule = [50,0.99],
                  optimizer=None,
                  regk=0.,
                  verbose=False):
    """
    Train a detector with respect to the data from generator
    :param detector: A detector network
    :param generator: A generator
    :param n_iter: number of iterations
    :param loss: loss function
    :param lr: starting learning rate
    :param margin: margin for ignore
    :param decay_schedule: pair of a which iteration and how should we decay
    :param use_gpu: if system has cuda support, the training can be run on GPU
    :return: trained network, errors
    """
    
    if optimizer is None:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, detector.parameters()), lr=lr)
    
    _, y = generator()
    sumy = y[0,0].sum()
    if 0==sumy:
        sumy=1
    ky = torch.FloatTensor([(y.shape[-1]*y.shape[-2]-y.sum()/y.shape[0])/sumy*y.shape[0]]).to(device)
    
    floss = soft_dice_loss
    if loss=="bce":
        floss = nn.BCELoss()
    elif loss== "L2":
        floss = nn.MSELoss()
    elif loss=="softdice":
        floss = soft_dice_loss
    elif loss=="weightbce":
        floss = create_weight(ky)
    elif loss=="focal":
        lossfunc = focal_loss(alpha,gamma)
        floss = lossfunc.loss
    if verbose:
        print(ky,y.shape,y.shape[-1]*y.shape[-2],y[0,0].sum(),y[0,0].max())
        print(floss) 

    if verbose:
        print("Training started!")
    percent_old = 0
    
    train_err = []
    test_err = []
    visual_test = torch.empty(n_epoch, 230, 230).cpu()
    
    detector.eval()
    #train error calculation
    randidx = np.random.permutation(n_train_samples)
    randidx = np.append(randidx,randidx[:(batch_size-np.mod(n_train_samples,batch_size))]);
    randidx = randidx.reshape(batch_size,-1)
    curr_train_err = 0
    for batch in range(randidx.shape[0]):
        batch_idx = randidx[batch]
        x_train, y_train = generator(index = batch_idx.astype(int), batch_size = batch_size)
        y_prediction_train, _ = detector(x_train)
        train_loss = floss(y_prediction_train, y_train)
        curr_train_err+=train_loss.item()
    #test error calculation
    randidx = np.random.permutation(n_test_samples)
    randidx = np.append(randidx,randidx[:(batch_size-np.mod(n_test_samples,batch_size))]);
    randidx = randidx.reshape(batch_size,-1)
    curr_test_err = 0
    for batch in range(randidx.shape[0]):
        batch_idx = randidx[batch]
        x_test, y_test = testgenerator(index = batch_idx.astype(int), batch_size = batch_size)
        y_prediction_test, _ = detector(x_test)
        test_loss = floss(y_prediction_test, y_test)
        curr_test_err+=test_loss.item()
    # track history
    train_err.append(curr_train_err/n_train_samples)
    test_err.append(curr_test_err/n_test_samples)
        
    
    for epoch in tnrange(n_epoch):
        randidx = np.random.permutation(n_train_samples)
        randidx = np.append(randidx,randidx[:(batch_size-np.mod(n_train_samples,batch_size))]);
        randidx = randidx.reshape(batch_size,-1)
        # training the network 
        detector.train()
        curr_train_err = 0
        for batch in range(randidx.shape[0]):
            batch_idx = randidx[batch]
            #train error calculation
            x_train, y_train = generator(index = batch_idx.astype(int), batch_size = batch_size)
            optimizer.zero_grad()
            y_prediction_train, _ = detector(x_train)
            train_loss = floss(y_prediction_train, y_train)
            curr_train_err+=train_loss.item()
            # update weights
            train_loss.backward()
            optimizer.step()
        
        detector.eval()
        #test error calculation
        randidx = np.arange(0,n_test_samples)
        randidx = np.append(randidx,randidx[:(batch_size-np.mod(n_test_samples,batch_size))]);
        randidx = randidx.reshape(batch_size,-1)
        curr_test_err = 0
        for batch in range(randidx.shape[0]):
            batch_idx = randidx[batch]
            x_test, y_test = testgenerator(index = batch_idx.astype(int), batch_size = batch_size)
            y_prediction_test, _ = detector(x_test)
            
            if batch == 0:
                y_history = y_prediction_test[0][0].detach().clone().cpu()
                visual_test[epoch, :, :]=y_history
            
            test_loss = floss(y_prediction_test, y_test)
            curr_test_err+=test_loss.item()
        # track history
        
        train_err.append(curr_train_err/n_train_samples)
        test_err.append(curr_test_err/n_test_samples)

        if 0 == (epoch + 1) % decay_schedule[0]:
            lr = lr * decay_schedule[1]
            update_rate(optimizer, lr)

    print("\nTraining finished!")
    return detector, train_err, test_err, visual_test
