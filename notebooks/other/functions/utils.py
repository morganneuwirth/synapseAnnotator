import numpy as np
from matplotlib.patches import Ellipse
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from sklearn import mixture
import sys
from skimage.draw import circle

import matplotlib
import matplotlib.pyplot as plt
import os
from oiffile import imread
from PIL import Image
import random
import skimage
from skimage.transform import resize
import glob
import torch
import sys
sys.path.insert(0, '..')
import scipy.io as sio
import matplotlib.patches as patches
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.transform import rescale
from sklearn.metrics import roc_curve, auc

def get_trainingMasks(path1):
    """
    Get saved training masks generated during image annotation
    """
    trainingMasks=[]
    for i in path1:
        im = np.load(i)
        trainingMasks.append(im)
        
    return trainingMasks

def get_trainingImages(path2, zeroScale, oneScale, twoScale):
    
    """
    Get and scale training images saved during image annotation
    """
    trainingImages=[]
    for i in path2:
        im = np.load(i)
        im[0,:,:] = im[0,:,:]/zeroScale
        im[1,:,:] = im[1,:,:]/oneScale
        im[2,:,:] = im[2,:,:]/twoScale
        trainingImages.append(im)
    new_images = np.stack(trainingImages, axis = 3)
    collman = np.transpose(new_images,(0,3,1,2))
    
    return trainingImages, collman

def make_labels(img,ys,xs,radius=4):
    labels = np.zeros(img.shape[1:])
    if len(ys)==0:
        return labels
    for xv,yv in zip(xs,ys):
        rr,cc = circle(xv,yv,radius,labels.shape)
        
        labels[rr,cc]=1
    return labels

def make_labels_gaussian(img,ys,xs,radius=4):
    labels = np.zeros(img.shape[1:])
    if len(ys)==0:
        return labels
    
    for xv,yv in zip(xs,ys):
        if xv+11>230:
            xin=230-xv
            xin=int(math.ceil(xin))
        else:
            xin=11
        if xv<10:
            xde=xv
            xde=int(math.floor(xde))
        else:
            xde=10
        if yv+11>230:
            yin=230-yv
            yin=int(math.ceil(yin))
        else:
            yin=11
        if yv<10:
            yde=yv
            yde=int(math.floor(yde))
        else:
            yde=10
            
        gimg = np.zeros((21,21))
        
        gimg[10][10]=1
        gaussimg = gaussian_filter(gimg,4)
        gaussimg=gaussimg[(10-xde):(10+xin),(10-yde):(10+yin)]
        gaussimg.astype(int)
        labels[int(xv-xde):int(xv+xin),int(yv-yde):int(yv+yin)] = np.maximum(labels[int(xv-xde):int(xv+xin),int(yv-yde):int(yv+yin)],gaussimg) 
    labels = labels/(np.amax(labels))
    return labels

def make_training_set(labels,indexes):
    train_images = []
    train_labels = []
    for i in indexes:
        if len(labels[i])==0:
            d = np.zeros_like(collman[0,0,:,:])
        else:
            d = make_labels(collman[:,0],np.array(labels[i])[:,0],np.array(labels[i])[:,1])
        train_images.append(collman[:,i])
        train_labels.append(d)
    return train_images,train_labels

def inference(net,image,get_inter = False,device = torch.device("cpu")):
    x = np.expand_dims(image,0)
    vx = torch.from_numpy(x).float()
    cvx = vx.to(device)
    res, inter = net(cvx)
    if get_inter:
        return res.data.cpu().numpy(),inter.data.cpu().numpy()
    return res.data[0].cpu().numpy()

def get_coords(x,testingNumber,len_trainingImages):
    
    """
    Get coordinates of testing images
    """
    
    b = [[] for j in range(0,len_trainingImages)]
    
    for q in range(x,x+testingNumber):
        num = int(q/4)
        if q%4 == 0:
            quad = "LL"
        elif q%4 == 1:
            quad = "LR"
        elif q%4 == 2:
            quad = "UL"
        elif q%4 == 3:
            quad = "UR"

        lines = []   
        with open("../datasets/flocculusA/imagesUsed.txt") as fin:
            lines = fin.readlines()
        lines.sort()
        fname = lines[num]
        fname = fname[:12]

        imageName = "../datasets/flocculusA/trainingCoordinates/%s_%s_coords" % (fname,quad)
        with open(imageName) as fi:
            coors = fi.readlines()

        coordins = [[] for j in range(0,len(coors))]

        for i in range(0,len(coors)):
            xy = coors[i].split(",")
            xy[1]=xy[1][:24]
            xy[0]=float(xy[0])
            xy[1]=float(xy[1])
            coordins[i]=np.array(xy)
            
        b[q]=coordins
    
    return b
