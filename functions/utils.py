import numpy as np
from matplotlib.patches import Ellipse
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from sklearn import mixture
import sys
from skimage.draw import circle
import dognet


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

def get_trainingImages(path2, scale):
    
    """
    Get and scale training images saved during image annotation
    """
    trainingImages=[]
    for i in path2:
        im = np.load(i)
        image1 = im/scale
        trainingImages.append(image1)
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

def calc_fitting(pts1, pts2, tau):
    """
    Calc how well the pts2 are related to pts1
    :param pts1: point ground truth
    :param pts2: point detected
    :param tau:  maximal distance between points
    :return: matched pairs
    """
    cost = np.zeros((pts1.shape[0], pts2.shape[0]))
    good1 = []
    good2 = []
    gt_res = []
    for index1, p1 in enumerate(pts1.astype(np.float32)):
        for index2, p2 in enumerate(pts2.astype(np.float32)):
            cost[index1, index2] = np.linalg.norm(p1 - p2)
            if cost[index1, index2] < tau and index1 not in good1 and index2 not in good2:
                good1.append(index1)
                good2.append(index2)
                gt_res.append((index1, index2))

    return gt_res

def get_metric(pts1, pts2, s=5.):
    """
    Calc how well the pts2 are related to pts1
    :param pts1: point ground truth
    :param pts2: point detected
    :param s: maximal distance between pair of points
    :return precision,recall,f1_score and point pairs
    """
    gt_res = calc_fitting(pts1, pts2, s)
    precision = float(len(gt_res)) / float(pts1.shape[0])
    recall = float(len(gt_res)) / float(pts2.shape[0])
    
    total_positive = float(pts1.shape[0])
    correct_positive = float(len(gt_res))
    if  0==(precision + recall):
        return 0,0,0,0,0,[]
    f1 = 2 * (precision * recall) / (precision + recall)
    return total_positive, correct_positive, precision, recall, f1, gt_res

def inference(net,image,get_inter = False):
    x = np.expand_dims(image,0)
    vx = torch.from_numpy(x).float().cpu()
    res,inter = net(vx)
    if get_inter:
        return res.data.cpu().numpy(),inter.data.cpu().numpy()
    return res.data.cpu().numpy()

def estimate_quality(collman,net,layer,slices=[2,3,4,5,6],th=0.4):
    mprecision=[]
    mrecall=[]
    mf1_score=[]
    auc_score=[]
    dic=[]
    
    for s in slices:
        y  = inference(net,collman[:,s])
        #plt.imshow(y[0].mean(axis=0))
        
        if len(layer[s])==0:
            y_gt = np.zeros((collman[:,0]).shape[1:])
        else:
            y_gt = make_labels(collman[:,0],np.array(layer[s])[:,0],np.array(layer[s])[:,1])

        fpr, tpr, thresholds = roc_curve( y_gt.flatten(),y[0,0].flatten())
        auc_score.append(auc(fpr, tpr))
            
        #if len(layer[s])==0:
            #gt_pts = np.array([])
        #else:
            #gt_pts = np.array([np.array(layer[s])[:,1],np.array(layer[s])[:,0]]).transpose(1,0)

        #coords = np.array([ list(p.centroid) for p in regionprops(label(y[0,0]>th)) if p.area>4])

        #if coords.size == 0:
            #print("empty")
            #dog_pts = np.array()
        #else:
            #dog_pts = np.array([coords[:,1],coords[:,0]]).transpose(1,0)
        
        #total_positive,correct_positive,precision,recall,f1_score,_ = get_metric(gt_pts,dog_pts,s=5.)
        
        #mprecision.append(precision)
        #mrecall.append(recall)
        #mf1_score.append(f1_score)
        #dic.append(abs(float(gt_pts.shape[0])-float(dog_pts.shape[0])))
        
   #with open("../datasets/flocculusA/totalf13.txt", "a") as fin:
        #fin.write(str(f1_score))
        #fin.write("\n")
            
    #with open("../datasets/flocculusA/totalp3.txt", "a") as fin:
        #fin.write(str(precision))
        #fin.write("\n")
            
   # with open("../datasets/flocculusA/totalr3.txt", "a") as fin:
        #fin.write(str(recall))
        #fin.write("\n")
           
    return y[0].mean(axis=0)#,total_positive,correct_positive,np.mean(mf1_score),np.mean(mprecision),np.mean(mrecall),np.mean(auc_score),np.mean(dic)

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

def show_prediction(x,testingNumber,trainingImages,trainingMasks,collman,net,b):
    """
    Show original image, image mask, and prediciton
    """
    plt.figure(figsize=(20,10))
    for j in range(0,testingNumber):
        plt.subplot(testingNumber,3,3*j+1)
        plt.imshow(trainingImages[x+j].mean(axis=0))
        plt.subplot(testingNumber,3,3*j+2)
        plt.imshow(trainingMasks[x+j])
        plt.subplot(testingNumber,3,3*j+3)
        plt.imshow(estimate_quality(collman,net,b,slices=range(x+j,x+j+1)))