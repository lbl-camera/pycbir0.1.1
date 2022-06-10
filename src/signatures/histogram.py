'''
Created on 14 de mar de 2016

@author: romuere

Script to compute 1a order texture features
Entropy, energy, mean, variance, skewness, kurtosis, roughness
'''

import numpy as np
from scipy.stats.mstats_basic import skew, kurtosis
import copy

def histogram(imagem,name, label):
    
    imagem = imagem.astype(np.uint32)
    texture = np.zeros((7))
    im_mask = imagem
    t = im_mask.shape[0]*im_mask.shape[1]
    
    hist,bin = np.histogram(im_mask,density=False,bins = 256) #compute the histogram
    hist = np.float32(hist)
    hist = hist/t
    hist_temp = copy.copy(hist)#copy of hist 
    hist_temp[hist_temp==0] = 1 #replaces 0 by 1 -> log2(1) == 0

    ##Histogram Attributes
    #Entropy
    Entropy = -1*np.sum(hist_temp*np.log2(hist_temp))
    #print Entropy

    #Energy
    Energy = np.sum((hist/t)**2)
    #print Energy
    
    #Mean
    Mean = np.mean(imagem)
    #print Mean
    
    #Variance
    Variance = np.var(hist)
    #print Variance
    
    #Skewness
    Skewness = skew(hist)
    Skewness = Skewness.tolist()
    #print Skewness
    
    #Kurtosis
    Kurtosis = kurtosis(hist)
    Kurtosis = Kurtosis.tolist() 
    #print Kurtosis
    
    #Roughness
    R = 1 - (1/(1+Variance))
    #print R
    texture[0] = Entropy 
    texture[1] = Energy
    texture[2] = Mean
    texture[3] = Variance
    texture[4] = Skewness
    texture[5] = Kurtosis
    texture[6] = R
    
    return texture,name,label