'''
Created on 05 de out de 2016

@author: romuere
'''

import numpy as np
from skimage.color import rgb2gray

'''
Computes de Haar Havelet and returns four decompositions as a single image
'''

def preprocessing(im, name, label, nLevels):
    
    wp = pywt.WaveletPacket2D(data = rgb2gray(im),wavelet = 'haar')
    CA = reescale(wp['a'*nLevels].data)
    CH = reescale(wp['h'*nLevels].data)
    CV = reescale(wp['v'*nLevels].data)
    CD = reescale(wp['d'*nLevels].data)
    
    im1 = np.concatenate((CA,CH),axis=1)
    im2 = np.concatenate((CV,CD),axis=1)
    
    dec = np.concatenate((im1,im2),axis=0)
    
    im3 = np.zeros((im.shape[0],im.shape[1],4))
    #im = np.concatenate((im1,im2),axis=0)    
    
    im3[:,:,0:3] = im
    im3[:,:,3] = dec 
    
    return (np.uint8(im3),name,label)

def reescale(im):
    return (im-im.min())*255/(im.max()-im.min())    
