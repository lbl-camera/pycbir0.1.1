'''
Created on 05 de out de 2016

@author: romuere
'''

import numpy as np
from skimage.color import rgb2gray
#import pywt

'''
Computes de Haar Havelet and returns four decompositions as a single image
'''

def preprocessing(im, name, label, nLevels):
      
    wp = pywt.WaveletPacket2D(data = rgb2gray(im),wavelet = 'haar')
    CA = reescale(wp['a'*nLevels].data)
    CH = reescale(wp['h'*nLevels].data)
    CV = reescale(wp['v'*nLevels].data)
    CD = reescale(wp['d'*nLevels].data)
    
    im = np.zeros((CA.shape[0],CA.shape[1],4))
    im[:,:,0] = CA    
    im[:,:,1] = CH
    im[:,:,2] = CV
    im[:,:,3] = CD
    
    return (np.uint8(im),name,label)

def reescale(im):
    return (im-im.min())*255/(im.max()-im.min())    

'''
def preprocessing(im):

    im_8 = im
    a = (im_8 == 0)
    im_8[a] = 1
    im_log = np.log(im_8)

    #uint8
    im_log = np.uint8(((im_log)*255)/np.max(im_log))

    return im_log
'''