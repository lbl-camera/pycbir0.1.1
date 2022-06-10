'''
Created on 05 de out de 2016

@author: romuere
'''

import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist

'''
This preprocessing do a transformation to grayscale and after an histogram equalization
'''

def preprocessing(im, name, label):

    if len(im.shape) > 2:
        im = rgb2gray(im)

    im = (equalize_hist(im)*255).astype(np.uint8)
    return (im, name, label)
