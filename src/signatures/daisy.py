'''
Computes Daisy features
@author: romue
'''
import numpy as np
from skimage.feature import daisy
from skimage.transform import resize

def daisy_features(image, name, label, step_ = 4, rings_=3, histograms_=2, orientations_=8):
    
    if(image.shape[0] < 32 or image.shape[1] < 32): 
        image = resize(image, (32, 32))
    a = daisy(image,step = step_, rings = rings_, histograms = histograms_, orientations=orientations_)
    result = a.reshape(-1)
    result = np.asarray(result)
    return result,name,label

