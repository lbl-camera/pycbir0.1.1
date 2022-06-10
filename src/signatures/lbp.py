import numpy as np
from skimage.feature import local_binary_pattern as lbp

def lbpTexture (im, name, label, neighbors=2, radio=16):
    '''
    radio = 2
    neighbors = 8 * radio
    '''
    
    lbp_image = lbp(im,P = neighbors, R = radio)
    t = im.shape[0]
    hist,bin = np.histogram(lbp_image,density=False,bins = 256) #compute the histogram
    hist = np.float32(hist)
    hist = hist/t
    return hist,name,label