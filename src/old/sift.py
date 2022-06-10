'''
Created on 4 de mar de 2016
Compute the SIFT features with or without mask
@author: romue
'''
import numpy as np
from skimage.feature import greycomatrix,greycoprops
from skimage.exposure import exposure
from skimage.color import rgb2gray
from skimage.io import imsave
import os
from subprocess import call
import warnings
from preprocessing import pp1
warnings.filterwarnings("ignore")
#def sift(imagem,name,label,d,grayLevels_new, grayLevels_old):
    
def sift(imagem, name,label):
    """ Process an image and save results in a file. """
   
    imsave('tmp.pgm',imagem)
    
    imageName = 'tmp.pgm'
    resultName = 'tmp.sift'
    
    cmmd = str("sift " + imageName + " --output " + resultName)
    
    os.system(cmmd)
    #call([cmmd])
    
    """ Read features and return matrix form. """
    f = np.loadtxt(resultName)
    
    os.remove(imageName) 
    os.remove(resultName) 
    
    # return feature locations (first 4 components) and descriptors (last 128 components)
    return f[:, 4:].reshape(-1),name,label


def teste():
    im = np.random.random((100,100,3))
    im,_,_ = pp1.preprocessing(im,0,0)
    a,_,_ = sift(im,'','')
    print(a)
    
#teste()

"""
How to install sift (Linux and MAC):
    $ wget http://www.vlfeat.org/download/vlfeat-0.9.20-bin.tar.gz
    $ tar zxvf <file.tar.gz>
    $ export PATH=<file path>/bin/glnxa64:$PATH
    $ sift (to test if it was correctly installed)
    
"""
