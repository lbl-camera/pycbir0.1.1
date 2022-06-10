'''
Created on 30 de jun de 2016

@author: romuere
'''
from skimage.feature import hog

def HOG(imagem):
    
    
    size = imagem.shape
    #h = hog(imagem,orientations = 8,pixels_per_cell = (size[0],size[1]),cells_per_block = (1,1))
    h = hog(imagem,orientations=1,pixels_per_cell = (size[0],size[1]))
    
    return h