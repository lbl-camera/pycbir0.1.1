'''
Created on 4 de mar de 2016
Compute the GLCM features with or without mask
@author: romue
'''
import numpy as np
from skimage.feature import greycomatrix,greycoprops

def glcm(imagem,name,label,d,grayLevels_new, grayLevels_old):

    if grayLevels_new != grayLevels_old:
        imagem = categorizar(imagem,grayLevels_new,grayLevels_old)
    matrix0 = greycomatrix(imagem, [d], [0], levels=2**grayLevels_new,normed=True)
    matrix1 = greycomatrix(imagem, [d], [np.pi/4], levels=2**grayLevels_new,normed=True)
    matrix2 = greycomatrix(imagem, [d], [np.pi/2], levels=2**grayLevels_new,normed=True)
    matrix3 = greycomatrix(imagem, [d], [3*np.pi/4], levels=2**grayLevels_new,normed=True)
    matrix = (matrix0+matrix1+matrix2+matrix3)/4 #isotropic glcm

    props = np.zeros((6))
    props[0] = greycoprops(matrix,'contrast')
    props[1] = greycoprops(matrix,'dissimilarity')
    props[2] = greycoprops(matrix,'homogeneity')
    props[3] = greycoprops(matrix,'energy')
    props[4] = greycoprops(matrix,'correlation')
    props[5] = greycoprops(matrix,'ASM')
    return props,name,label

#function to change the number of gray scale values
"""
def categorizar(imagem,nbits=8):
    L,C = imagem.shape;
    limites = np.arange(0,256,256/nbits)
    for z in range(0,len(limites)-1):
        aux = ((imagem >= limites[z]) & (imagem < limites[z+1]))
        imagem[aux==True] = z
    aux = (imagem >= limites[nbits-1])
    imagem[aux==True] = nbits-1
    return imagem
"""

def categorizar(image, new, old):
    L,C = image.shape;
    image = np.array(image,dtype = np.float64)
    for i in range(L):
        for j in range(C):
            image[i,j] = (((2**new)-1)*image[i,j])/((2**old)-1)
    image = np.array(image,dtype = np.int)
    return image


def teste():
    from skimage.io import imread_collection
    import pp2
    im = imread_collection('/Users/romuere/Desktop/als/kyager_data_raw/2011Jan28-BrentCarey/*')
    for i in im:
        img = pp2.preprocessing(i, '', 0)[0]
        features = glcm(img, '', 0, 1, 8,8)[0]
        #for f in features:
        #    print(f)

#teste()
