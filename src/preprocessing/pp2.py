'''
Created on 05 de out de 2016

@author: romuere
'''

import numpy as np

'''
This preprocessing:
    16 to 8 bits
    resize the images
    log transformation
'''

def preprocessing(im, name, label):
    
    #offset = np.mean(im[90:100,0:10])
    #im = im-offset

    im_8 = im#categorizar(im, 8)
    a = (im_8 == 0)
    im_8[a] = 1
    im_log = np.log(im_8)
    
    #uint8
    im_log = np.uint8(((im_log)*255)/np.max(im_log))
    #im_log = im_log[:,0:100]   
    #resize = transform.seam_carve(img, eimg, 'vertical', im.shape[1]/2)
    return (im_log, name, label)
    
    
    
def categorizar(image, new):    
    old = 16
    L,C = image.shape;
    image = np.array(image,dtype = np.float64)
    for i in range(L):
        for j in range(C):
            image[i,j] = (((2**new)-1)*image[i,j])/((2**old)-1)
    return image    

