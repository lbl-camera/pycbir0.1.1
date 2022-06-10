'''
Computes Daisy features
@author: romue
'''
import numpy as np
from skimage.feature import daisy

def daisy_features(image, name, label,step_ = 4, rings_=3, histograms_=2, orientations_=8):
    
    a = daisy(image,step = step_, rings = rings_, histograms = histograms_, orientations=orientations_)
    a = np.asarray(a)
    a = a.reshape(-1,56)
    a = a.reshape(-1)
    return a,name, label


def test():
    
    #test set
    files = ['/home/users/romuere/databases/cells/cells_test/database/abnormal/*.tif'
             '/home/users/romuere/databases/cells/cells_test/database/normal/*.tif']
    
    #im = np.random.random((100,100))
    #v = daisy_features(im)
    f = []
    for id_f,f in enumerate(files):
        im_collection = imread_collection(f)
        features = np.zeros((len(im_collection),18145))
        for id_im,im in enumerate(im_collection):
            features[id_im,:-1] = daisy_features(im)
            features[id_im,-1] = id_f
        f.append(features)
    
    features_test = np.concatenate((features[0],features[1]))
    
    #test set
    files = ['/home/users/romuere/databases/cells/cells_test/database/abnormal/*.tif'
             '/home/users/romuere/databases/cells/cells_test/database/normal/*.tif']
    
    #im = np.random.random((100,100))
    #v = daisy_features(im)
    f = []
    for id_f,f in enumerate(files):
        im_collection = imread_collection(f)
        features = np.zeros((len(im_collection),18145))
        for id_im,im in enumerate(im_collection):
            features[id_im,:-1] = daisy_features(im)
            features[id_im,-1] = id_f
        f.append(features)
    
    features_train = np.concatenate((features[0],features[1]))
    
    path_output = '/home/users/romuere/databases/cells/'
    np.savetxt(path_output, features_train,delimiter = ',')
    np.savetxt(path_output, features_test,delimiter = ',')
    
#test()