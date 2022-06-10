import signatures.fisherVector as fv
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer

def teste():
    
    import csv
    import itertools

    path = '/Users/romuere/Dropbox/Berkeley/workspace/pycbir 2/files/'
    file_train_features = 'feature_vectors_cnn_training.csv'
    file_test_features = 'feature_vectors_cnn_test.csv'
    file_train_labels = 'labels_cnn_training.csv'
    file_test_labels = 'labels_cnn_test.csv'
    
    
    reader = csv.reader(open(path+file_train_features),delimiter=',')
    x = list(reader)
    train_features = np.array(x).astype(dtype = np.float64)
    
    reader = csv.reader(open(path+file_test_features),delimiter=',')
    x = list(reader)
    test_features = np.array(x).astype(dtype = np.float64)
    
    reader = csv.reader(open(path+file_train_labels),delimiter=',')
    x = list(reader)
    train_labels = np.array(x).astype(dtype = np.uint16)
    train_labels = train_labels.reshape(-1)
    
    reader = csv.reader(open(path+file_test_labels),delimiter=',')
    x = list(reader)
    test_labels = np.array(x).astype(dtype = np.uint16)
    test_labels = test_labels.reshape(-1)
    
    
    
    """
    feature_vectors_train = np.zeros((3990,8))
    feature_vectors_train[:,:-1] = np.concatenate((feature_vectors_database[5:2001,:],feature_vectors_database[2006:,:]))
    labels_train = np.concatenate((labels_database[5:2001],labels_database[2006:]))
    feature_vectors_train[:,-1] = labels_train
    
    feature_vectors_test = np.zeros((10,8))
    feature_vectors_test[:,:-1] = np.concatenate((feature_vectors_database[0:5,:],feature_vectors_database[2001:2006,:]))
    labels_test = np.concatenate((labels_database[0:5],labels_database[2001:2006]))
    feature_vectors_test[:,-1] = labels_test
    """
    feature_size = 192
    n_comp = 2
    a = fv.fisher(train_features, test_features, n_comp, feature_size)
    
    b = 1
teste()