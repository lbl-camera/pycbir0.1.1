'''
Created on 30 de aug de 2016

@author: romuere
'''
import numpy as np
from skimage.io import imread_collection
from src.signatures import glcm,histogram,lbp,hog,daisy
from src.util import parallel
import glob
from src.preprocessing import preprocessing

def descriptor_one_image(imagem,feature_extraction_method,list_of_parameters):
    """
    Function to compute feature vector for one image
    
    Parameters
    ----------
    imagem : numpy array
        RGB or grayscale image
    feature_extraction_method : string
        descriptor to compute the features
    list_of_parameters : numpy array or list
        parameters of each descriptor, can be different sizes depending the descriptor
        
    Returns
    -------
    features : numpy array
        the feature vector computed by the descriptor 'feature_extraction_method'
    """
    if feature_extraction_method == 'glcm':
        features = glcm.glcm(imagem,[],[], int(list_of_parameters[0]), int(list_of_parameters[1]), int(list_of_parameters[2]))
    elif feature_extraction_method == 'fotf':
        features = histogram.histogram(imagem,[],[])
    elif feature_extraction_method == 'lbp':
        features = lbp.lbpTexture(imagem,[],[], 8*int(list_of_parameters[0]), int(list_of_parameters[0]))
    elif feature_extraction_method == 'hog':
        features = hog.HOG(imagem, [], [], int(list_of_parameters[0]), int(list_of_parameters[1]))
    elif feature_extraction_method == 'daisy':
        features = daisy.daisy_features(imagem,[],[], int(list_of_parameters[0]),int(list_of_parameters[1]),int(list_of_parameters[2]),int(list_of_parameters[3]))
    return features[0]

def get_number_of_features(filename,feature_extraction_method,list_of_parameters):
    
    """
    Function to compute the closest value of 'number' in an array
    
    Parameters
    ----------
    vector : list or numpy array
        array with N values
    number : list or numpy array
        a target number
        
    Returns
    -------
    result : int, double
        the value most similar to 'number' in 'vector'
    """
    image = filename
    number_of_features = np.asarray(descriptor_one_image(image,feature_extraction_method,list_of_parameters))
    return number_of_features.shape[0]

def descriptor(collection, collection_filenames, labels,feature_extraction_method,list_of_parameters):
    """
    Function to compute feature vector for an image
    
    Parameters
    ----------
    imagem : numpy array
        RGB or grayscale image
    feature_extraction_method : string
        descriptor to compute the features
    list_of_parameters : numpy array or list
        parameters of each descriptor, can be different sizes depending the descriptor
        
    Returns
    -------
    features : numpy array
        the feature vector computed by the descriptor 'feature_extraction_method'
    """
    #r = apply_parallel(collection, some_processing)
    if feature_extraction_method == 'glcm':
        features = parallel.apply_parallel(collection, collection_filenames, labels, glcm.glcm, {"d": int(list_of_parameters[0]), "grayLevels_new": int(list_of_parameters[1]),"grayLevels_old": int(list_of_parameters[2])})
    elif feature_extraction_method == 'fotf':
        features = parallel.apply_parallel(collection, collection_filenames, labels, histogram.histogram)
    elif feature_extraction_method == 'lbp':
        features = parallel.apply_parallel(collection, collection_filenames, labels, lbp.lbpTexture, {"neighbors": 8*int(list_of_parameters[0]), "radio": 8*int(list_of_parameters[0])} )
    elif feature_extraction_method == 'hog':
        features = parallel.apply_parallel(collection, collection_filenames, labels, hog.HOG, {"cells": int(list_of_parameters[0]), "blocks": int(list_of_parameters[1])})
    elif feature_extraction_method == 'daisy':
        features = parallel.apply_parallel(collection, collection_filenames, labels, daisy.daisy_features,{"step_": int(list_of_parameters[0]), "rings_": int(list_of_parameters[1]),"histograms_": int(list_of_parameters[2]),"orientations_": int(list_of_parameters[3])})
    return features

def descriptor_all_database(filenames,labels,feature_extraction_method,list_of_parameters,preprocessing_method):
    
    """
    Function to compute the closest value of 'number' in an array
    
    Parameters
    ----------
    vector : list or numpy array
        array with N values
    number : list or numpy array
        a target number
        
    Returns
    -------
    result : int, double
        the value most similar to 'number' in 'vector'
    """
    

    len_data = len(filenames) #image database size
    
    
    #matrix to storage the feature vectors
    collection_filenames = []
    collection = imread_collection(filenames)
    collection_filenames = collection.files
    if(preprocessing_method != 'none'):
        collection, collection_filenames, labels = preprocessing.preprocessing(collection,labels,preprocessing_method)

    aux = descriptor(collection, collection_filenames, labels,feature_extraction_method,list_of_parameters)
    number_of_features = len(aux[0][0])
    database_features = np.zeros((len_data,number_of_features))
    collection_filenames = []
    for cont,e in enumerate(aux):
        database_features[cont,:] = e[0]
        collection_filenames.append(e[1])
        labels[cont] = e[2]
    
    database_features = normalize(database_features)
        
    return (collection_filenames,database_features,labels)

def normalize(database_features):
    
    for id in range(len(database_features)):
        database_features[id,:] = (database_features[id,:] - np.min(database_features[id,:]))/(np.max(database_features[id,:])-np.min(database_features[id,:]))
        
    return database_features

def test_function():
    folders = ['/Users/romuere/Dropbox/CBIR/fibers/database/no_fibers/*','/Users/romuere/Dropbox/CBIR/fibers/database/yes_fibers/*']
    filenames = []
    labels = np.empty(0)
    for id,f in enumerate(folders):
        files = glob.glob(f)
        labels = np.append(labels, np.zeros(len(files))+id)
        filenames = filenames+files
        
    feature_extraction_method = 'fotf'
    list_of_parameters = []
    
    fnames,features,labels = descriptor_all_database(filenames,labels,feature_extraction_method,list_of_parameters)
    np.savetxt('feature_vectors_database.csv',features, delimiter = ',')
    np.savetxt('fname_databases.csv',fnames, fmt='%s')
    np.savetxt('labels.csv',labels,fmt = '%d')
    
#test_function()