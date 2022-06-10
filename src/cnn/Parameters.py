'''
Created on 2 de set de 2016
Last Modified on 26 de set de 2019

@author: flavio, dani
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import warnings 
warnings.filterwarnings("ignore")

from imageio import imread
import numpy as np
from src.preprocessing import pp2

class Parameters(object):

    IMAGE_SIZE1 = 0
    IMAGE_SIZE2 = 0
    NEW_IMAGE_SIZE1 = 0
    NEW_IMAGE_SIZE2 = 0
    NUM_CHANNELS = 0
    NUM_CLASSES = 0
    BATCH_SIZE = 0
    LEARNING_RATE = 0
    NUM_EPOCHS = 0
    NAME_IMAGES = []
    LABELS = []
    PATH_OUTPUT = ''
    PATH_CNN_PRE_TRAINED = ''
    PATH_SAVE_CNN = ''
    PREPROCESSING = 'none'
    FEATURE_EXTRACTION_METHOD = ''
    LIST_ERROR = []

    #Not used
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0
    FEATURES_LAYER = []
    CLASSIFICATION_LAYER = []
    X = []
    NORMALIZATION = []
    IMAGE_NORMALIZATION = []
    ID_CHANNEL = 0
    NUM_LEVEL = 0


    #Filling the parameters
    def __init__(self, batch_size, name_images, labels, path_output,path_cnn_pre_trained,path_save_cnn,list_parameters,preprocessing,feature_extraction_method = 'none'):
        self.BATCH_SIZE = batch_size
        self.NAME_IMAGES = name_images
        self.LABELS = labels
        self.PATH_OUTPUT = path_output
        self.PATH_CNN_PRE_TRAINED = path_cnn_pre_trained
        self.PATH_SAVE_CNN = path_save_cnn
        self.PREPROCESSING = preprocessing
        self.FEATURE_EXTRACTION_METHOD = feature_extraction_method

        try:
            #Learning rate and number of epochs
            if(not list_parameters):
                list_parameters.append(0)
                list_parameters.append(0)
            elif(list_parameters[0] != '' and list_parameters[1] != ''):
                self.LEARNING_RATE = np.double(list_parameters[0])
                self.NUM_EPOCHS = int(list_parameters[1])
        except:
            raise ValueError('Error in the parameters!')


        #number of examples per epochs for train
        self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = len(name_images)

        #number of classes
        self.NUM_CLASSES = np.unique(labels).shape[0]

        #Size and number of channels of the images
        im = imread(name_images[0])
        self.IMAGE_SIZE1 = im.shape[0]
        self.IMAGE_SIZE2 = im.shape[1]
        self.NEW_IMAGE_SIZE1 = im.shape[0]
        self.NEW_IMAGE_SIZE2 = im.shape[1]

        if(len(im.shape) > 2 or feature_extraction_method[0:11] == "fine_tuning"):
            self.NUM_CHANNELS = 3
        else:
            self.NUM_CHANNELS = 1
