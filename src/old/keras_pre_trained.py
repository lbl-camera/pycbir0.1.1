'''
Created on Jun 8, 2019

@author: flavio
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import warnings 
warnings.filterwarnings("ignore")

from keras.preprocessing import image
import keras.applications.vgg16 as vgg16
import keras.applications.vgg19 as vgg19
import keras.applications.xception as xception
import keras.applications.inception_resnet_v2 as inception_resnet
import keras.applications.resnet50 as resnet
import keras.applications.nasnet as nasnet
from keras.models import load_model
import numpy as np
from keras.models import Model
import tensorflow as tf
import sys

def extract_feature_one_image(img_path,intermediate_layer_model,feature_extraction_method,input_img):
    img = image.load_img(img_path, target_size=(input_img, input_img))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    
    if(feature_extraction_method == 'pretrained_lenet'):
        img_data = img_data/255
    elif(feature_extraction_method == 'pretrained_vgg16'):
        img_data = vgg16.preprocess_input(img_data)
    elif(feature_extraction_method == 'pretrained_vgg19'):
        img_data = vgg19.preprocess_input(img_data)
    elif(feature_extraction_method == 'pretrained_xception'):
        img_data = xception.preprocess_input(img_data)
    elif(feature_extraction_method == 'pretrained_resnet'):
        img_data = resnet.preprocess_input(img_data)
    elif(feature_extraction_method == 'pretrained_inception_resnet'):
        img_data = inception_resnet.preprocess_input(img_data)
    elif(feature_extraction_method == 'pretrained_nasnet'):
        img_data = nasnet.preprocess_input(img_data)
        
    features = intermediate_layer_model.predict(img_data)
    features = features.reshape((-1))
    return features

def create_model(feature_extraction_method,path_cnn_pre_trained,input_size):
    
    if(feature_extraction_method == 'pretrained_lenet'):
        model = load_model(path_cnn_pre_trained)
        input_image = input_size
    elif(feature_extraction_method == 'pretrained_vgg16'):
        model = vgg16.VGG16(weights='imagenet', include_top=True)
        #layer_name = 'fc2'
        input_image = 224
    elif(feature_extraction_method == 'pretrained_vgg19'):
        model = vgg19.VGG19(weights='imagenet', include_top=True)
        #layer_name = 'fc2'
        input_image = 224
    elif(feature_extraction_method == 'pretrained_xception'):
        model = xception.Xception(weights='imagenet', include_top=True)
        #layer_name = 'avg_pool'
        input_image = 299
    elif(feature_extraction_method == 'pretrained_resnet'):
        model = resnet.ResNet50(weights='imagenet', include_top=True)
        #layer_name = 'avg_pool'
        input_image = 224
    elif(feature_extraction_method == 'pretrained_inception_resnet'):
        model = inception_resnet.InceptionResNetV2(weights='imagenet', include_top=True)
        #layer_name = 'avg_pool'
        input_image = 299
    elif(feature_extraction_method == 'pretrained_nasnet'):
        model = nasnet.NASNetLarge(weights='imagenet', include_top=True)
        #layer_name = 'global_average_pooling2d_1'
        input_image = 331
    
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
       
    #model.summary()
    
    return intermediate_layer_model, input_image

def feature_extraction(name_images,labels,path_cnn_pre_trained,feature_extraction_method,input_size=0):
    
    intermediate_layer_model, input_img = create_model(feature_extraction_method,path_cnn_pre_trained,input_size)
    features = []
    
    for id_name,name in enumerate(name_images):
        sys.stdout.write("Extractiong features using {}. Image {} from {}.\r".format(feature_extraction_method,id_name+1,len(name_images)))
        features.append(extract_feature_one_image(name, intermediate_layer_model,feature_extraction_method,input_img))
        sys.stdout.flush()
    sys.stdout.write("\n")
    features = np.asarray(features)
    return features, name_images, labels
