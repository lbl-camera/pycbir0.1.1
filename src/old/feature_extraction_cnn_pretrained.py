'''
Created on 7 de set de 2016

@author: flavio
'''

import numpy as np
import tensorflow as tf
from skimage.io import imread
slim = tf.contrib.slim
from src.old.models.inception_resnet_v2 import *
from src.old.models.resnet152 import *
from src.old.models.inception_v4 import *
from src.old.models.alexnet import *
from src.old.models.vgg16 import *
from src.old.models.nasnet import *
from skimage.transform import resize


def im_resize(im,size1,size2):
    
    resize_ = np.zeros((size1,size2,3),dtype=np.uint8)
    if(len(im.shape) == 2):
        r = resize(im[:,:], (size1, size2),preserve_range=True)
        resize_[:,:,0] = r
        resize_[:,:,1] = r
        resize_[:,:,2] = r
    else:  
        r = resize(im[:,:,0], (size1, size2),preserve_range=True)
        g = resize(im[:,:,1], (size1, size2),preserve_range=True)
        b = resize(im[:,:,2], (size1, size2),preserve_range=True)
        resize_[:,:,0] = r
        resize_[:,:,1] = g
        resize_[:,:,2] = b
    
    return resize_

def feature_extraction(name_images,labels,path_model,feature_extraction_method):
    #resetting the graph to clean the variable if it exists
    tf.reset_default_graph()
    print('Model will be restored from ' + path_model)
    
    if(feature_extraction_method == "pretrained_inception_resnet"):
        number_of_features = 1536
        image_size_1 = 299
        image_size_2 = 299
        feature_extraction_layer = 'PreLogitsFlatten'
     
        #the inception_resnet was trained using normalized images
        input_tensor = tf.placeholder(tf.float32, shape=(None,image_size_1,image_size_2,3), name='input_image')
        scaled_input_tensor = tf.scalar_mul((1.0/255), input_tensor)
        scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
        scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)
        
        #Load the model
        sess = tf.Session()
        arg_scope = inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points,_ = inception_resnet_v2(scaled_input_tensor, num_classes=1001 ,is_training=False)
        #tf.reset_default_graph()
    
    
    elif(feature_extraction_method == 'pretrained_vgg'):
        number_of_features = 4096
        image_size_1 = 224
        image_size_2 = 224
        feature_extraction_layer = 'vgg_16/fc7'
        
        input_tensor = tf.placeholder(tf.float32, shape=(None,image_size_1,image_size_2,3), name='input_image')
        
        #Load the model
        sess = tf.Session()
        arg_scope = vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points, _ = vgg_16(input_tensor, num_classes=1000 ,is_training=False)
        

    elif(feature_extraction_method == 'pretrained_nasnet'):
        number_of_features = 4032
        image_size_1 = 331
        image_size_2 = 331
        feature_extraction_layer = 'global_pool'
        
        input_tensor = tf.placeholder(tf.float32, shape=(None,image_size_1,image_size_2,3), name='input_image')
        scaled_input_tensor = tf.scalar_mul((1.0/255), input_tensor)
        scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
        scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)
        
        #Load the model
        sess = tf.Session()
        arg_scope = nasnet_large_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = build_nasnet_large(scaled_input_tensor, num_classes=1001 ,is_training=False)
        
            
    elif(feature_extraction_method == 'pretrained_inception_v4'):
        number_of_features = 1536
        image_size_1 = 299
        image_size_2 = 299
        #checkpoint_exclude_scopes = 'InceptionV4/Logits,InceptionV4/AuxLogits'
        feature_extraction_layer = 'PreLogitsFlatten'
        
        #the inception_resnet was trained using normalized images
        input_tensor = tf.placeholder(tf.float32, shape=(None,image_size_1,image_size_2,3), name='input_image')
        scaled_input_tensor = tf.scalar_mul((1.0/255), input_tensor)
        scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
        scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)
        
        #Load the model
        sess = tf.Session()
        arg_scope = inception_v4_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_v4(scaled_input_tensor, num_classes=1001 ,is_training=False)
     
        
    features_query_images = np.zeros((len(name_images),number_of_features))
    
    '''
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in checkpoint_exclude_scopes.split(',')]
    
        # TODO(sguada) variables.filter_variables()
        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
           
    sess.run(tf.initialize_all_variables())#Remover   
    saver = tf.train.Saver(variables_to_restore)  
    '''
    saver = tf.train.Saver()
    saver.restore(sess, path_model)
    
    cont_index = 0
    for name in name_images:
        
        im = imread(name)
        im = im_resize(im,image_size_1,image_size_2) 
        im = np.array(im)
        im = im.reshape(-1,image_size_1,image_size_2,3)
        
        features_values, _ = sess.run([end_points[feature_extraction_layer], logits], feed_dict={input_tensor: im})
        features_query_images[cont_index,:] = np.reshape(features_values, number_of_features)
        
        print('Number of images = ' + str(cont_index))
        cont_index+=1
            
        
    return features_query_images, name_images, labels
            
            
