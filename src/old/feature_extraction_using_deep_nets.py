'''
Created on 3 de dez de 2016

@author: flavio
'''

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from skimage.io import imread
slim = tf.contrib.slim
from cnn.inception_resnet_v2 import *
from skimage.transform import resize
from models import nets_factory

def format_cnn_tensorFlow(name_images):
    
    X_query = []
    for name in name_images:
        im = imread(name)
        
        #inception only accept images with 3 channels
        if(len(im.shape) < 3):
            im2 = np.zeros((im.shape[0],im.shape[1],3),dtype=np.uint8)
            
            im2[:,:,0]=im[:,:]
            im2[:,:,1]=im[:,:]
            im2[:,:,2]=im[:,:]
            im = im2
            
        X_query.append(im)
        
    return X_query

def features_extraction(name_images,labels,path_inception,feature_method):
    
    X_query = format_cnn_tensorFlow(name_images)
        
    if(feature_method == "cnn"):
        features_query_images = np.zeros((len(X_query),2048))#2048 is the number of features
        
        with gfile.FastGFile(path_inception, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        
        with tf.Session() as sess:
            FC3 = sess.graph.get_tensor_by_name('pool_3:0')
        
            cont_=0
            for i in X_query:  
                features_query_images[cont_,:] = np.squeeze(sess.run(
                    FC3,
                    {'DecodeJpeg:0': np.array(i).astype(np.float32)})) 
                cont_+=1

    elif(feature_method == "cnn_probability"):
        features_query_images = np.zeros((len(X_query),1000))#2048 is the number of features
        
        with gfile.FastGFile(path_inception, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        
        with tf.Session() as sess:
            
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            
            cont_=0
            for i in X_query:  
                features_query_images[cont_,:] = np.squeeze(sess.run(
                    softmax_tensor,
                    {'DecodeJpeg:0': np.array(i).astype(np.float32)})) 
                cont_+=1
  
    return features_query_images, name_images, labels

########### New inception Resnet v2 ###################################

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

def features_extraction_new(name_images,labels,path_inception,feature_method):
    #resetting the graph to clean the variable if it exists
    tf.reset_default_graph()
    
    
    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        'inception_v4',
        num_classes=36,
        is_training=False)
    #logits, _ = network_fn(images)
    
    print(path_inception)
    X_query = format_cnn_tensorFlow(name_images)
    if(feature_method == "cnn"):
        features_query_images = np.zeros((len(X_query),1536))#1536 is the number of features resnet
        
        input_tensor_no_resize = tf.placeholder(tf.float32, shape=(None,None,3), name='input_image_no_resize')
        im_resize = tf.image.resize_image_with_crop_or_pad(input_tensor_no_resize, 299, 299)
        im_resize = tf.reshape(im_resize,[1,299,299,3])
        
        #input_tensor = tf.placeholder(tf.float32, shape=(None,299,299,3), name='input_image')
        scaled_input_tensor = tf.scalar_mul((1.0/255), im_resize)
        scaled_input_tensor = tf.sub(scaled_input_tensor, 0.5)
        scaled_input_tensor = tf.mul(scaled_input_tensor, 2.0)
        
        #Load the model
        sess = tf.Session()
        #arg_scope = inception_resnet_v2_arg_scope()
        #with slim.arg_scope(arg_scope):
        logits, end_points = network_fn(scaled_input_tensor)
        #tf.reset_default_graph()
        saver = tf.train.Saver()
        saver.restore(sess, path_inception)
        
        cont_index = 0
        for im in X_query:
            
            #im = im_resize(im,299,299) #se a imagem for menor que 299 fazer o crop
            #im = np.array(im)
            #im = im.reshape(-1,299,299,3)#logit_values contains the probability of each one of 1001 classes
            features_values, logit_values = sess.run([end_points['PreLogitsFlatten'], logits], feed_dict={input_tensor_no_resize: im})
            features_query_images[cont_index,:] = np.reshape(features_values, 1536)
            #print(cont_index)
            cont_index+=1
        
        return features_query_images, name_images, labels
            
            