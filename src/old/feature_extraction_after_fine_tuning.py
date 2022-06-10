'''
Created on 4 de set de 2016

@author: flavio
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import skimage.io as io
from src.old.models import lenet
from src.preprocessing import pp2
from src.util import parallel

slim = tf.contrib.slim
from src.old.models.inception_resnet_v2 import *
from src.old.models.inception_v4 import *
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

def read_images_parallel(im, name, label, im_size1 = 0, im_size2 = 0, preprocessing='none', num_channels = 3, feature_extraction_method = 'none'):
    
    if(im_size1 != 0):
        im = im_resize(im,im_size1,im_size2)
    
    if(preprocessing == 'log'):
        im,_,_ = pp2.preprocessing(im, '', 0)
        
    if( len(im.shape) == 2 and feature_extraction_method[0:11] == "fine_tuning"):
        im2 = np.zeros((im.shape[0],im.shape[1],3),dtype=np.uint8)
        im2[:,:,0]=im[:,:]
        im2[:,:,1]=im[:,:]
        im2[:,:,2]=im[:,:]
        im = im2
      
    if(num_channels == 1):
        im = im.reshape((im.shape[0],im.shape[1],1))
                                
    return im,name,label,im.shape
        
def read_images(parameters):
    
    collection = io.imread_collection(parameters.NAME_IMAGES)
    
    res = parallel.apply_parallel(collection, collection.files, parameters.LABELS, read_images_parallel, {'preprocessing': parameters.PREPROCESSING, 'im_size1': parameters.NEW_IMAGE_SIZE1, 'im_size2': parameters.NEW_IMAGE_SIZE2, 'num_channels': parameters.NUM_CHANNELS, 'feature_extraction_method': parameters.FEATURE_EXTRACTION_METHOD} )
    
    images = []
    files = []
    for cont,e in enumerate(res):
        images.append(e[0])
        files.append(e[1])
        parameters.LABELS[cont] = e[2]   
    parameters.NAME_IMAGES = files
    parameters.IMAGE_SIZE1 = res[0][3][0]  
    parameters.IMAGE_SIZE2 = res[0][3][1]
    parameters.NUM_CHANNELS = res[0][3][2]
    
    return images


def features_extraction_lenet(parameters):
    
    #Number of features extracted of the LeNet
    number_of_features = 192
    
    images = read_images(parameters)
    
    tf.reset_default_graph()
    parameters.X = tf.placeholder(tf.float32, [parameters.BATCH_SIZE, parameters.IMAGE_SIZE1,parameters.IMAGE_SIZE2,parameters.NUM_CHANNELS])
    _,parameters.FEATURES_LAYER,parameters.CLASSIFICATION_LAYER = lenet.inference_(parameters.X, parameters)
        
    parameters.IMAGE_NORMALIZATION = tf.placeholder(np.uint8, [parameters.IMAGE_SIZE1, parameters.IMAGE_SIZE2, parameters.NUM_CHANNELS])
    parameters.NORMALIZATION = tf.image.per_image_standardization(tf.cast(parameters.IMAGE_NORMALIZATION, tf.float32))

    saver = tf.train.Saver()
    
    #tf.get_default_graph().finalize()
    with tf.Session() as sess:
        try:
            saver.restore(sess, parameters.PATH_SAVE_CNN)
            print('Model restored to extract features from ' + parameters.PATH_SAVE_CNN)
        except:
            raise ValueError('Model not found or model not compatible',parameters.PATH_SAVE_CNN)
        
        len_ = (np.uint16(len(images)/parameters.BATCH_SIZE) + 1) * parameters.BATCH_SIZE
        features_vector = np.zeros((len_,number_of_features))
        probability_vector = np.zeros((len_,parameters.NUM_CLASSES))
        cont=0
        
        while(cont <= len(images)):
            aux = []
            for i in range(cont,cont+parameters.BATCH_SIZE):
                if(i < len(images)):
                    aux.append(sess.run(parameters.NORMALIZATION,feed_dict={parameters.IMAGE_NORMALIZATION: images[i]}))
                else:
                    aux.append(sess.run(parameters.NORMALIZATION,feed_dict={parameters.IMAGE_NORMALIZATION: images[-1]}))
                               
            features_vector[cont:cont+parameters.BATCH_SIZE,:] = np.array(sess.run(parameters.FEATURES_LAYER, feed_dict={parameters.X: aux})).reshape(parameters.BATCH_SIZE,number_of_features)
            probability_vector[cont:cont+parameters.BATCH_SIZE,:] = np.array(sess.run(parameters.CLASSIFICATION_LAYER, feed_dict={parameters.X: aux})).reshape(parameters.BATCH_SIZE,parameters.NUM_CLASSES)
            cont+=parameters.BATCH_SIZE
    
    features_vector = features_vector[0:len(images),:]
    probability_vector = probability_vector[0:len(images),:]
    
    return features_vector, parameters.NAME_IMAGES, parameters.LABELS, probability_vector


def features_extraction_cnn(parameters):
    
    if(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_inception_resnet'):
        parameters.NEW_IMAGE_SIZE1 = 299
        parameters.NEW_IMAGE_SIZE2 = 299
        number_of_features = 1536
        checkpoint_exclude_scopes = 'InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits'
        feature_extraction_layer = 'PreLogitsFlatten'
        
    elif(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_vgg'):
        parameters.NEW_IMAGE_SIZE1 = 224
        parameters.NEW_IMAGE_SIZE2 = 224
        number_of_features = 4096
        checkpoint_exclude_scopes = 'vgg_16/fc8'
        feature_extraction_layer = 'vgg_16/fc7'
            
    elif(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_nasnet'):
        parameters.NEW_IMAGE_SIZE1 = 331
        parameters.NEW_IMAGE_SIZE2 = 331
        number_of_features = 4032
        checkpoint_exclude_scopes = 'final_layer,aux_11'
        feature_extraction_layer = 'global_pool'
        
    elif(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_inception_v4'):
        parameters.NEW_IMAGE_SIZE1 = 299
        parameters.NEW_IMAGE_SIZE2 = 299
        number_of_features = 1536
        checkpoint_exclude_scopes = 'InceptionV4/Logits,InceptionV4/AuxLogits'
        feature_extraction_layer = 'PreLogitsFlatten'
    
    images = read_images(parameters)
    
    tf.reset_default_graph()#Testar sem normalizacao e com normalizacao
    parameters.X = tf.placeholder(tf.float32, [parameters.BATCH_SIZE, parameters.NEW_IMAGE_SIZE1,parameters.NEW_IMAGE_SIZE2,parameters.NUM_CHANNELS])
    parameters.IMAGE_NORMALIZATION = tf.placeholder(np.uint8, [parameters.IMAGE_SIZE1, parameters.IMAGE_SIZE2, parameters.NUM_CHANNELS])
    #parameters.IMAGE_NORMALIZATION = tf.placeholder(np.uint8, [parameters.IMAGE_SIZE1, parameters.IMAGE_SIZE2, parameters.NUM_CHANNELS])
    #parameters.NORMALIZATION = tf.image.per_image_standardization(tf.cast(parameters.IMAGE_NORMALIZATION, tf.float32))
    
    
    scaled_input_tensor = tf.scalar_mul((1.0/255), tf.cast(parameters.IMAGE_NORMALIZATION, tf.float32))
    scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
    parameters.NORMALIZATION = tf.multiply(scaled_input_tensor, 2.0)
    
    
    if(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_inception_resnet'):
        arg_scope = inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            _, end_points, _ = inception_resnet_v2(parameters.X,num_classes=parameters.NUM_CLASSES, is_training=False)
    
    elif(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_vgg'):  
        arg_scope = vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            _, end_points = vgg_16(parameters.X,num_classes=parameters.NUM_CLASSES, is_training=False)
        
    elif(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_nasnet'): 
        arg_scope = nasnet_large_arg_scope()
        with slim.arg_scope(arg_scope):
            _, end_points = build_nasnet_large(parameters.X,num_classes=parameters.NUM_CLASSES, is_training=False)
            
    elif(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_inception_v4'):
        arg_scope = inception_v4_arg_scope()
        with slim.arg_scope(arg_scope):
            _, end_points = inception_v4(parameters.X,num_classes=parameters.NUM_CLASSES, is_training=False)
    
    
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
                    
    saver = tf.train.Saver(variables_to_restore)
    
    #tf.get_default_graph().finalize()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        try:
            saver.restore(sess, parameters.PATH_SAVE_CNN)
            print('Model read from %s' % parameters.PATH_SAVE_CNN)
        except:
            raise ValueError('Model not found or incompatible',parameters.PATH_SAVE_CNN)
        
        len_ = (np.uint16(len(images)/parameters.BATCH_SIZE) + 1) * parameters.BATCH_SIZE
        features_vector = np.zeros((len_,number_of_features))
        probability_vector = np.zeros((len_,parameters.NUM_CLASSES))
        cont=0
        
        while(cont <= len(images)):
            aux = []
            for i in range(cont,cont+parameters.BATCH_SIZE):
                if(i < len(images)):
                    aux.append(sess.run(parameters.NORMALIZATION,feed_dict={parameters.IMAGE_NORMALIZATION: images[i]}))
                    #aux.append(images[i])
                else:
                    aux.append(sess.run(parameters.NORMALIZATION,feed_dict={parameters.IMAGE_NORMALIZATION: images[-1]}))
                    #aux.append(images[-1])
            
            features_vector_aux = sess.run([end_points[feature_extraction_layer]], feed_dict={parameters.X: aux})
            features_vector[cont:cont+parameters.BATCH_SIZE,:] = np.reshape(features_vector_aux, (parameters.BATCH_SIZE,number_of_features))
            #probability_vector[cont:cont+parameters.BATCH_SIZE,:] = np.reshape(probability_values, (parameters.BATCH_SIZE,parameters.NUM_CLASSES))
            
            cont+=parameters.BATCH_SIZE
    
    return features_vector[0:len(images),:], parameters.NAME_IMAGES, parameters.LABELS, probability_vector

'''
def features_extraction_vgg(parameters):
    
    images = read_images(parameters)
    
    tf.reset_default_graph()
    parameters.X = tf.placeholder(tf.float32, [parameters.BATCH_SIZE, parameters.IMAGE_SIZE1,parameters.IMAGE_SIZE2,parameters.NUM_CHANNELS])
    parameters.IMAGE_NORMALIZATION = tf.placeholder(np.uint8, [parameters.IMAGE_SIZE1, parameters.IMAGE_SIZE2, parameters.NUM_CHANNELS])
    parameters.NORMALIZATION = tf.image.per_image_standardization(tf.cast(parameters.IMAGE_NORMALIZATION, tf.float32))
    
    arg_scope = vgg_arg_scope()
    with slim.arg_scope(arg_scope):
        _, end_points = vgg_16(parameters.X,num_classes=parameters.NUM_CLASSES, is_training=False)
    
    checkpoint_exclude_scopes = 'vgg_16/fc8'
    #checkpoint_exclude_scopes = ''
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
                    
    saver = tf.train.Saver(variables_to_restore)
    
    #tf.get_default_graph().finalize()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        #if os.path.isfile(parameters.PATH_SAVE_CNN):
        try:
            #print("Model read from %s" % parameters.PATH_SAVE_CNN[:-10])
            #ckpt = tf.train.get_checkpoint_state(parameters.PATH_SAVE_CNN[:-10])
            #print("Model read from %s" % ckpt.model_checkpoint_path)
            
            #saver.restore(sess, ckpt.model_checkpoint_path)
            saver.restore(sess, parameters.PATH_SAVE_CNN)
            #print("Model read from %s" % ckpt.model_checkpoint_path)
            print("Model read from %s" % parameters.PATH_SAVE_CNN)
        except:
            raise ValueError('No model.ckpt found',parameters.PATH_SAVE_CNN)
        
        len_ = (np.uint16(len(images)/parameters.BATCH_SIZE) + 1) * parameters.BATCH_SIZE
        features_vector = np.zeros((len_,4096))# is the number of features
        probability_vector = np.zeros((len_,parameters.NUM_CLASSES))
        cont=0
        
        while(cont <= len(images)):
            aux = []
            for i in range(cont,cont+parameters.BATCH_SIZE):
                if(i < len(images)):
                    aux.append(sess.run(parameters.NORMALIZATION,feed_dict={parameters.IMAGE_NORMALIZATION: images[i]}))
                else:
                    aux.append(sess.run(parameters.NORMALIZATION,feed_dict={parameters.IMAGE_NORMALIZATION: images[-1]}))
            
            features_vector_aux = sess.run([end_points['vgg_16/fc7']], feed_dict={parameters.X: aux})
            features_vector[cont:cont+parameters.BATCH_SIZE,:] = np.reshape(features_vector_aux, (parameters.BATCH_SIZE,4096))
            #probability_vector[cont:cont+parameters.BATCH_SIZE,:] = np.reshape(probability_values, (parameters.BATCH_SIZE,parameters.NUM_CLASSES))
            
            cont+=parameters.BATCH_SIZE
    
    return features_vector[0:len(images),:], parameters.NAME_IMAGES, parameters.LABELS, probability_vector

def features_extraction_nasnet(parameters):
    
    images = read_images(parameters)
    
    tf.reset_default_graph()
    parameters.X = tf.placeholder(tf.float32, [parameters.BATCH_SIZE, parameters.IMAGE_SIZE1,parameters.IMAGE_SIZE2,parameters.NUM_CHANNELS])
    parameters.IMAGE_NORMALIZATION = tf.placeholder(np.uint8, [parameters.IMAGE_SIZE1, parameters.IMAGE_SIZE2, parameters.NUM_CHANNELS])
    parameters.NORMALIZATION = tf.image.per_image_standardization(tf.cast(parameters.IMAGE_NORMALIZATION, tf.float32))
    
    arg_scope = nasnet_large_arg_scope()
    with slim.arg_scope(arg_scope):
        _, end_points = build_nasnet_large(parameters.X,num_classes=parameters.NUM_CLASSES, is_training=False)
    
    checkpoint_exclude_scopes = 'final_layer,aux_11'
    #checkpoint_exclude_scopes = ''
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
                    
    saver = tf.train.Saver(variables_to_restore)
    
    #tf.get_default_graph().finalize()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        #if os.path.isfile(parameters.PATH_SAVE_CNN):
        try:
            #print("Model read from %s" % parameters.PATH_SAVE_CNN[:-10])
            #ckpt = tf.train.get_checkpoint_state(parameters.PATH_SAVE_CNN[:-10])
            #print("Model read from %s" % ckpt.model_checkpoint_path)
            
            #saver.restore(sess, ckpt.model_checkpoint_path)
            saver.restore(sess, parameters.PATH_SAVE_CNN)
            #print("Model read from %s" % ckpt.model_checkpoint_path)
            print("Model read from %s" % parameters.PATH_SAVE_CNN)
        except:
            raise ValueError('No model.ckpt found',parameters.PATH_SAVE_CNN)
        
        len_ = (np.uint16(len(images)/parameters.BATCH_SIZE) + 1) * parameters.BATCH_SIZE
        features_vector = np.zeros((len_,4032))# is the number of features
        probability_vector = np.zeros((len_,parameters.NUM_CLASSES))
        cont=0
        
        while(cont <= len(images)):
            aux = []
            for i in range(cont,cont+parameters.BATCH_SIZE):
                if(i < len(images)):
                    aux.append(sess.run(parameters.NORMALIZATION,feed_dict={parameters.IMAGE_NORMALIZATION: images[i]}))
                else:
                    aux.append(sess.run(parameters.NORMALIZATION,feed_dict={parameters.IMAGE_NORMALIZATION: images[-1]}))
            
            features_vector_aux = sess.run([end_points['global_pool']], feed_dict={parameters.X: aux})
            features_vector[cont:cont+parameters.BATCH_SIZE,:] = np.reshape(features_vector_aux, (parameters.BATCH_SIZE,4032))
            #probability_vector[cont:cont+parameters.BATCH_SIZE,:] = np.reshape(probability_values, (parameters.BATCH_SIZE,parameters.NUM_CLASSES))
            
            cont+=parameters.BATCH_SIZE
    
    return features_vector[0:len(images),:], parameters.NAME_IMAGES, parameters.LABELS, probability_vector

'''
