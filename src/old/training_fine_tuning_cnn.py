
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import skimage.io as io
import numpy as np
import tensorflow as tf
from src.util import parallel
from src.cnn import training_parameters
from src.preprocessing import pp2
from src.old.models import lenet
from src.old.models.inception_resnet_v2 import *
from src.old.models.inception_v4 import *
from src.old.models.vgg16 import *
from src.old.models.nasnet import *
from skimage.transform import resize

##################################### Functions for cnn without using wavelet ###################################

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

def transform_database_parallel(im, name, label, im_size1 = 0, im_size2 = 0, preprocessing='none', num_channels = 3):
    
    if(im_size1 != 0):
        im = im_resize(im,im_size1,im_size2)
    
    if(preprocessing == 'log'):
        im,_,_ = pp2.preprocessing(im, '', 0)
    
    #sometimes there are gray level images together with rgb images
    if(num_channels > 1 and len(im.shape)==2):
        im2 = np.zeros((im.shape[0],im.shape[1],3),dtype=np.uint8)
        im2[:,:,0]=im[:,:]
        im2[:,:,1]=im[:,:]
        im2[:,:,2]=im[:,:]
        im = im2
    
    if(num_channels == 3):
        r = im[:,:,0].flatten()
        g = im[:,:,1].flatten()
        b = im[:,:,2].flatten()
        line_image = np.array(list([label]) + list(r) + list(g) + list(b),np.uint8)
    else:
        r = im[:,:].flatten()
        line_image = np.array(list([label]) + list(r),np.uint8)

    return line_image,name,label,im.shape

def transform_database(parameters):
    path_output = parameters.PATH_OUTPUT + '/label_images.bin'
    
    collection = io.imread_collection(parameters.NAME_IMAGES)
    
    res = parallel.apply_parallel(collection, collection.files, parameters.LABELS, transform_database_parallel, {'preprocessing': parameters.PREPROCESSING, 'im_size1': parameters.NEW_IMAGE_SIZE1, 'im_size2': parameters.NEW_IMAGE_SIZE2, 'num_channels': parameters.NUM_CHANNELS} )
    
    vector_images = []
    files = []
    for cont,e in enumerate(res):
        vector_images.append(e[0])
        files.append(e[1])
        parameters.LABELS[cont] = e[2]   
    parameters.NAME_IMAGES = files
    parameters.IMAGE_SIZE1 = res[0][3][0]  
    parameters.IMAGE_SIZE2 = res[0][3][1]
    
    vector_images = np.asarray(vector_images)
    vector_images.tofile(path_output)
    
def train_lenet(parameters):
    transform_database(parameters)
    list_error = []
    
    tf.reset_default_graph()
    parameters.X = tf.placeholder(tf.float32, [parameters.BATCH_SIZE, parameters.IMAGE_SIZE1,parameters.IMAGE_SIZE2,parameters.NUM_CHANNELS])
    parameters.FEATURES_LAYER = lenet.inference_pred(parameters)
        
    parameters.IMAGE_NORMALIZATION = tf.placeholder(np.uint8, [parameters.IMAGE_SIZE1, parameters.IMAGE_SIZE2, parameters.NUM_CHANNELS])
    parameters.NORMALIZATION = tf.image.per_image_standardization(tf.cast(parameters.IMAGE_NORMALIZATION, tf.float32))

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        
        # Get images and labels
        images, labels = training_parameters.distorted_inputs(parameters)
        
        # Build a Graph that computes the logits predictions from the inference model.
        logits,_,_ = lenet.inference_(images, parameters)
        
        # Calculate loss.
        loss = training_parameters.loss(logits, labels)
        
        # Build a Graph that trains the model with one batch of examples and updates the model parameters.
        train_op = training_parameters.train(loss, global_step,parameters)
        
        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=False))
        sess.run(init)
        
        #if exists a trained model, we restored to do the fine-tuning
        try:
            if(parameters.PATH_CNN_PRE_TRAINED != ''):
                saver.restore(sess, parameters.PATH_CNN_PRE_TRAINED)
                print("Model restored from " + parameters.PATH_CNN_PRE_TRAINED)
            else:
                print("Initializing model randomly!")
        except:
            print("Initializing model randomly!")
            pass
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        
        step = 0
        cont_aux = 0
        loss_per_epoch = 0
        while step < parameters.NUM_EPOCHS:
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            
            loss_per_epoch+=loss_value
            
            if(cont_aux == 0): 
                format_str = ('%s: step %d, loss = %.4f')
                print (format_str % (datetime.now(), step, loss_value))
                list_error.append([step, loss_value])
                
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            
            cont_aux+=1
            if( cont_aux >= (parameters.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/parameters.BATCH_SIZE) ):
                num_examples_per_step = parameters.BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                
                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_per_epoch/cont_aux,
                                     examples_per_sec, sec_per_batch))
                list_error.append([step+1, loss_per_epoch/cont_aux])
                loss_per_epoch = 0
                cont_aux = 1
                step+=1
                
                # Save the model checkpoint periodically.
                if (step + 1) % 10 == 0:
                    saver.save(sess, parameters.PATH_SAVE_CNN)
                
        saver.save(sess, parameters.PATH_SAVE_CNN)
        
    parameters.LIST_ERROR = list_error
    #remove the file created in transform_database
    os.remove(parameters.PATH_OUTPUT + "/label_images.bin")


def fine_tuning_cnn(parameters):
    
    if(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_inception_resnet'):
        parameters.NEW_IMAGE_SIZE1 = 299
        parameters.NEW_IMAGE_SIZE2 = 299
        checkpoint_exclude_scopes = 'InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits'
        
    elif(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_vgg'):
        parameters.NEW_IMAGE_SIZE1 = 224
        parameters.NEW_IMAGE_SIZE2 = 224
        checkpoint_exclude_scopes = 'vgg_16/fc8'
        
        
    elif(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_nasnet'):
        parameters.NEW_IMAGE_SIZE1 = 331
        parameters.NEW_IMAGE_SIZE2 = 331
        checkpoint_exclude_scopes = 'final_layer,aux_11'
        
    elif(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_inception_v4'):
        parameters.NEW_IMAGE_SIZE1 = 299
        parameters.NEW_IMAGE_SIZE2 = 299
        checkpoint_exclude_scopes = 'InceptionV4/Logits,InceptionV4/AuxLogits'
    
    transform_database(parameters)
    list_error = []
    
    tf.reset_default_graph()
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        
        # Get images and labels
        images, labels = training_parameters.distorted_inputs(parameters)
        
        # Build a Graph that computes the logits predictions from the inference model.
        if(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_inception_resnet'):
            arg_scope = inception_resnet_v2_arg_scope()
            with slim.arg_scope(arg_scope):
                logits,_,_ = inception_resnet_v2(images, num_classes=parameters.NUM_CLASSES,is_training=True)
                
        elif(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_vgg'):  
            arg_scope = vgg_arg_scope()
            with slim.arg_scope(arg_scope):
                logits,_ = vgg_16(images, num_classes=parameters.NUM_CLASSES,is_training=True)
        
        elif(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_nasnet'):  
            arg_scope = nasnet_large_arg_scope()
            with slim.arg_scope(arg_scope):
                logits,_ = build_nasnet_large(images, num_classes=parameters.NUM_CLASSES, is_training=True)
                
        elif(parameters.FEATURE_EXTRACTION_METHOD == 'fine_tuning_inception_v4'):
            arg_scope = inception_v4_arg_scope()
            with slim.arg_scope(arg_scope):
                logits,_ = inception_v4(images, num_classes=parameters.NUM_CLASSES,is_training=True)
        
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
            
        # Calculate loss.
        loss = training_parameters.loss(logits, labels)
        
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = training_parameters.train(loss, global_step,parameters)
        
        # Create a saver.
        saver = tf.train.Saver(variables_to_restore)
        
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=False))
        sess.run(init)
        
        #Melhorar aqui para ler no try catch e caso nao de certo treinar do zero
        #if the cnn exist, we restore
        #if os.path.isfile(parameters.PATH_CNN_PRE_TRAINED):
        saver.restore(sess, parameters.PATH_CNN_PRE_TRAINED)
        print("Model restored from %s" % parameters.PATH_CNN_PRE_TRAINED)
            
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        
        step = 0
        cont_aux = 0
        loss_per_epoch = 0
        while step < parameters.NUM_EPOCHS:
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            
            loss_per_epoch+=loss_value
            
            if(cont_aux == 0): 
                format_str = ('%s: step %d, loss = %.4f')
                print (format_str % (datetime.now(), step, loss_value))
                list_error.append([step, loss_value])
                
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            
            #if (step + 1) % 10 == 0:
            #    summary_str = sess.run(summary_op)
            #    summary_writer.add_summary(summary_str, step)
                    
            # Save the model checkpoint periodically.
            #if (step + 1) == parameters.NUM_EPOCHS:
            #    saver.save(sess, parameters.PATH_CNN_TRAINED)
            
            cont_aux+=1
            if( cont_aux >= (parameters.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/parameters.BATCH_SIZE) ):
                num_examples_per_step = parameters.BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                
                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_per_epoch/cont_aux,
                                     examples_per_sec, sec_per_batch))
                list_error.append([step+1, loss_per_epoch/cont_aux])
                loss_per_epoch = 0
                cont_aux = 1
                step+=1
                
        saver.save(sess, parameters.PATH_SAVE_CNN)
        
    parameters.LIST_ERROR = list_error
    #remove the file created in transform_database
    os.remove(parameters.PATH_OUTPUT + "/label_images.bin")

'''
def train_vgg(parameters):
    transform_database(parameters)
    list_error = []
    
    tf.reset_default_graph()
    parameters.X = tf.placeholder(tf.float32, [parameters.BATCH_SIZE, parameters.IMAGE_SIZE1,parameters.IMAGE_SIZE2,parameters.NUM_CHANNELS])
    parameters.FEATURES_LAYER = cnn_tensorFlow.inference_pred(parameters)
        
    parameters.IMAGE_NORMALIZATION = tf.placeholder(np.uint8, [parameters.IMAGE_SIZE1, parameters.IMAGE_SIZE2, parameters.NUM_CHANNELS])
    parameters.NORMALIZATION = tf.image.per_image_standardization(tf.cast(parameters.IMAGE_NORMALIZATION, tf.float32))

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        
        # Get images and labels
        images, labels = cnn_tensorFlow.distorted_inputs(parameters)
        
        # Build a Graph that computes the logits predictions from the
        # inference model.
        #logits = cnn_tensorFlow.inference_train(images,parameters)
        arg_scope = vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, _ = vgg_16(images, num_classes=parameters.NUM_CLASSES,is_training=True)
        
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
                    
            #variables_to_restore = slim.get_variables_to_restore()
            
            
            
        # Calculate loss.
        loss = cnn_tensorFlow.loss(logits, labels)
        
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cnn_tensorFlow.train(loss, global_step,parameters)
        
        # Create a saver.
        saver = tf.train.Saver(variables_to_restore)
        
        # Build the summary operation based on the TF collection of Summaries.
        #summary_op = tf.merge_all_summaries()
        
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=False))
        sess.run(init)
        
        #if the cnn exist, we restore
        if os.path.isfile(parameters.PATH_CNN_PRE_TRAINED):
            saver.restore(sess, parameters.PATH_CNN_PRE_TRAINED)
            print("Model restored from %s" % parameters.PATH_CNN_PRE_TRAINED)
            
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        
        #summary_writer = tf.train.SummaryWriter(parameters.PATH_OUTPUT + "tensorboard/", sess.graph)
        
        step = 0
        cont_aux = 0
        loss_per_epoch = 0
        while step < parameters.NUM_EPOCHS:
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            
            loss_per_epoch+=loss_value
            
            if(cont_aux == 0): 
                format_str = ('%s: step %d, loss = %.4f')
                print (format_str % (datetime.now(), step, loss_value))
                list_error.append([step, loss_value])
                
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            
            #if (step + 1) % 10 == 0:
            #    summary_str = sess.run(summary_op)
            #    summary_writer.add_summary(summary_str, step)
                    
            # Save the model checkpoint periodically.
            #if (step + 1) == parameters.NUM_EPOCHS:
            #    saver.save(sess, parameters.PATH_CNN_TRAINED)
            
            cont_aux+=1
            if( cont_aux >= (parameters.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/parameters.BATCH_SIZE) ):
                num_examples_per_step = parameters.BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                
                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_per_epoch/cont_aux,
                                     examples_per_sec, sec_per_batch))
                list_error.append([step+1, loss_per_epoch/cont_aux])
                loss_per_epoch = 0
                cont_aux = 1
                step+=1
                
        saver.save(sess, parameters.PATH_SAVE_CNN)
        
    parameters.LIST_ERROR = list_error
    #remove the file created in transform_database
    os.remove(parameters.PATH_OUTPUT + "/label_images.bin")

def train_nasnet(parameters):
    transform_database(parameters)
    list_error = []
    
    tf.reset_default_graph()
    parameters.X = tf.placeholder(tf.float32, [parameters.BATCH_SIZE, parameters.IMAGE_SIZE1,parameters.IMAGE_SIZE2,parameters.NUM_CHANNELS])
    parameters.FEATURES_LAYER = cnn_tensorFlow.inference_pred(parameters)
        
    parameters.IMAGE_NORMALIZATION = tf.placeholder(np.uint8, [parameters.IMAGE_SIZE1, parameters.IMAGE_SIZE2, parameters.NUM_CHANNELS])
    parameters.NORMALIZATION = tf.image.per_image_standardization(tf.cast(parameters.IMAGE_NORMALIZATION, tf.float32))

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        
        # Get images and labels
        images, labels = cnn_tensorFlow.distorted_inputs(parameters)
        
        # Build a Graph that computes the logits predictions from the
        # inference model.
        #logits = cnn_tensorFlow.inference_train(images,parameters)
        arg_scope = nasnet_large_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = build_nasnet_large(images, num_classes=parameters.NUM_CLASSES, is_training=True)
        
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
                    
            #variables_to_restore = slim.get_variables_to_restore()
            
        # Calculate loss.
        loss = cnn_tensorFlow.loss(logits, labels)
        
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cnn_tensorFlow.train(loss, global_step,parameters)
        
        # Create a saver.
        saver = tf.train.Saver(variables_to_restore)
        
        # Build the summary operation based on the TF collection of Summaries.
        #summary_op = tf.merge_all_summaries()
        
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=False))
        sess.run(init)
        
        #if the cnn exist, we restore
        if os.path.isfile(parameters.PATH_CNN_PRE_TRAINED):
            saver.restore(sess, parameters.PATH_CNN_PRE_TRAINED)
            print("Model restored from %s" % parameters.PATH_CNN_PRE_TRAINED)
            
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        
        #summary_writer = tf.train.SummaryWriter(parameters.PATH_OUTPUT + "tensorboard/", sess.graph)
        
        step = 0
        cont_aux = 0
        loss_per_epoch = 0
        while step < parameters.NUM_EPOCHS:
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            
            loss_per_epoch+=loss_value
            
            if(cont_aux == 0): 
                format_str = ('%s: step %d, loss = %.4f')
                print (format_str % (datetime.now(), step, loss_value))
                list_error.append([step, loss_value])
                
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            
            #if (step + 1) % 10 == 0:
            #    summary_str = sess.run(summary_op)
            #    summary_writer.add_summary(summary_str, step)
                    
            # Save the model checkpoint periodically.
            #if (step + 1) == parameters.NUM_EPOCHS:
            #    saver.save(sess, parameters.PATH_CNN_TRAINED)
            
            cont_aux+=1
            if( cont_aux >= (parameters.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/parameters.BATCH_SIZE) ):
                num_examples_per_step = parameters.BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                
                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_per_epoch/cont_aux,
                                     examples_per_sec, sec_per_batch))
                list_error.append([step+1, loss_per_epoch/cont_aux])
                loss_per_epoch = 0
                cont_aux = 1
                step+=1
                
        saver.save(sess, parameters.PATH_SAVE_CNN)
        
    parameters.LIST_ERROR = list_error
    #remove the file created in transform_database
    os.remove(parameters.PATH_OUTPUT + "/label_images.bin")
'''
