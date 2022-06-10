'''
Created on Nov 17, 2016

@author: flavio
'''

import skimage.io as io
import numpy as np
from scipy.misc import imrotate
from skimage.transform import resize
from skimage import transform as tf
import copy
import glob
import os
import random

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

def translate_image(im,list_size):
    list_images = []
    for coord in list_size:
        X = [coord, -coord]
        Y = [coord, -coord]
        
        for x in X:
            for y in Y:
                tform2 = tf.SimilarityTransform(translation=(x, y))
                im_translated = (tf.warp(im, tform2)*255).astype(np.uint8)

                im_translated_mirror = copy.copy(im_translated)
                if(x > 0 and y > 0):
                    im_translated_mirror[-x:,:,:] = (im_translated[im.shape[0]-2*x:-x,:,:])[::-1,:,:]
                    im_translated_mirror[:,-y:,:] = (im_translated_mirror[:,im.shape[1]-2*y:-y,:])[:,::-1,:]
                elif(x < 0 and y < 0):
                    im_translated_mirror[0:-x,:,:] = (im_translated[-x:2*(-x),:,:])[::-1,:,:]   
                    im_translated_mirror[:,0:-y:,:] = (im_translated_mirror[:,-y:2*(-y),:])[:,::-1,:]
                elif(x < 0 and y > 0):
                    im_translated_mirror[x:,:,:] = (im_translated[im.shape[0]+2*x:x,:,:])[::-1,:,:]
                    im_translated_mirror[:,0:y,:] = (im_translated_mirror[:,y:2*y,:])[:,::-1,:]
                elif(x > 0 and y < 0):
                    im_translated_mirror[0:x,:,:] = (im_translated[x:2*x,:,:])[::-1,:,:]
                    im_translated_mirror[:,y:,:] = (im_translated_mirror[:,im.shape[1]+2*y:y,:])[:,::-1,:]
                list_images.append(copy.copy(im_translated_mirror))
    return list_images

def rotate(im,number_rotate):
    theta = np.int(360/number_rotate)
    ang = range(0,360,theta)
    list_images = []
    for i in ang:
        im_rotated = imrotate(im,i).astype(np.uint8)
        list_images.append(im_rotated)
    return list_images

def rotate_translate(im,number_rotate,list_size_translation):
    list_images = rotate(im,number_rotate)
    translated_images = translate_image(im,list_size_translation)
    for im_translated in translated_images:
        list_images.extend(rotate(im_translated,number_rotate))
    return list_images

def run_resize():
    #flavio machine 
    #path_database = '/Users/flavio/Desktop/fmd/fmd_train/database/'
    #path_write = '/Users/flavio/Desktop/fmd/fmd_train_augmentation/'
    
    #cnn machine
    path_database = '/home/users/flavio/databases/dtd/dtd_train/database/'
    path_write = '/home/users/flavio/databases/dtd/dtd_train_resize/'

    image_format = '.jpg'
    size1 = 100
    size2 = 100
    
    path_write = path_write + 'database/'
    if(not os.path.isdir(path_write)):
        os.mkdir(path_write)
    
    classes = glob.glob(path_database + '*/')
    
    for class_ in classes:
        if(not os.path.isdir(path_write + class_.split('/')[-2])):
            os.mkdir(path_write + class_.split('/')[-2])
            
        name_images = glob.glob(class_ + '*' + image_format)
        
        for name in name_images:
            im_orig = io.imread(name)
            im_orig = im_resize(im_orig,size1,size2)
            
            io.imsave(path_write + class_.split('/')[-2] + '/' + name.split('/')[-1],im_orig)
    
def run_augmentation():
    
    #flavio machine 
    #path_database = '/Users/flavio/Desktop/fmd/fmd_train/database/'
    #path_write = '/Users/flavio/Desktop/fmd/fmd_train_augmentation/'
    
    #cnn machine
    path_database = '/home/users/flavio/databases/dtd/dtd_train_resize/database/'
    path_write = '/home/users/flavio/databases/dtd/dtd_train_resize_augmentation/'
    
    image_format = '.jpg'
    #coords = [7,14,20]#cells
    coords = [8,16,24]#fmd
    number_rotate = 4
    
    path_write = path_write + 'database/'
    if(not os.path.isdir(path_write)):
        os.mkdir(path_write)
    
    classes = glob.glob(path_database + '*/')
    
    for class_ in classes:
        if(not os.path.isdir(path_write + class_.split('/')[-2])):
            os.mkdir(path_write + class_.split('/')[-2])
            
        name_images = glob.glob(class_ + '*' + image_format)
        
        for name in name_images:
            im_orig = io.imread(name)
            
            list_ = rotate_translate(im_orig,number_rotate,coords)
            
            cont_index =1
            for im_rotate in list_:
                name_write = name.split('/')[-1][:-4] + '_' + str(cont_index) + name.split('/')[-1][-4:]
                cont_index+=1
                io.imsave(path_write + class_.split('/')[-2] + '/' + name_write,im_rotate)

def run_create_train_test_set():
    image_format = 'jpg'
    percent_train = 0.5
    name_folder_database = 'Saxsgen'
    path_database = '/home/users/flavio/databases/Saxsgen/database/'

    path_write_train = '/home/users/flavio/databases/' + name_folder_database + '/' + name_folder_database + '_train/'
    path_write_test = '/home/users/flavio/databases/' + name_folder_database + '/' + name_folder_database + '_test/'

    folders = glob.glob(path_database + '*')
    
    for class_ in folders:
        name_images = glob.glob(class_ + '/*.' + image_format)

        if not os.path.isdir(path_write_train + class_.split('/')[-1]):
            os.mkdir(path_write_train + '/database/' + class_.split('/')[-1])
            #os.mkdir(path_write_train + class_.split('/')[-1] + '/database/')

        if not os.path.isdir(path_write_test + class_.split('/')[-1]):
            os.mkdir(path_write_test + '/database/' + class_.split('/')[-1])
            #os.mkdir(path_write_test + class_.split('/')[-1] + '/database/')

        random.shuffle(name_images)

        name_images_train = name_images[0:np.int(len(name_images)*percent_train)]
        name_images_test = name_images[np.int(len(name_images)*percent_train):]

        for name in name_images_train:
            im = io.imread(name)
            io.imsave(path_write_train + '/database/' + class_.split('/')[-1] + '/' + name.split('/')[-1],im)

        for name in name_images_test:
            im = io.imread(name)
            io.imsave(path_write_test + '/database/' + class_.split('/')[-1] + '/' + name.split('/')[-1],im)

#run_augmentation()

#run_resize()

run_create_train_test_set()
    
    