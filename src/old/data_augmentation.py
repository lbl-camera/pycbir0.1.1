'''
Created on Jul 17, 2018

@author: flavio
'''

import numpy as np
import tensorflow as tf
import skimage.io as io
from scipy.misc import imrotate
from math import ceil, floor
import glob
import os
import random
from skimage.transform import resize

def change_image_scale(X_imgs, scales, im_size1, im_size2):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([im_size1, im_size2], dtype = np.int32)
    
    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, im_size2, im_size2, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.uint8)
    return X_scale_data

def get_translate_parameters(index, percentage, im_size1, im_size2):
    if index == 0: # Translate left 20 percent
        offset = np.array([0.0, percentage], dtype = np.float32)
        size = np.array([im_size1, ceil((1 - percentage) * im_size2)], dtype = np.int32)
        w_start = 0
        w_end = int(ceil((1 - percentage) * im_size2))
        h_start = 0
        h_end = im_size1
    elif index == 1: # Translate right 20 percent
        offset = np.array([0.0, -percentage], dtype = np.float32)
        size = np.array([im_size1, ceil((1 - percentage) * im_size2)], dtype = np.int32)
        w_start = int(floor((percentage) * im_size2))
        w_end = im_size2
        h_start = 0
        h_end = im_size1
    elif index == 2: # Translate top 20 percent
        offset = np.array([percentage, 0.0], dtype = np.float32)
        size = np.array([ceil((1 - percentage) * im_size1), im_size2], dtype = np.int32)
        w_start = 0
        w_end = im_size2
        h_start = 0
        h_end = int(ceil((1 - percentage) * im_size1)) 
    else: # Translate bottom 20 percent
        offset = np.array([-percentage, 0.0], dtype = np.float32)
        size = np.array([ceil((1 - percentage) * im_size1), im_size2], dtype = np.int32)
        w_start = 0
        w_end = im_size2
        h_start = int(floor((percentage) * im_size1))
        h_end = im_size1 
        
    return offset, size, w_start, w_end, h_start, h_end

def translate_images(X_imgs, percentage, direction, im_size1, im_size2):
    offsets = np.zeros((len(X_imgs), 2), dtype = np.float32)
    n_translations = 4
    X_translated_arr = []
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (len(X_imgs), im_size2, im_size2, 3))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        X_translated = np.zeros((len(X_imgs), im_size1, im_size2, 3), dtype = np.float32)
        X_translated.fill(0.0) # Filling background color
        base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(direction, percentage, im_size1, im_size2)
        offsets[:, :] = base_offset 
        glimpses = tf.image.extract_glimpse(X, size, offsets)

        glimpses = sess.run(glimpses, feed_dict = {X: X_imgs})
        X_translated[:, h_start: h_start + size[0], w_start: w_start + size[1], :] = glimpses
        X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype = np.uint8)
    
    return X_translated_arr

def rotate(im,number_rotate):
    theta = np.int(360/number_rotate)
    ang = range(0,360,theta)
    list_images = []
    for i in ang:
        im_rotated = imrotate(im,i).astype(np.uint8)
        list_images.append(im_rotated)
    return list_images


def variation_of_ilumination(X_imgs,var):
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    
    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        gaussian_img = (X_img*var) + (gaussian * (1-var))
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.uint8)
    return gaussian_noise_imgs

#As imagens da base estão com algumas linhas e colunas das extremidades brancas. Essa função substitui esses pixels por preto
def remove_white_lines(image):
    image[:,-1,:] = 0
    image[:,0,:] = 0
    image[-1,:,:] = 0
    image[0,:,:] = 0
    
    return image

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
    
def run_augmentation():
    path_database = '/Users/flavio/Downloads/SIIM/SIIM/SIIM_train/database/'
    path_save_new_database = '/Users/flavio/Downloads/SIIM/SIIM/SIIM_train_augmented/'
    
    image_format = 'png'
    im_size1 = 200
    im_size2 = 200
    
    #Número de rotações que a imagem sofrerá
    number_of_rotations = 10
    
    #Cada imagem rotacionada passa por operações de variação de escalas com os valores a baixo
    #Valores usados para a transformacao de escala. Valores menores que 1 aumentam a escala.
    #Valores maiores que 1 diminuem a escala da imagem
    scales = [0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35]
    
    #Cada imagem rotacionada e com variação de escala sofre variação na iluminação da imagem com ruído gaussiano.
    #A variação na iluminação é feita com valor aleatório entre ilumination_min e ilumination_max 
    ilumination_min = 0.2 # 0 representa a imagem totalmente preta
    ilumination_max = 1 # 1 representa a própria imagem
    
    #Após a variação na iluminação por uma translação com porcentagem e direção aleatória 
    percentage_translated_min = 0
    percentage_translated_max = 0.2
    
    
    folders = glob.glob(path_database + '*') 
        
    for class_ in folders:
        
        #Criando o diretorio para salvar as imagens caso ele nao exista
        if not os.path.isdir(path_save_new_database + class_.split('/')[-1]):
            os.mkdir(path_save_new_database + class_.split('/')[-1])
            
        name_images = glob.glob(class_ + '/*.' + image_format)
        
        for name in name_images:
            im = io.imread(name)
            
            im = remove_white_lines(im)
            
            if(im.shape[0] != im_size1 or im.shape[1] != im_size2):
                im = im_resize(im, im_size1, im_size2)
            
            io.imsave(path_save_new_database + class_.split('/')[-1] + '/' + name.split('/')[-1],im)
            
            #Rotações da imagem
            rotated_images = rotate(im, number_of_rotations)
            
            #Variação de escala em todas as imagens rotacionadas
            scaled_images = change_image_scale(rotated_images, scales, im.shape[0], im.shape[1])
            
            cont = 1
            #Variando a iluminação da imagem e a translação de forma aleatória
            for X_image in scaled_images:
                
                #Fator que controla variação de iluminação
                ilumination_factor = random.uniform(ilumination_min,ilumination_max)
                
                #Variação da iluminação
                image_variation_of_ilumination = variation_of_ilumination([X_image], ilumination_factor)
                
                #Varia de 0 a 3 e representam as 4 direções da translação
                direction = random.randint(0,3)
                
                #Porcentagem da translação
                percentage_translated = random.uniform(percentage_translated_min,percentage_translated_max)
                
                #Translação da imagem
                final_image = translate_images(image_variation_of_ilumination, percentage_translated, direction, im.shape[0], im.shape[1])
    
                io.imsave(path_save_new_database + class_.split('/')[-1] + '/' + name.split('/')[-1][:-4] + '_' + str(cont) + '.png', final_image[0])
                
                cont += 1
            
    

run_augmentation()