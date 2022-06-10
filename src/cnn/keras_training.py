'''
Created on Jun 7, 2019

@author: flavio
'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.filterwarnings("ignore")

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";  # The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"]="0";  # Do other imports now...
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.layers import MaxPooling2D, Activation, BatchNormalization
from keras import optimizers
import keras.applications.vgg16 as vgg16
import keras.applications.vgg19 as vgg19
import keras.applications.xception as xception
import keras.applications.inception_resnet_v2 as inception_resnet
import keras.applications.resnet50 as resnet
import keras.applications.nasnet as nasnet
from keras.models import load_model
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from skimage.transform import resize
import skimage.io as io
from src.util import parallel

''' These functions are not being used
'''
def im_resize(im, size1, size2):
    resize_ = np.zeros((size1, size2, 3), dtype=np.uint8)
    if (len(im.shape) == 2):
        r = resize(im[:, :], (size1, size2), preserve_range=True)
        resize_[:, :, 0] = r
        resize_[:, :, 1] = r
        resize_[:, :, 2] = r
    else:
        r = resize(im[:, :, 0], (size1, size2), preserve_range=True)
        g = resize(im[:, :, 1], (size1, size2), preserve_range=True)
        b = resize(im[:, :, 2], (size1, size2), preserve_range=True)
        resize_[:, :, 0] = r
        resize_[:, :, 1] = g
        resize_[:, :, 2] = b

    return resize_


def read_database_parallel(im, name, label, im_size1=0, im_size2=0, num_channels=3):
    if (im_size1 != 0):
        im = im_resize(im, im_size1, im_size2)

    # sometimes there are gray level images together with rgb images
    if (num_channels > 1 and len(im.shape) == 2):
        im2 = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
        im2[:, :, 0] = im[:, :]
        im2[:, :, 1] = im[:, :]
        im2[:, :, 2] = im[:, :]
        im = im2

    if (num_channels == 1):
        im = np.reshape(im, (im.shape[0], im.shape[1], num_channels))

    return im, name, label, im.shape


def read_database(parameters):
    collection = io.imread_collection(parameters.NAME_IMAGES)
    res = parallel.apply_parallel(collection, collection.files, parameters.LABELS, read_database_parallel,
                                  {'im_size1': parameters.NEW_IMAGE_SIZE1, 'im_size2': parameters.NEW_IMAGE_SIZE2,
                                   'num_channels': parameters.NUM_CHANNELS})

    vector_images = []
    files = []
    for cont, e in enumerate(res):
        vector_images.append(e[0])
        files.append(e[1])
        parameters.LABELS[cont] = e[2]
    parameters.NAME_IMAGES = files
    parameters.IMAGE_SIZE1 = res[0][3][0]
    parameters.IMAGE_SIZE2 = res[0][3][1]

    vector_images = np.asarray(vector_images)

    return vector_images, parameters.LABELS
'''

'''

def create_model_from_scratch(shape_in, num_classes, dropout_value=0.5):
    model = Sequential()

    # First convolutional layer
    model.add(Conv2D(64, kernel_size=5, strides=(1, 1), padding='same', use_bias=True,
                     input_shape=(shape_in[0], shape_in[1], shape_in[2]),
                     name='conv_1'))
    model.add(Activation("relu"))

    # First max-pool layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           name='pool_1', padding='same'))

    model.add(BatchNormalization())

    # Second convolutional layer
    model.add(Conv2D(64, kernel_size=3, strides=(1, 1), use_bias=True,
                     padding='same', name='conv_2'))
    model.add(Activation("relu"))

    model.add(BatchNormalization())

    # Second max-pool layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           name='pool_2', padding='same'))

    # Third convolutional layer
    model.add(Conv2D(48, kernel_size=3, strides=(1, 1), use_bias=True,
                     padding='same', name='conv_3'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # Third max-pool layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           name='pool_3', padding='same'))

    # Dropout
    model.add(Dropout(dropout_value))

    model.add(Flatten(name='flatten'))

    # First Dense
    model.add(Dense(192, name='dense_1', activation='relu', use_bias=True))

    # Dropout
    model.add(Dropout(dropout_value))

    # Second Dense
    model.add(Dense(64, name='dense_last', activation='relu', use_bias=True))

    # Dropout
    model.add(Dropout(dropout_value))

    # Softmax
    model.add(Dense(num_classes, activation='softmax', name='classification'))
    # model.summary()
    return model

def create_model(feature_extraction_method, num_classes):

    if (feature_extraction_method == 'training_vgg16'):
        model = vgg16.VGG16(weights= None, include_top=True)
        input_image = 224
    elif (feature_extraction_method == 'training_vgg19'):
        model = vgg19.VGG19(weights= None, include_top=True)
        input_image = 224
    elif (feature_extraction_method == 'training_xception'):
        model = xception.Xception(weights= None, include_top=True)
        input_image = 299
    elif (feature_extraction_method == 'training_resnet'):
        model = resnet.ResNet50(weights= None, include_top=True)
        input_image = 224
    elif (feature_extraction_method == 'training_inception_resnet'):
        model = inception_resnet.InceptionResNetV2(weights= None, include_top=True)
        input_image = 299
    elif (feature_extraction_method == 'training_nasnet'):
        model = nasnet.NASNetLarge(weights= None, include_top=True)
        input_image = 331

    # Removing the last layer
    model.layers.pop()
    new_layer = Dense(num_classes, activation='softmax', name='predictions')
    model = Model(model.input, new_layer(model.layers[-1].output))

    model.summary()

    return model, input_image

def create_model_from_imagenet(feature_extraction_method, num_classes):

    if (feature_extraction_method == 'fine_tuning_vgg16' or feature_extraction_method == 'pretrained_vgg16'):
        model = vgg16.VGG16(weights= 'imagenet', include_top=True)
        input_image = 224
    elif (feature_extraction_method == 'fine_tuning_vgg19' or feature_extraction_method == 'pretrained_vgg19'):
        model = vgg19.VGG19(weights= 'imagenet', include_top=True)
        input_image = 224
    elif (feature_extraction_method == 'fine_tuning_xception' or feature_extraction_method == 'pretrained_xception'):
        model = xception.Xception(weights= 'imagenet', include_top=True)
        input_image = 299
    elif (feature_extraction_method == 'fine_tuning_resnet' or feature_extraction_method == 'pretrained_resnet'):
        model = resnet.ResNet50(weights= 'imagenet', include_top=True)
        input_image = 224
    elif (feature_extraction_method == 'fine_tuning_inception_resnet' or feature_extraction_method == 'pretrained_inception_resnet'):
        model = inception_resnet.InceptionResNetV2(weights= 'imagenet', include_top=True)
        input_image = 299
    elif (feature_extraction_method == 'fine_tuning_nasnet' or feature_extraction_method == 'pretrained_nasnet'):
        model = nasnet.NASNetLarge(weights= 'imagenet', include_top=True)
        input_image = 331

    # Removing the last layer
    model.layers.pop()
    new_layer = Dense(num_classes, activation='softmax', name='predictions')
    model = Model(model.input, new_layer(model.layers[-1].output))

    model.summary()

    return model, input_image

def train_model(parameters):

    if(parameters.FEATURE_EXTRACTION_METHOD[0:8] == 'training'):
        if(parameters.FEATURE_EXTRACTION_METHOD == 'training_lenet'):
            shape_in = (parameters.IMAGE_SIZE1, parameters.IMAGE_SIZE2, parameters.NUM_CHANNELS)
            model = create_model_from_scratch(shape_in, parameters.NUM_CLASSES, dropout_value=0.5)
            parameters.NEW_IMAGE_SIZE1 = parameters.IMAGE_SIZE1
            parameters.NEW_IMAGE_SIZE2 = parameters.IMAGE_SIZE2
        else:
            model, input_image_size = create_model(parameters.FEATURE_EXTRACTION_METHOD, parameters.NUM_CLASSES)
            parameters.NEW_IMAGE_SIZE1 = input_image_size
            parameters.NEW_IMAGE_SIZE2 = input_image_size
    elif(parameters.FEATURE_EXTRACTION_METHOD[0:11] == 'fine_tuning'):
        if(parameters.PATH_CNN_PRE_TRAINED != ''):
            try:
                model = load_model(parameters.PATH_CNN_PRE_TRAINED)
                parameters.NEW_IMAGE_SIZE1 = parameters.IMAGE_SIZE1
                parameters.NEW_IMAGE_SIZE2 = parameters.IMAGE_SIZE2
                print("Model restored from " + parameters.PATH_CNN_PRE_TRAINED)
            except:
                print('Model not found or incompatible: ',parameters.PATH_CNN_PRE_TRAINED)
                raise ValueError('Model not found or model not compatible from: ', parameters.PATH_CNN_PRE_TRAINED)
        else:
            model, input_image_size = create_model_from_imagenet(parameters.FEATURE_EXTRACTION_METHOD, parameters.NUM_CLASSES)
            parameters.NEW_IMAGE_SIZE1 = input_image_size
            parameters.NEW_IMAGE_SIZE2 = input_image_size

    sgd = optimizers.SGD(lr=parameters.LEARNING_RATE, decay=1e-3)
    # keras.optimizers.Adam(lr = parameters.LEARNING_RATE)
    model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    # Creating dataframe with paths of the images and labels
    df_paths_labels = pd.DataFrame(parameters.NAME_IMAGES, columns=['Paths'])
    df_paths_labels['Labels'] = parameters.LABELS
    df_paths_labels['Labels'] = df_paths_labels['Labels'].astype(str)

    train_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(dataframe=df_paths_labels, x_col='Paths', y_col='Labels',
                                      batch_size=parameters.BATCH_SIZE, shuffle=True,
                                      target_size=(parameters.NEW_IMAGE_SIZE1, parameters.NEW_IMAGE_SIZE2))

    history = model.fit_generator(train_generator, steps_per_epoch=len(df_paths_labels) / parameters.BATCH_SIZE,
        epochs=parameters.NUM_EPOCHS, verbose=1)

    model.save(parameters.PATH_SAVE_CNN)
    print('CNN model saved at: ', parameters.PATH_SAVE_CNN)

def features_extraction(parameters):
    try:
        model = load_model(parameters.PATH_SAVE_CNN)
        print('Model restored to extract features from ' + parameters.PATH_SAVE_CNN)
    except:
        try:
            model = load_model(parameters.PATH_CNN_PRE_TRAINED)
            print('Model restored to extract features from ' + parameters.PATH_CNN_PRE_TRAINED)
        except:
            raise ValueError('Model not found or model not compatible from: ', parameters.PATH_SAVE_CNN, '\nor\n',
                             parameters.PATH_CNN_PRE_TRAINED)

    # Creating dataframe with paths of the images and labels
    df_paths_labels = pd.DataFrame(parameters.NAME_IMAGES, columns=['Paths'])
    df_paths_labels['Labels'] = parameters.LABELS
    df_paths_labels['Labels'] = df_paths_labels['Labels'].astype(str)

    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    # Normalizing
    features_datagen = ImageDataGenerator(rescale=1. / 255)

    features_generator = features_datagen.flow_from_dataframe(dataframe=df_paths_labels, x_col='Paths', y_col='Labels',
                                                        batch_size=parameters.BATCH_SIZE, shuffle=False,
                                                        target_size=(parameters.NEW_IMAGE_SIZE1, parameters.NEW_IMAGE_SIZE1))

    feature_vectors_database = intermediate_layer_model.predict_generator(features_generator, verbose=1)

    probability_vector = np.zeros((len(feature_vectors_database), parameters.NUM_CLASSES))
    return feature_vectors_database, parameters.NAME_IMAGES, parameters.LABELS, probability_vector

def features_extraction_from_pretrained_cnn(parameters):

    if(parameters.PATH_CNN_PRE_TRAINED != ''):
        return features_extraction(parameters)
    else:
        model, input_image_size = create_model_from_imagenet(parameters.FEATURE_EXTRACTION_METHOD, parameters.NUM_CLASSES)
        parameters.NEW_IMAGE_SIZE1 = input_image_size
        parameters.NEW_IMAGE_SIZE2 = input_image_size

        # Creating dataframe with paths of the images and labels
        df_paths_labels = pd.DataFrame(parameters.NAME_IMAGES, columns=['Paths'])
        df_paths_labels['Labels'] = parameters.LABELS
        df_paths_labels['Labels'] = df_paths_labels['Labels'].astype(str)

        intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)

        # Normalizing
        features_datagen = ImageDataGenerator(rescale=1. / 255)

        features_generator = features_datagen.flow_from_dataframe(dataframe=df_paths_labels, x_col='Paths', y_col='Labels',
                                                            batch_size=parameters.BATCH_SIZE, shuffle=False,
                                                            target_size=(parameters.NEW_IMAGE_SIZE1, parameters.NEW_IMAGE_SIZE1))

        feature_vectors_database = intermediate_layer_model.predict_generator(features_generator, verbose=1)

        probability_vector = np.zeros((len(feature_vectors_database), parameters.NUM_CLASSES))
        return feature_vectors_database, parameters.NAME_IMAGES, parameters.LABELS, probability_vector
