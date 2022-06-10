'''
Created on Jul 3, 2019

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
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout 
from keras.layers import MaxPooling2D, BatchNormalization, Activation
from keras.losses import categorical_crossentropy
from keras import optimizers
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Model
from src.cnn.keras_training import read_database

def create_model(feature_extraction_method,num_classes,path_cnn_pre_trained,input_size):
    
    if(feature_extraction_method == 'fine_tuning_lenet'):
        model = load_model(path_cnn_pre_trained)
        input_image = input_size
    if(feature_extraction_method == 'fine_tuning_vgg16'):
        model = vgg16.VGG16(weights='imagenet', include_top=True)
        #layer_name = 'fc2'
        input_image = 224
    elif(feature_extraction_method == 'fine_tuning_vgg19'):
        model = vgg19.VGG19(weights='imagenet', include_top=True)
        #layer_name = 'fc2'
        input_image = 224
    elif(feature_extraction_method == 'fine_tuning_xception'):
        model = xception.Xception(weights='imagenet', include_top=True)
        #layer_name = 'avg_pool'
        input_image = 299
    elif(feature_extraction_method == 'fine_tuning_resnet'):
        model = resnet.ResNet50(weights='imagenet', include_top=True)
        #layer_name = 'avg_pool'
        input_image = 224
    elif(feature_extraction_method == 'fine_tuning_inception_resnet'):
        model = inception_resnet.InceptionResNetV2(weights='imagenet', include_top=True)
        #layer_name = 'avg_pool'
        input_image = 299
    elif(feature_extraction_method == 'fine_tuning_nasnet'):
        model = nasnet.NASNetLarge(weights='imagenet', include_top=True)
        #layer_name = 'global_average_pooling2d_1'
        input_image = 331
    
    #Removing the last layer 
    model.layers.pop()
    new_layer = Dense(num_classes, activation='softmax', name='predictions')
    model = Model(model.input, new_layer(model.layers[-1].output))
    
    model.summary()
    
    return model, input_image

def fine_tuning_cnn(parameters):
    model, input_image_size = create_model(parameters.FEATURE_EXTRACTION_METHOD, parameters.NUM_CLASSES, parameters.PATH_CNN_PRE_TRAINED, parameters.IMAGE_SIZE1)
    parameters.NEW_IMAGE_SIZE1 = input_image_size
    parameters.NEW_IMAGE_SIZE2 = input_image_size
    
    #try:
    #    if(parameters.PATH_CNN_PRE_TRAINED != ''):
    #        model = load_model(parameters.PATH_CNN_PRE_TRAINED)
    #        print("Model restored from " + parameters.PATH_CNN_PRE_TRAINED)
    #except:
    #    print("Initializing model with ImageNet weights!")
    #    pass
    
    train_datagen = ImageDataGenerator(rescale=1./255)
     
    sgd = optimizers.SGD(lr=parameters.LEARNING_RATE, decay=1e-3)
    #keras.optimizers.Adam(lr = parameters.LEARNING_RATE)
    model.compile(sgd,
              loss='categorical_crossentropy', metrics=['accuracy'])
    
    X_train, Y_train = read_database(parameters)
    
    Y_train = to_categorical(Y_train)
    
    history = model.fit_generator(
      train_datagen.flow(X_train,Y_train, batch_size=parameters.BATCH_SIZE),
      steps_per_epoch=len(Y_train)/parameters.BATCH_SIZE,
      epochs=parameters.NUM_EPOCHS,verbose=1)
    
    model.save(parameters.PATH_SAVE_CNN)
    
def feature_extraction(name_images,labels,path_model,feature_extraction_method, num_classes,input_img_size):
    
    #model, input_img_size = create_model(feature_extraction_method, num_classes)
    model = load_model(path_model)
    #To get the output of the CNN in the layer before the last
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    
    features = []
    
    for id_name,name in enumerate(name_images):
        sys.stdout.write("Extractiong features using {}. Image {} from {}.\r".format(feature_extraction_method,id_name+1,len(name_images)))
        features.append(extract_feature_one_image(name, intermediate_layer_model,feature_extraction_method,input_img_size))
        sys.stdout.flush()
    sys.stdout.write("\n")

    features = np.asarray(features)
    
    probability_vector = np.zeros((len(features),num_classes))
    return features, name_images, labels, probability_vector

def extract_feature_one_image(img_path,intermediate_layer_model,feature_extraction_method,input_img):
    img = image.load_img(img_path, target_size=(input_img, input_img))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = img_data/255
    #if(feature_extraction_method == 'fine_tuning_vgg16'):
    #    #img_data = vgg16.preprocess_input(img_data)
    #img_data = img_data/255
    #elif(feature_extraction_method == 'fine_tuning_vgg19'):
    #    #img_data = vgg19.preprocess_input(img_data)
    #   img_data = img_data/255
    #elif(feature_extraction_method == 'fine_tuning_xception'):
    #    #img_data = xception.preprocess_input(img_data)
    #    img_data = img_data/255
    #elif(feature_extraction_method == 'fine_tuning_resnet'):
    #   #img_data = resnet.preprocess_input(img_data)
    #    img_data = img_data/255
    #elif(feature_extraction_method == 'fine_tuning_inception_resnet'):
    #    #img_data = inception_resnet.preprocess_input(img_data)
    #    img_data = img_data/255
    #elif(feature_extraction_method == 'fine_tuning_nasnet'):
    #    #img_data = nasnet.preprocess_input(img_data)
    #    img_data = img_data/255
        
    features = intermediate_layer_model.predict(img_data)
    features = features.reshape((-1))
    return features
    
