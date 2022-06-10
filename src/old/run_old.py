'''
Created on 30 de aug de 2016

@author: romuere
'''
import numpy as np
import math
import csv
from skimage.io import imread
import glcm, histogram,lbp,hog_rom
from skimage.color import rgb2gray
from glob import glob
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import os

from Parameters import Parameters
import train_cnn_tensorFlow
import image_feature_extraction_tensorFlow
import cnn_tensorFlow
import inception_feature_extraction
import searching
import parallel
#---------------------------------------------------------------------------------------------------------------#
'''
CBIR SYSTEM
'''

def descriptor(imagem,desc,list_of_parameters):
    """
    Function to compute feature vector for an image
    
    Parameters
    ----------
    imagem : numpy array
        RGB or grayscale image
    desc : string
        descriptor to compute the features
    list_of_parameters : numpy array or list
        parameters of each descriptor, can be different sizes depending the descriptor
        
    Returns
    -------
    features : numpy array
        the feature vector computed by the descriptor 'desc'
    """
    if desc == 'glcm':
        features = glcm.glcm(imagem,[], int(list_of_parameters[1]), int(list_of_parameters[0]))
    elif desc == 'fotf':
        features = histogram.histogram(imagem,[])
    elif desc == 'lbp':
        features = lbp.lbpTexture(imagem,[], 8*int(list_of_parameters[0]), int(list_of_parameters[0]))
    elif desc == 'hog':
        features = hog_rom.HOG(imagem,[],int(list_of_parameters[0]),int(list_of_parameters[1]))

    return features[0]

def get_number_of_features(folders,image_format,desc,list_of_parameters):
    
    """
    Function to compute the closest value of 'number' in an array
    
    Parameters
    ----------
    vector : list or numpy array
        array with N values
    number : list or numpy array
        a target number
        
    Returns
    -------
    result : int, double
        the value most similar to 'number' in 'vector'
    """
    number_of_features = np.asarray(descriptor(imread(glob(folders[0]+'*.'+image_format)[4]),desc,list_of_parameters))
    return number_of_features.shape[0]

def descriptor_all_database(path,folders,image_format,desc,number_of_features,list_of_parameters):
    
    """
    Function to compute the closest value of 'number' in an array
    
    Parameters
    ----------
    vector : list or numpy array
        array with N values
    number : list or numpy array
        a target number
        
    Returns
    -------
    result : int, double
        the value most similar to 'number' in 'vector'
    """
 
    collection = [] #collection of images
    collection_filenames = [] #collection of images

    if (folders != []) & (folders != -1): #this case computes feature vectors for all database

        len_data = 0 #image database size
        number_of_classes = len(folders) #each folder represents one class
        labels = range(number_of_classes) #each class is one different label

        #compute the image database size
        for classes in folders:
            len_data += len(glob(classes+'*.'+image_format))
        
        #matrix to storage the feature vectors
        database_features = np.zeros((len_data,number_of_features))

        cont = 0
        for l,f in enumerate(folders):
            a = glob(f+'*.'+image_format)
            for i in range(len(a)):
                file = imread(a[i])
                collection_filenames.append(a[i])
                database_features[cont,:] = descriptor(file,desc,list_of_parameters)
                cont += 1

    elif folders == []:#this else computes the descriptors in retrieval images

        #total of retrieval images
        len_data = len(glob(path+'*.'+image_format))

        #matrix to storage the features vectors
        database_features = np.zeros((len_data,number_of_features))

        #compute descriptors
        a = glob(path+'*.'+image_format)
        for i in range(len(a)):
            file = imread(a[i])
            collection.append(file)
            collection_filenames.append(a[i])
            aux = descriptor(file,desc,list_of_parameters)
            database_features[i,:] = aux#descriptor(file,desc,list_of_parameters)

    else: #to compute for a single image
        database_features = np.zeros((number_of_features))
        a = imread(path)
        collection.append(a)
        collection_filenames.append(path)
        database_features[0,:] = descriptor(file,desc,list_of_parameters)

    return (collection,collection_filenames,database_features)

def accuracy(small_distances,folder_classes,k):
    """
    This is the function compute the accuracy based on the retrieval images.
    In fact, it is just the probability of the object to be of each class 
    
    Parameters
    ----------
    small_distances:list of lists of str
        Each position corresponds to a list of paths. For each retrieval image it has an list of the retrieval sorted
    folder_classes: list of str
        The length of this list corresponds to the number of classes in the database    
    k: int
        Number of image to retrieve for each query
    Returns
    -------
    probabilities: list
        Each position corresponds to a string with the probability of each object. 
    """
    
    for cont,i in enumerate(folder_classes):
        folder_classes[cont] = folder_classes[cont].split('/')[-2]
        
    result = np.zeros((len(small_distances),len(folder_classes)))
    for cont1,i in enumerate(small_distances):
        for j in range(k):
            label = i[j].split('/')
            label = label[-2]
            for cont2,l in enumerate(folder_classes):
                if label == l:
                    result[cont1,cont2] += 1  
    probabilities = []
    for i in range(len(small_distances)):
        percent = (max(result[i,:])/sum(result[i,:]))*100
        cl = folder_classes[np.argmax(result[i,:])]
        probabilities.append(str(percent)+'\n'+cl)       
    return probabilities

def show_retrieval_indexing(images_retrieval,small_distances,k, path_database, feature_extraction_method, distance, folder_classes,searching_method):
    
    """
    This is the function to show and save the visual result of pyCBIR.
    
    Parameters
    ----------
    images_retrievael: list of numpy
        Retrieval images, each position of the list corresponds to an image
    small_distances:list of lists of str
        Each position corresponds to a list of paths. For each retrieval image it has an list of the retrieval sorted
    k: int
        Number of image to retrieve for each query
    path_database: str
        Complete path of the database
    feature_extraction_method: str
        Feature extraction method that follows the README specification
    distace: str
        Similarity metric that follows the README specification
    folder_classes: list of str
        The length of this list corresponds to the number of classes in the database    
    
    Returns
    -------
    
    """
    
    acc = accuracy(small_distances,folder_classes,k)
    fig, ax = plt.subplots(len(images_retrieval),k+1, sharex=True, sharey=True)
    gs1 = GridSpec(len(images_retrieval),k+1)
    gs1.update(wspace=0.025, hspace=0.5) # set the spacing between axes.
    if len(images_retrieval) > 1:
        cont = 0
        for cont2,i in enumerate(images_retrieval):
            ax[cont2,0].imshow(i,cmap='gray',interpolation = 'none')
            ax[cont2,0].set_adjustable('box-forced')
            ax[cont2,0].set_yticks([])
            ax[cont2,0].set_xticks([])
            ax[cont2,0].set_ylabel(acc[cont2],fontsize = 6)
            cont += 1
            #for each retrieval image returns the k nearer images
            for j in range(k):
                ax[cont2,j+1].imshow(imread(small_distances[cont2][j]),cmap='gray',interpolation = 'none')
                ax[cont2,j+1].set_adjustable('box-forced')
                ax[cont2,j+1].set_yticks([])
                ax[cont2,j+1].set_xticks([])

                shortName = small_distances[cont2][j]
                shortName = shortName.split('/')[-1]
                shortName = shortName[0:6]
                ax[cont2,j+1].set_title(shortName,fontsize=8)
                cont += 1
    else:
        ax[0].imshow(images_retrieval[0],cmap='gray',interpolation = 'none')
        ax[0].set_adjustable('box-forced')
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        ax[0].set_ylabel('Input ',fontsize = 8)
        #for each retrieval image returns the k nearer images
        for j in range(k):
            ax[j+1].imshow(imread(small_distances[0][j]),cmap='gray',interpolation = 'none')
            ax[j+1].set_adjustable('box-forced')
            ax[j+1].set_yticks([])
            ax[j+1].set_xticks([])

            shortName = fname_database[int(small_distances[0,j])]
            shortName = shortName.split('/')[-1]
            shortName = shortName[0:6]
            ax[j+1].set_title(shortName,fontsize=8)

    fig.savefig(path_database + "results/result" + "_" + feature_extraction_method + "_" + distance + "_" + str(k) + "_searching_method_" + searching_method +".png")   # save the figure to file   # save the figure to file
    #os.system(file)

def get_extension(folders):
    """
    This is function get the extention of the images in the database
    
    Parameters
    ----------    
    folders: list of str
        Complete path of the database
    
    Returns
    -------
    ext : string or int
        return the image database extension in case this is a valid one, and -1 otherwise 
    """
    
    extension = ["jpg", "JPG","jpeg","JPEG", "tif","TIF", "bmp", "BMP", "png", "PNG"]#extension that the system accept
    file = glob(folders[0]+'*')[0]
    ext = file.split('/')[-1][-3:]
    if (ext in extension):
        return ext
    else:
        return -1

def cnn_features_extraction(path_database,path_retrieval,path_cnn_trained,folders,image_format,feature_extraction_method,list_of_parameters):
    
    """
    Flavio will fill the documentation of this function
    """
    
    if(feature_extraction_method == "cnn_training" or feature_extraction_method == "cnn_probability_training"):
        parameters = Parameters(256 ,path_database,folders,image_format, path_database + "database/",list_of_parameters)
        
        #if the train is necessary
        if(parameters.NUM_EPOCHS > 0):
            _,_ = train_cnn_tensorFlow.train(parameters)
        
        file = path_database + "features/result" + "_" + feature_extraction_method + ".csv"
        
        if os.path.isfile(file) and parameters.NUM_EPOCHS == 0: 
            fname_database = []
            reader = csv.reader(open(file),delimiter=',')
            x = list(reader)
            feature_vectors_database = np.array(x).astype('float')
            for f in folders:
                a = glob(f+'*.'+image_format)
                for i in range(len(a)):
                    fname_database.append(a[i])
        else:
            feature_vectors_database, _, fname_database = image_feature_extraction_tensorFlow.features_extraction(parameters)
            np.savetxt(file, feature_vectors_database,delimiter = ',')
        
        #calling the extraction of features for the retrieval images
        parameters.PATH_TEST = path_retrieval
        parameters.CLASSES = []
        feature_vectors_retrieval,ims_retrieval,_ = image_feature_extraction_tensorFlow.features_extraction(parameters)
        
        return fname_database, feature_vectors_database, ims_retrieval, feature_vectors_retrieval, file
    
    elif(feature_extraction_method == "cnn" or feature_extraction_method == "cnn_probability"):
        
        file = path_database + "features/result" + "_" + feature_extraction_method + ".csv"
        
        if os.path.isfile(file): 
            fname_database = []
            reader = csv.reader(open(file),delimiter=',')
            x = list(reader)
            feature_vectors_database = np.array(x).astype('float')
            for f in folders:
                a = glob(f+'*.'+image_format)
                for i in range(len(a)):
                    fname_database.append(a[i])
        else:
            feature_vectors_database, fname_database, _ = inception_feature_extraction.features_extraction(path_database + "database/",path_cnn_trained,image_format, feature_extraction_method,True)
            np.savetxt(file, feature_vectors_database,delimiter = ',')
            
        feature_vectors_retrieval,_ , ims_retrieval  = inception_feature_extraction.features_extraction(path_retrieval,path_cnn_trained,image_format, feature_extraction_method,False)
        return fname_database, feature_vectors_database, ims_retrieval, feature_vectors_retrieval, file
   
    
def run_command_line(path_database,path_retrieval,path_cnn_trained,feature_extraction_method,distance,number_of_images,list_of_parameters):
    
    """
    This is the main function of the pycbir project, the interface will call 
    this function. At the end of the process it will show the result and will 
    save a .png file with the result. Also, it will return the 'result' 
    variable to be used to compute the retrieval performance.
    
    Parameters
    ----------
    path_databse: str
        Complete path of the database folder
    path_retrieval:str
        Complete path of the retrieval images folder, if this value is '', then it will compute the retrieval for one image.
    path_image: str
        Complete path of a single retrievial image
    path_cnn_trained: str
        Complete path of a trained cnn
    feature_extraction_method: str
        Feature extraction method that follows the README specification
    distace: str
        Similarity metric that follows the README specification
    number_of_images: int
        Number of images to retrieve
    list_of_parameters: list
        Parameters for the feature extraction method chosed
    
    Returns
    -------
    result : list
        list of all points between start and end
    """
        
    folders = glob(path_database + 'database/*/')
    image_format = get_extension(folders)
    if image_format == -1:
        print('pyCBIR can not read the database images in the current format, to show the formats accepted look in the documentation (README).')
        sys.exit(0)
    
    if(feature_extraction_method[0:3] == "cnn"):
            fname_database, feature_vectors_database, ims_retrieval, feature_vectors_retrieval, file = cnn_features_extraction(path_database,path_retrieval,path_cnn_trained,folders,image_format,feature_extraction_method,list_of_parameters)
    else: 
        #Get the csv file
        #check if there is a file computed for this descriptor-distance
        parameters_name = ""
        for parameter in list_of_parameters:
            parameters_name = parameters_name + "_" + parameter
            
        file = path_database + "features/result" + "_" + feature_extraction_method + parameters_name + ".csv"
        file2 = path_database + "features/image_paths" + "_" + feature_extraction_method + parameters_name + ".csv"
        if os.path.isfile(file): 
            fname_database = []
            reader = csv.reader(open(file),delimiter=',')
            x = list(reader)
            feature_vectors_database = np.array(x).astype('float')
            
            import itertools
            reader = csv.reader(open(file2))
            fname_database = list(reader)
            fname_database  = list(itertools.chain(*fname_database))
            #for f in folders:
            #    a = glob(f+'*.'+image_format)
            #    for i in range(len(a)):
            #        fname_database.append(a[i])
        else:
            #get the number of features
            number_of_features = get_number_of_features(folders, image_format,feature_extraction_method,list_of_parameters)
            
            #computing features for the database
            _,fname_database,feature_vectors_database = descriptor_all_database(path_database+'database/', folders, image_format,feature_extraction_method,number_of_features,list_of_parameters)
            np.savetxt(file, feature_vectors_database,delimiter = ',')
            np.savetxt(file2, fname_database,fmt='%s')
        #get the number of features
        number_of_features = get_number_of_features(folders, image_format,feature_extraction_method,list_of_parameters)
        #computing features for the retrieval image(s)
        if not ("." + image_format) in path_retrieval:
            ims_retrieval,_,feature_vectors_retrieval = descriptor_all_database(path_retrieval, [], image_format,feature_extraction_method,number_of_features,list_of_parameters)
        else:
            ims_retrieval,_,feature_vectors_retrieval = descriptor_all_database(path_retrieval, -1, image_format,feature_extraction_method,number_of_features,list_of_parameters)
      
    #compute the ranked outputs
    searching_method = 'kd'
    result = searching.searching(feature_vectors_database,feature_vectors_retrieval, distance,fname_database,number_of_images,list_of_parameters,feature_extraction_method,path_database,searching_method)
    #show the ranked output
    show_retrieval_indexing(ims_retrieval, result, number_of_images, path_database, feature_extraction_method, distance,folders,searching_method)
    
    return result

def run_sorting():
    
    """
    This is just a test function, to avoid run the GUI every time.    
    """
    
    path_database = '/Users/romuere/Dropbox/CBIR/fibers/' 
    path_retrieval = '/Users/romuere/Dropbox/CBIR/fibers/retrieval/' 
    path_inception = ''
    #my laptop
    #path_database = '/Users/flavio/Dropbox/Compartilhadas/Romuere/CBIR/cells/'
    #path_retrieval = '/Users/flavio/Dropbox/Compartilhadas/Romuere/CBIR/cells/retrieval/'
    #path_inception = '/Users/flavio/Dropbox/Compartilhadas/Romuere/CBIR/inception-2015-12-05/classify_image_graph_def.pb'
    
    #devBox(cnn machine)
    #path_database = '/home/flavio/databases/cifar/'
    #path_retrieval = '/home/flavio/databases/cifar/retrieval/'
    #path_inception = '' 
    
    feature_extraction_method = 'fotf'
    distance = 'cs'
    number_of_images = 10
    list_of_parameters = []
    #list_of_parameters = ['0.1', '0']#learning rate, number_of_epochs
    
    run_command_line(path_database,path_retrieval,path_inception,feature_extraction_method,distance,number_of_images,list_of_parameters)

#run_sorting()