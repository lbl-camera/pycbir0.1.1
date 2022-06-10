'''
Created on 30 de aug de 2016

@author: romuere
'''
import numpy as np
from scipy.spatial.distance import euclidean,cityblock,chebyshev,cosine
from scipy.stats import pearsonr
from scipy.stats import chisquare
from scipy.stats import entropy,ks_2samp
import math
import csv
#from scipy.misc import imread
from skimage.io import imread
np.set_printoptions(threshold='nan')
import glcm, histogram,lbp,hog_rom,CNN_feature_extraction
from skimage.color import rgb2gray
from glob import glob
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pickle 
from rtree import index
from rtree.index import Rtree
from Parameters import Parameters
import train_cnn_tensorFlow
import image_feature_extraction_tensorFlow
import cnn_tensorFlow
import inception_feature_extraction
#---------------------------------------------------------------------------------------------------------------#
'''
Above some distance functions
'''
def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def dist_pearson(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)

def dist_jeffrey(h1,h2):
    h1 = np.array(h1)
    h2 = np.array(h1)
    d = 0;
    m = (h1+h2)/2;

    for i in range(1,len(h1)):
        if (m[i]==0):
            continue;
        x1 = h1[i]*np.log10(h1[i]/m[i]);
        if (np.isnan(x1) == False):
            d = d + x1;
        x2 = h2[i]*np.log10(h2[i]/m[i]);
        if (np.isnan(x2) == False):
            d = d + x2;
    return d

def dist_cvm(h1,h2):

    y1 = np.cumsum(h1);
    y2 = np.cumsum(h2);

    d = sum((y1-y2)**2);
    return d

#---------------------------------------------------------------------------------------------------------------#
'''
CBIR SYSTEM
'''
def similarity_metrics(vec1,vec2,med='all'):
    """
    Function that computes the similarity/distance between two vectors
    
    Parameters
    ----------
    vec1 : list numpy array
        the first vector
    vec2 : list numpy array  
        the second vector
    med : string
        the metric that will be computed
        Minkowski and Standard Measures
            Euclidean Distance : 'ED'
            Cityblock Distance : 'CD'
            Infinity Distance : 'ID'
            Cosine Similarity : 'CS'
        Statistical Measures
            Pearson Correlation Coefficient : 'PCC'
            Chi-Square Dissimilarity : 'CSD'
            Kullback-Liebler Divergence : 'KLD'
            Jeffrey Divergence : 'JD'
            Kolmogorov-Smirnov Divergence : 'KSD'
            Cramer-von Mises Divergence : 'CMD'
            
    Returns
    -------
    similarity/distance : float
        the similarity/distance between the two vectors
    """
    distance = 0
    if med == 'ed':
        distance = euclidean(vec1,vec2)
    elif med == 'cd':
        distance = cityblock(vec1, vec2)
    elif med == 'id':
        distance = chebyshev(vec1,vec2)
    elif med == 'cs':
        distance = cosine(vec1, vec2)
    elif med == 'pcc':
        distance = dist_pearson(vec1, vec2)
    elif med == 'csd':
        distance = chisquare(vec1, vec2)[0]
    elif med == 'kld':
        distance = entropy(vec1,vec2)
    elif med == 'jd':
        distance = dist_jeffrey(vec1, vec2)
    elif med == 'ksd':
        distance = ks_2samp(vec1, vec2)[0]
    elif med == 'cmd':
        distance = dist_cvm(vec1, vec2)
    
    return distance

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
        features = glcm.glcm(imagem, int(list_of_parameters[1]), int(list_of_parameters[0]))
    elif desc == 'fotf':
        features = histogram.histogram(imagem)
    elif desc == 'lbp':
        features = lbp.lbpTexture(imagem, 8*int(list_of_parameters[0]), int(list_of_parameters[0]))
    elif desc == 'hog':
        features = hog_rom.HOG(imagem,int(list_of_parameters[0]),int(list_of_parameters[1]))

    return features

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
    
    '''
    Rodar o descritor na base toda
    path: diretorio da base
    number_of_features: quantidade de atributos que o descritor vai retornar
    folders: cada classe vai estar separada em uma pasta
    '''

#    if folders != []:
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
            database_features[i,:] = descriptor(file,desc,list_of_parameters)

    else: #to compute for a single image
        database_features = np.zeros((number_of_features))
        a = imread(path)
        collection.append(a)
        collection_filenames.append(path)
        database_features[0,:] = descriptor(file,desc,list_of_parameters)

    return (collection,collection_filenames,database_features)


def searching(feature_vectors_database,feature_vectors_retrieval, similarity_metric,image_paths,retrieval_number,file,list_of_parameters,feature_extraction_method,path_database):

    '''
    feature_vectors: atriutos calculados
    labels: label de cada classe
    similarity_metric: qual medida utilizar
    recuperados as k imagens com menor distancia. Se k = 0, entao o valor eh
    setado como sendo o tamanho da classe da imagem
    '''
    
    #name to save the pickle file
    parameters_name = ""
    for parameter in list_of_parameters:
        parameters_name = parameters_name + "_" + parameter
    
    file = path_database + "features/sortingRTree" + "_" + feature_extraction_method + parameters_name +'_'+similarity_metric
           
    feature_vectors_retrieval = preprocessing.scale(feature_vectors_retrieval)
    
    if not(os.path.isfile(file+'.dat')):
        
        #normalize signatures
        feature_vectors_database = preprocessing.scale(feature_vectors_database)
        
        # Create a N-Dimensional index
        p = index.Property()
        p.dimension = feature_vectors_database.shape[1]
        idx = index.Index(file,properties=p)
    
        # Create the tree
        for i,vector in enumerate(feature_vectors_database):
            idx.add(i, vector.tolist())

        #save_format = idx.dumps(idx)
        #with open(file, 'wb') as handle:   
        #pickle.dump(save_format, handle)
    else:
        # Create a N-Dimensional index
        p = index.Property()
        p.dimension = feature_vectors_database.shape[1]
        idx = Rtree(file,properties = p)
        #with open(file, 'rb') as handle:
        #    idx = pickle.load(handle)
           
    # Find closests pair for the first N points
    ########### debug this part ###########
    small_distances = []
    for id1,query in enumerate(feature_vectors_retrieval):
        nearest = list(idx.nearest(query.tolist(), retrieval_number))
        small_distances.append(nearest)
        
    result = []
    for cont1,i in enumerate(small_distances):
        aux = []
        for j in i:
            aux.append(image_paths[j])
        result.append(aux) 
    
    return result
    
def accuracy(small_distances,folder_classes,k):
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
    x = []
    for i in range(len(small_distances)):
        percent = (max(result[i,:])/sum(result[i,:]))*100
        cl = folder_classes[np.argmax(result[i,:])]
        x.append(str(percent)+'\n'+cl)       
    return x

def show_retrieval_indexing(images_retrieval,small_distances,k, path_database, feature_extraction_method, distance, folder_classes):
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

    fig.savefig(path_database + "results/result" + "_" + feature_extraction_method + "_" + distance + "_" + str(k) + "_sorting.png")   # save the figure to file   # save the figure to file
    plt.show()
    #os.system(file)

def get_extension(folders):
    '''
    This is function get the extention of the images in the database
    
    Parameters
    ----------    
    folders:
        Complete path of the database
    
    Returns
    -------
    ext : string or int
        return the image database extension in case this is a valid one, and -1 otherwise 
    '''
    
    extension = ["jpg", "JPG","jpeg","JPEG", "tif","TIF", "bmp", "BMP", "png", "PNG"]#extension that the system accept
    file = glob(folders[0]+'*')[0]
    ext = file.split('/')[-1][-3:]
    if (ext in extension):
        return ext
    else:
        return -1

def cnn_features_extraction(path_database,path_retrieval,path_cnn_trained,folders,image_format,feature_extraction_method,list_of_parameters):
    
    if(feature_extraction_method == "cnn_training" or feature_extraction_method == "cnn_probability_training"):
        parameters = Parameters(256 ,path_database,folders,image_format, path_database + "database/",list_of_parameters)
        
        #se precisar fazer treianmento
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
    
    '''
    This is the main function of the pycbir project, the interface will call this function
    Parameters:
        path_databse:
            Complete path of the database folder
        path_retrieval:
            Complete path of the retrieval images folder, if this value is '', then we will compute the retrieval for one image.
        path_image:
            Complete path of a single retrievial image
    '''
    
    folders = glob(path_database + 'database/*/')
    image_format = get_extension(folders)
    if image_format == -1:
        print('pyCBIR can not read the database images in the current format, to show the formats accepted look in the documentation.')
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
            #get the number of features
            number_of_features = get_number_of_features(folders, image_format,feature_extraction_method,list_of_parameters)
            
            #computing features for the database
            _,fname_database,feature_vectors_database = descriptor_all_database(path_database+'database/', folders, image_format,feature_extraction_method,number_of_features,list_of_parameters)
            np.savetxt(file, feature_vectors_database,delimiter = ',')
            
        #get the number of features
        number_of_features = get_number_of_features(folders, image_format,feature_extraction_method,list_of_parameters)
        #computing features for the retrieval image(s)
        if not ("." + image_format) in path_retrieval:
            ims_retrieval,_,feature_vectors_retrieval = descriptor_all_database(path_retrieval, [], image_format,feature_extraction_method,number_of_features,list_of_parameters)
        else:
            ims_retrieval,_,feature_vectors_retrieval = descriptor_all_database(path_retrieval, -1, image_format,feature_extraction_method,number_of_features,list_of_parameters)
      
    #compute the ranked outputs
    result = searching(feature_vectors_database,feature_vectors_retrieval, distance,fname_database,number_of_images,file,list_of_parameters,feature_extraction_method,path_database)
    #show the ranked output
    #show_retrieval_indexing(ims_retrieval, result, number_of_images, path_database, feature_extraction_method, distance,folders)
    
    return result

def run_sorting():
    #path_database = '/Users/romuere/Dropbox/CBIR/fibers/' 
    #path_retrieval = '/Users/romuere/Dropbox/CBIR/fibers/retrieval/' 
    
    #my laptop
    path_database = '/Users/flavio/Dropbox/Compartilhadas/Romuere/CBIR/fibers/'
    path_retrieval = '/Users/flavio/Dropbox/Compartilhadas/Romuere/CBIR/fibers/retrieval/'
    path_inception = '/Users/flavio/Dropbox/Compartilhadas/Romuere/CBIR/inception-2015-12-05/classify_image_graph_def.pb'
    
    #devBox
    #path_database = '/home/flavio/databases/cifar/'
    #path_retrieval = '/home/flavio/databases/cifar/retrieval/'
    #path_inception = '' 
    
    feature_extraction_method = 'cnn_training'
    distance = 'ed'
    number_of_images = 10
    list_of_parameters = ['0.1', '0']#learning rate, number_of_epochs
    
    run_command_line(path_database,path_retrieval,path_inception,feature_extraction_method,distance,number_of_images,list_of_parameters)

#run_sorting()