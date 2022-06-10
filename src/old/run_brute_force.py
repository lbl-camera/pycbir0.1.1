'''
Created on 3 de jun de 2016

@author: romuere
'''
import numpy as np
from scipy.spatial.distance import euclidean,cityblock,chebyshev,cosine
from scipy.stats import pearsonr
from scipy.stats import chisquare
from scipy.stats import entropy,ks_2samp
import math
import csv
from scipy.misc import imread
np.set_printoptions(threshold='nan')
import glcm, histogram,lbp,hog_rom,CNN_feature_extraction
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
'''
1) Defining your image descriptor - ok
2) Indexing your dataset: apply this image descriptor to each database image - ok
3) Defining your similarity metric - ok
4) Searching: For each new image, return the best results (smallest distances)
'''

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
        features = glcm.glcm(imagem, [], int(list_of_parameters[1]), int(list_of_parameters[0]))
    elif desc == 'fotf':
        features = histogram.histogram(imagem, [])
    elif desc == 'lbp':
        features = lbp.lbpTexture(imagem, [], 8*int(list_of_parameters[0]), int(list_of_parameters[0]))
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

    if (folders != []) & (folders != -1):
        len_data = 0 #total de imagens da base
        number_of_classes = len(folders) #cada pasta eh uma classe
        labels = range(number_of_classes) #cada classe tera um label

        #calcular o tamanho da base
        for classes in folders:
            len_data += len(glob(classes+'*.'+image_format))
        #cria uma matrix para guardar os features
        database_features = np.zeros((len_data,number_of_features+1))


        cont = 0
        for l,f in enumerate(folders):
            a = glob(f+'*.'+image_format)
            #a = imread_collection(f+'*.'+image_format)
            for i in range(len(a)):
                #print(a[i])
                file = imread(a[i])
                collection.append(file)
                collection_filenames.append(a[i])
                file = rgb2gray(file)
                database_features[cont,:-1] = descriptor(file,desc,list_of_parameters)
                database_features[cont,-1] = labels[l]
                cont += 1


    elif folders == []:#this else computes the descriptors in retrieval images

        #total of retrieval images
        len_data = 0
        files = len(glob(path+'*.'+image_format))
        len_data = len(glob(path+'*.'+image_format))

        #create a matrix to storage the features computed
        database_features = np.zeros((len_data,number_of_features))

        #compute descriptors
        cont = 0
        a = glob(path+'*.'+image_format)
        for i in range(len(a)):
            file = imread(a[i])
            collection.append(file)
            collection_filenames.append(a[i])
            file = rgb2gray(file)
            database_features[cont,:] = descriptor(file,desc,list_of_parameters)
            cont += 1
        #return the images to show in a plot


    else: #to compute for a single image
        len_data = 1
        database_features = np.zeros((len_data,number_of_features))
        a = imread(path)
        collection.append(a)
        collection_filenames.append(path)
        file = rgb2gray(a)
        database_features[0,:] = descriptor(file,desc,list_of_parameters)

    return (collection,collection_filenames,database_features)


def searching(feature_vectors_database,labels,feature_vectors_retrieval, similarity_metric):

    '''
    feature_vectors: atriutos calculados
    labels: label de cada classe
    similarity_metric: qual medida utilizar
    recuperados as k imagens com menor distancia. Se k = 0, entao o valor eh
    setado como sendo o tamanho da classe da imagem
    '''
    n = len(feature_vectors_retrieval)
    shape = feature_vectors_database.shape[0]
    #computing the distance between retrieval images and all database
    distances = np.zeros((n,shape))
    for cont1,i in enumerate(feature_vectors_retrieval):
        for cont2,j in enumerate(feature_vectors_database):
            distances[cont1,cont2] = similarity_metrics(i, j, med = similarity_metric)
    #the values are sorted by the less to bigger distances and returns the index
    small_distances = np.zeros((n,shape))
    for cont,d in enumerate(distances):
        small_distances[cont,:] = [i[0] for i in sorted(enumerate(d), key=lambda x:x[1])]

    '''
    Pra essa parte funcionar tem q ter o label das imagens retrieval... teremos?
    #quantidade de elementos em cada classe
    clas = np.unique(labels)
    classes = np.zeros(len(clas))
    for i in range(len(clas)):
        classes[i] = sum(labels == clas[i])

    #criar uma lista de matrizes de confusao, uma para cada classe
    matrizes = []; #lista vazia para armazenar as matrizes
    cont = 0
    base = 0
    kk = k
    for i in classes: #percorre cada classe
        if k == 0:
            kk = i
        aux = np.zeros((int(i),int(kk))) #cria uma matriz com o tamanho igual a quantidade de elementos da classe atual
        cont1 = 0
        linha = 0
        for sd in small_distances: #percorrer o vetor de distancias
            if int(labels[linha]) == int(clas[cont]): #se o vetor de distancias for da mesma classe atual
                cont2 = 0
                for s in sd[0:int(kk)]: #para as i primeiras distancias calculadas
                    if int(clas[cont]) == labels[int(s)]: #se o label for igual -> acertou
                        aux[cont1,cont2] = 1 #a posicao do acerto recebe 1
                    cont2 += 1
                cont1 += 1
            linha += 1
        matrizes.append(aux) #colocar a matriz dentro da lista de matrizes
        cont += 1
    return matrizes
    '''
    return small_distances

#nao esta usando
def compute_accuracy(matrizes,k):
    '''
    matrizes: resultado da funcao searching
    k: number of retrivial images
    this function is only called when the second part of searching function is executed
    '''
    classe = 0

    for i in matrizes:
        acc = []
        imagem = 0
        for j in i:
            print ('Accuracy class '+ str(classe) + ' and image ' + str(imagem) + ' ' + str(sum(j[0:k])/k))
            acc.append(sum(j[0:k])/k)
            imagem += 1
        plt.plot(np.sort(np.array(acc)),'-',label = 'Class '+str(classe))
        classe += 1
    plt.xlabel('Images')
    plt.ylabel('Accuracy')
    plt.legend(loc = 0)
    plt.ylim((-0.1,1.1))
    #plt.show()

def show_retrieval(images_database,images_retrieval,small_distances,k, path_database, feature_extraction_method, distance,fname_database, label_database, folder_classes):
    accu_ = compute_accuracy2(label_database, folder_classes, small_distances, k)
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
            ax[cont2,0].set_ylabel(str(accu_[cont2][0]) + ' \n' + str(accu_[cont2][1]),fontsize = 6)
            #ax[cont2,0].set_ylabel('Input '+str(cont2+1),fontsize = 8)
            cont += 1
            #for each retrieval image returns the k nearer images
            for j in range(k):
                ax[cont2,j+1].imshow(images_database[int(small_distances[cont2,j])],cmap='gray',interpolation = 'none')
                ax[cont2,j+1].set_adjustable('box-forced')
                ax[cont2,j+1].set_yticks([])
                ax[cont2,j+1].set_xticks([])

                shortName = fname_database[int(small_distances[cont2,j])]
                shortName = shortName.split('/')[-1]
                shortName = shortName[0:6]
                ax[cont2,j+1].set_title(shortName,fontsize=8)
                cont += 1
        #plt.show()
        fig.savefig(path_database + "results/result" + "_" + feature_extraction_method + "_" + distance + "_" + str(k) + ".png")   # save the figure to file
        #plt.close(fig)
    else:
        ax[0].imshow(images_retrieval[0],cmap='gray',interpolation = 'none')
        ax[0].set_adjustable('box-forced')
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        ax[0].set_ylabel('Input ',fontsize = 8)
        #for each retrieval image returns the k nearer images
        for j in range(k):
            ax[j+1].imshow(images_database[int(small_distances[0,j])],cmap='gray',interpolation = 'none')
            ax[j+1].set_adjustable('box-forced')
            ax[j+1].set_yticks([])
            ax[j+1].set_xticks([])
            #ax[j+1].set_title('Result: '+str(j+1),fontsize=8)
            shortName = fname_database[int(small_distances[0,j])]
            shortName = shortName.split('/')[-1]
            shortName = shortName[0:6]
            ax[j+1].set_title(shortName,fontsize=8)
    #plt.show()
    file = path_database + "results/result" + "_" + feature_extraction_method + "_" + distance + "_" + str(k) + ".png"
    fig.savefig(file)   # save the figure to file
    #os.system(file)

#retorna uma lista, cada coluna possui o nome da classe e acuracia para aquela classe. Cada linha representa uma imagem
def compute_accuracy2(label_database, folders_classes,result,number_of_images):
    list_result = []
    name_classes = []
    for i in folders_classes:
        str_ = i.split("/")
        name_classes.append(str_[-2])
        
    acc = np.zeros((result.shape[0],len(folders_classes)),np.double)
    for i in range(result.shape[0]):
        for j in range(number_of_images):
            acc[int(i),int(label_database[int(result[int(i),int(j)])])] = acc[int(i),int(label_database[int(result[int(i),int(j)])])] + 1
    acc = (acc/number_of_images)*100
    
    for i in range(acc.shape[0]):
        list_result.append([np.max(acc[i,:]), name_classes[np.argmax(acc[i,:])]])
    return list_result

def get_extension(folders):
    extension = ["jpg", "JPG","jpeg","JPEG", "tif","TIF", "bmp", "BMP", "png", "PNG"]#extension that the system accept
    for i in folders:
        for ext in extension: 
            numb = len(glob(i + "/*." + ext))
            if(numb > 0):
                return ext

    
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

    # this is a particular case of cnn to feature extraction inception
    if feature_extraction_method == 'cnn' or feature_extraction_method == 'cnn_probability':
        if not ("." + image_format) in path_retrieval:
            #ims_database,fname_database,feature_vectors_database,ims_retrieval,_,feature_vectors_retrieval = CNN_feature_extraction.cnn_features_extraction_probability(path_database, path_retrieval, 0, image_format,feature_extraction_method,1,list_of_parameters)
            ims_database,fname_database,feature_vectors_database,ims_retrieval,_,feature_vectors_retrieval = CNN_feature_extraction.cnn_features_extraction_using_tensorFlow(path_database, path_retrieval, path_cnn_trained, image_format,feature_extraction_method)  
         
        else:#O retrieval para uma imagem ainda esta usando o theano... 
            ims_database,fname_database,feature_vectors_database,ims_retrieval,_,feature_vectors_retrieval = CNN_feature_extraction.cnn_features_extraction_probability(path_database, path_retrieval, -1, image_format,feature_extraction_method,1,list_of_parameters)
    
    #training the CNN with the database
    elif feature_extraction_method == 'cnn_training' or feature_extraction_method == 'cnn_probability_training':
        if not ("." + image_format) in path_retrieval:
                
            parameters = Parameters(256 ,path_database,folders,image_format, path_database + "database/",list_of_parameters)
            
            #calling the training process
            if(parameters.NUM_EPOCHS > 0):
                ims_database,fname_database = train_cnn_tensorFlow.train(parameters)
            
            #calling the extraction of features for the database images
            feature_vectors_database, ims_database, fname_database = image_feature_extraction_tensorFlow.features_extraction(parameters)
            
            #calling the extraction of features for the retrieval images
            parameters.PATH_TEST = path_retrieval
            parameters.CLASSES = []
            feature_vectors_retrieval,ims_retrieval,_ = image_feature_extraction_tensorFlow.features_extraction(parameters)
            feature_vectors_retrieval = feature_vectors_retrieval[:,:-1]
            
            #funcionando chamando a rede antigo do theano
            #ims_database,fname_database,feature_vectors_database,ims_retrieval,_,feature_vectors_retrieval = CNN_feature_extraction.cnn_features_extraction_probability(path_database, path_retrieval, 0, image_format,feature_extraction_method,1,list_of_parameters)
            #funcionando chamando a rede antiga do tensorFlow
            #ims_database,fname_database,feature_vectors_database,ims_retrieval,_,feature_vectors_retrieval = CNN_feature_extraction.cnn_training_using_tensorFlow(path_database, path_retrieval, path_cnn_trained, image_format,feature_extraction_method,list_of_parameters)  
            
        else:#O retrieval para uma imagem ainda esta usando o theano... 
            ims_database,fname_database,feature_vectors_database,ims_retrieval,_,feature_vectors_retrieval = CNN_feature_extraction.cnn_features_extraction_probability(path_database, path_retrieval, -1, image_format,feature_extraction_method,1,list_of_parameters)

    else:
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
            ims_database = []
            cont = 0
            for f in folders:
                a = glob(f+'*.'+image_format)
                for i in range(len(a)):
                    fname_database.append(a[i])
                    file = imread(a[i])
                    ims_database.append(file)
        
        else:
            #get the number of features
            number_of_features = get_number_of_features(folders, image_format,feature_extraction_method,list_of_parameters)
            
            #computing features for the database
            ims_database,fname_database,feature_vectors_database = descriptor_all_database(path_database+'database/', folders, image_format,feature_extraction_method,number_of_features,list_of_parameters)
            np.savetxt(file, feature_vectors_database,delimiter = ',')
                    
        #get the number of features
        number_of_features = get_number_of_features(folders, image_format,feature_extraction_method,list_of_parameters)
        #computing features for the retrieval image(s)
        if not ("." + image_format) in path_retrieval:
            ims_retrieval,_,feature_vectors_retrieval = descriptor_all_database(path_retrieval, [], image_format,feature_extraction_method,number_of_features,list_of_parameters)
        else:
            print(path_retrieval)
            ims_retrieval,_,feature_vectors_retrieval = descriptor_all_database(path_retrieval, -1, image_format,feature_extraction_method,number_of_features,list_of_parameters)

    #compute the ranked outputs
    result = searching(feature_vectors_database[:,:-1], feature_vectors_database[:,-1],feature_vectors_retrieval, similarity_metric=distance)    
    #show the ranked output
    show_retrieval(ims_database, ims_retrieval, result, number_of_images, path_database, feature_extraction_method, distance, fname_database, feature_vectors_database[:,-1], folders)
    
    
'''
path_database = "/Users/romuere/Dropbox/CBIR/fibers/"
#path_retrieval = ''
path_retrieval = "/Users/romuere/Dropbox/CBIR/fibers/retrieval/"
path_image = ""
#path_image = "/Users/romuere/Dropbox/CBIR/cifar/retrieval/accentor_s_000194.png"
extension_classes = ["png", "png"]

feature_extraction_method = 'fotf'
distance = "ed"
number_of_images = 10
list_of_parameters = []

run_command_line(path_database,path_retrieval,[],feature_extraction_method,distance,number_of_images,list_of_parameters)
'''
    
    
'''
path_database = "/Users/flavio/Dropbox/Compartilhadas/Romuere/CBIR/fibers/database/"
#path_database = "/Users/flavio/Dropbox/Compartilhadas/Romuere/CBIR/cifarDANI/database/"

#path_retrieval = "/Users/flavio/Dropbox/Compartilhadas/Romuere/CBIR/fibers/retrieval/"

path_image = ""
path_retrieval = "/Users/flavio/Dropbox/Compartilhadas/Romuere/CBIR/fibers/retrieval/"
#path_retrieval = "/Users/flavio/Dropbox/Compartilhadas/Romuere/CBIR/cifarDANI/retrieval/"

extension_classes = ["tif", "tif"]
#extension_classes = ["png", "png"]

feature_extraction_method = 'cnn_probability'
distance = "ED"
number_of_images = 10

run(path_database,path_retrieval,path_image,extension_classes,feature_extraction_method,distance,number_of_images)
'''
