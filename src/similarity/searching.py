'''
Created on 15 de sep de 2016
Last Modified on 26 de set de 2019

@author: romuere, dani
'''
import numpy as np
from src.similarity.similarity_metrics import similarity_metrics
import pickle
from scipy.spatial import ckdtree
from sklearn.cluster import KMeans
import os
import sys
sys.setrecursionlimit(1000000)
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings("ignore")


def searching(feature_vectors_database,feature_vectors_retrieval, labels_database,
              fname_database, similarity_metric, retrieval_number, list_of_parameters,
              feature_extraction_method,path_output,searching_method):
    """
    This is the function to search for similar objects.

    Parameters
    ----------
    feature_vector_database: numpy array
        It is an matrix and each line corresponds to a signature of a database image.
    feature_vector_retrieval : numpy array
        It is an matrix and each line corresponds to a signature of a retrieval image.
    labels_database : numpy array
        It is the labels of all database images
    fname_database : list of str
        Each position corresponds to the complete path of an image in the database, in the same order of the
        'feature_vector_database'.
    similarity_metric : str
        Similarity metric that follows the README specification
    retrieval_number : int
        Number of images to retrieve
    list_of_parameters : list
        Parameters for the feature extraction method chosen
    feature_extraction_method : str
        Feature extraction method that follows the README specification
    path_output : str
        Complete path of the database folder
    searching_method : str
        Method to be used to search the similar signatures (README)

    Returns
    -------
    result : list
        Each position correspond to an query image, and for each query image it
        has a list of str. Each str is the complete path of a retrieval image.
        This result is already sorted.
    """

    if searching_method == 'bf':
        return searching_bruteForce(feature_vectors_database,feature_vectors_retrieval,
                                    labels_database, fname_database, similarity_metric,
                                    retrieval_number, list_of_parameters, feature_extraction_method,
                                    path_output,searching_method)
    elif searching_method == 'rt':
        return searching_RTree(feature_vectors_database,feature_vectors_retrieval,
                               labels_database, fname_database, similarity_metric,
                               retrieval_number, list_of_parameters, feature_extraction_method,
                               path_output,searching_method)
    elif searching_method == 'kd':
        return searching_KDTree(feature_vectors_database,feature_vectors_retrieval,
                                labels_database, fname_database, similarity_metric,
                                retrieval_number, list_of_parameters, feature_extraction_method,
                                path_output,searching_method)
    elif searching_method == 'bt':
        return searching_BallTree(feature_vectors_database,feature_vectors_retrieval,
                                    labels_database, fname_database, similarity_metric,
                                    retrieval_number, list_of_parameters, feature_extraction_method,
                                    path_output,searching_method)

    elif searching_method == 'km':
        return searching_KMeans(feature_vectors_database,feature_vectors_retrieval,
                                labels_database, fname_database, similarity_metric,
                                retrieval_number, list_of_parameters, feature_extraction_method,
                                path_output,searching_method)


def searching_bruteForce(feature_vectors_database,feature_vectors_retrieval, labels_database,
                         fname_database, similarity_metric, retrieval_number, list_of_parameters,
                         feature_extraction_method,path_output,searching_method):

    n = len(feature_vectors_retrieval)
    shape = feature_vectors_database.shape[0]
    #computing the distance between retrieval images and all database
    distances = np.zeros((n,shape),dtype = np.float32)
    for cont1,i in enumerate(feature_vectors_retrieval):
        for cont2,j in enumerate(feature_vectors_database):
            distances[cont1,cont2] = similarity_metrics(i, j, med = similarity_metric)
    #the values are sorted by the less to bigger distances and returns the index
    #np.savetxt('/Users/romuere/Desktop/distances_brute_force.csv', distances)
    """
    small_distances = np.zeros((n,shape),dtype = np.uint32)
    for cont,d in enumerate(distances):
        small_distances[cont,:] = [i[0] for i in sorted(enumerate(d), key=lambda x:x[1])]
    """
    # Find closests pair for the first N points

    batch = 1000
    small_distances = []
    cont_batch = 0
    cont_n_batch = 0 #number of batch files
    for id1,d in enumerate(distances):
        small_distances.append([i[0] for i in sorted(enumerate(d), key=lambda x:x[1])])
        cont_batch += 1
        if (cont_batch == batch) or (cont_batch == (len(feature_vectors_retrieval)-cont_n_batch*batch)):
            small_distances = np.asarray(small_distances)
            np.save(path_output+'batch_'+str(cont_n_batch), small_distances)
            small_distances = []
            cont_batch = 0
            cont_n_batch += 1

    result_filenames = [] #file names
    result_labels = [] #labels
    for l in range(cont_n_batch):
        file = path_output+'batch_'+str(l)+'.npy'
        small_distances = np.load(file)
        os.remove(file)
        for i in range(len(small_distances)):
            aux1 = []
            aux2 = []
            for k in range(retrieval_number):
                aux1.append(fname_database[small_distances[i][k]])
                aux2.append(labels_database[small_distances[i][k]])
            result_filenames.append(aux1)
            result_labels.append(aux2)

    return (result_filenames,result_labels)

def searching_RTree(feature_vectors_database,feature_vectors_retrieval, labels_database,
                    fname_database, similarity_metric, retrieval_number, list_of_parameters,
                    feature_extraction_method,path_output,searching_method):

    #name to save the pickle file
    parameters_name = ""
    if not(feature_extraction_method == 'cnn' or feature_extraction_method == 'cnn_training'):
        for parameter in list_of_parameters:
            parameters_name = parameters_name + "_" + parameter

    file = path_output + "RTree" + "_" + feature_extraction_method + parameters_name +'_'+similarity_metric

    #feature_vectors_retrieval = preprocessing.scale(feature_vectors_retrieval)

    if not(os.path.isfile(file)):

        tree = ckdtree.cKDTree(feature_vectors_database,leafsize=15048)

        with open(file, 'wb') as handle:
            pickle.dump(tree, handle)
    else:
        with open(file, 'rb') as handle:
            tree = pickle.load(handle)

    # Find closests pair for the first N points
    ########### debug this part ###########
    small_distances = []
    for id1,query in enumerate(feature_vectors_retrieval):
        nearest = list(idx.nearest(query.tolist(), retrieval_number))
        small_distances.append(nearest)

    result_filenames = [] #file names
    result_labels = [] #labels
    for cont1,i in enumerate(small_distances):
        aux1 = []
        aux2 = []
        for j in range(retrieval_number):
            aux1.append(fname_database[small_distances[cont1][j]])
            aux2.append(labels_database[small_distances[cont1][j]])
        result_filenames.append(aux1)
        result_labels.append(aux2)

    return (result_filenames,result_labels)

def searching_KDTree(feature_vectors_database,feature_vectors_retrieval, labels_database,
                     fname_database, similarity_metric, retrieval_number, list_of_parameters,
                     feature_extraction_method,path_output,searching_method):

    #name to save the pickle file
    parameters_name = ""
    if not(feature_extraction_method == 'cnn' or feature_extraction_method == 'cnn_training'):
        for parameter in list_of_parameters:
            parameters_name = parameters_name + "_" + parameter

    file = path_output + "KDTree" + "_" + feature_extraction_method + parameters_name +'_'+similarity_metric+'.pickle'



    #feature_vectors_retrieval = preprocessing.scale(feature_vectors_retrieval)

    if not(os.path.isfile(file)):

        #normalize signatures
        #feature_vectors_database = preprocessing.scale(feature_vectors_database)

        #kdtree.node = ckdtree.cKDTree.node
        #kdtree.leafnode = ckdtree.cKDTree.leafnode
        #kdtree.innernode = ckdtree.cKDTree.innernode
        tree = ckdtree.cKDTree(feature_vectors_database,leafsize=15048)

        with open(file, 'wb') as handle:
            pickle.dump(tree, handle)
    else:
        with open(file, 'rb') as handle:
            tree = pickle.load(handle)

    # Find closests pair for the first N points
    small_distances = []
    for id1,query in enumerate(feature_vectors_retrieval):
        _,nearest = tree.query(query, retrieval_number)
        small_distances.append(nearest.tolist())

    result_filenames = [] #file names
    result_labels = [] #labels
    for cont1,i in enumerate(small_distances):
        aux1 = []
        aux2 = []
        for j in range(retrieval_number):
            aux1.append(fname_database[small_distances[cont1][j]])
            aux2.append(labels_database[small_distances[cont1][j]])
        result_filenames.append(aux1)
        result_labels.append(aux2)

    return (result_filenames,result_labels)

#this function is deprecated
def searching_BallTree(feature_vectors_database,feature_vectors_retrieval, labels_database,
                  fname_database, similarity_metric, retrieval_number, list_of_parameters,
                  feature_extraction_method,path_output,searching_method):

    #name to save the pickle file
    parameters_name = ""
    if not(feature_extraction_method == 'cnn' or feature_extraction_method == 'cnn_training'):
        for parameter in list_of_parameters:
            parameters_name = parameters_name + "_" + parameter

    file = path_output + "BallTree_" + feature_extraction_method + parameters_name +'_'+similarity_metric+'.pickle'

    if not(os.path.isfile(file)):

        bt = BallTree(feature_vectors_database)
        
        with open(file, 'wb') as handle:
            pickle.dump(bt,handle)
    else:
        with open(file, 'rb') as handle:
            bt = pickle.load(handle)

    # Find closests pair for the first N points

    _,small_distances = bt.query(feature_vectors_retrieval, retrieval_number)
    result_filenames = [] #file names
    result_labels = [] #labels
    for cont1,i in enumerate(small_distances):
        aux1 = []
        aux2 = []
        for j in range(retrieval_number):
            aux1.append(fname_database[small_distances[cont1][j]])
            aux2.append(labels_database[small_distances[cont1][j]])
        result_filenames.append(aux1)
        result_labels.append(aux2)

    return (result_filenames,result_labels)

    """
    batch = 1000
    small_distances = []
    cont_batch = 0
    cont_n_batch = 0 #number of batch files
    for id1,query in enumerate(feature_vectors_retrieval):
        _, indices = lshf.kneighbors(query, n_neighbors=retrieval_number)
        small_distances.append(indices[0])
        print(id1)
        cont_batch += 1
        if (cont_batch == batch) or (cont_batch == (len(feature_vectors_retrieval)-cont_n_batch*batch)):
            small_distances = np.asarray(small_distances)
            np.save(path_output+'batch_'+str(cont_n_batch), small_distances)
            small_distances = []
            cont_batch = 0
            cont_n_batch += 1


    result_filenames = [] #file names
    result_labels = [] #labels
    for l in range(cont_n_batch):
        file = path_output+'batch_'+str(l)+'.npy'
        small_distances = np.load(file)
        os.remove(file)
        for i in range(len(small_distances)):
            aux1 = []
            aux2 = []
            for k in range(retrieval_number):
                aux1.append(fname_database[small_distances[i][k]])
                aux2.append(labels_database[small_distances[i][k]])
            result_filenames.append(aux1)
            result_labels.append(aux2)


    return (result_filenames,result_labels)
    """

def searching_KMeans(feature_vectors_database,feature_vectors_retrieval, labels_database,
                     fname_database, similarity_metric, retrieval_number, list_of_parameters,
                     feature_extraction_method,path_output,searching_method):

    '''
    feature_vectors: atriutos calculados
    labels: label de cada classe
    similarity_metric: qual medida utilizar
    recuperados as k imagens com menor distancia. Se k = 0, entao o valor eh
    setado como sendo o tamanho da classe da imagem
    '''
    nClusters = 10 #question: how to compute this value

    #name to save the pickle file
    parameters_name = ""
    if not(feature_extraction_method == 'cnn' or feature_extraction_method == 'cnn_training'):
        for parameter in list_of_parameters:
            parameters_name = parameters_name + "_" + parameter

    file = path_output + "indexing" + "_" + feature_extraction_method + parameters_name +'_'
    +similarity_metric+'_'+str(retrieval_number)+'_nClusters_'+str(nClusters)+'.pickle'


    #feature_vectors_retrieval = preprocessing.scale(feature_vectors_retrieval)
    if not(os.path.isfile(file)):

        #normalize signatures
        #feature_vectors_database = preprocessing.scale(feature_vectors_database)

        nClusters = 20 #question: how to compute this value
        k = KMeans(n_clusters=nClusters)
        pred = k.fit_predict(feature_vectors_database)
        centers = k.cluster_centers_

        #create a list of lists with the signatures of each cluster, each position has all signatures of one cluster
        lists_of_signatures = []
        lists_of_image_paths = []
        for i in range(nClusters):
            list_aux_signatures = []
            list_aux_paths = []
            cluster = (pred == i)
            position = np.where(cluster == True)
            for j in position[0]:
                list_aux_signatures.append(feature_vectors_database[j,:])
                list_aux_paths.append(image_paths[j])
            lists_of_signatures.append(list_aux_signatures)
            lists_of_image_paths.append(list_aux_paths)

        with open(file, 'wb') as handle:
            pickle.dump([centers,lists_of_signatures,lists_of_image_paths], handle)
    else:
        with open(file, 'rb') as handle:
            centers,lists_of_signatures,lists_of_image_paths = pickle.load(handle)

    #compare the retrieval signature with the centers
    retrieval_distances = []
    retrieval_paths = []
    paths_cluster = np.zeros((len(feature_vectors_retrieval)))
    for cont1,i in enumerate(feature_vectors_retrieval):

        #look for the minimal distance between a query image and the n k-means centers
        distances_centers = []
        for j in centers:
            distances_centers.append(similarity_metrics(i, j, med = 'ed'))
        minimal_distance_center = distances_centers.index(min(distances_centers))

        #create two arrays to storage the distances between the query image and the feature vectors,
        #the second vector 'small_distance' has the 'distances' sorted
        distances = np.zeros((len(lists_of_signatures[minimal_distance_center])),dtype = np.float32)
        small_distances = np.zeros((len(lists_of_signatures[minimal_distance_center])))

        #save the cluster to look for the image path
        paths_cluster[cont1] = minimal_distance_center

        #fill the vectors with the distances
        for cont2,k in enumerate(lists_of_signatures[minimal_distance_center]):
            distances[cont2] = similarity_metrics(i, k, med = similarity_metric)
        small_distances[:] = [l[0] for l in sorted(enumerate(distances), key=lambda x:x[1])]
        retrieval_distances.append(small_distances)

    result_filenames = [] #file names
    result_labels = [] #labels
    for cont1,i in enumerate(small_distances):
        aux1 = []
        aux2 = []
        for j in range(retrieval_number):
            aux1.append(fname_database[small_distances[cont1,j]])
            aux2.append(labels_database[small_distances[cont1,j]])
        result_filenames.append(aux1)
        result_labels.append(aux2)

    return (result_filenames,result_labels)


def teste_function():
    import csv
    import itertools

    path = 'feature_vectors_database.csv'
    reader = csv.reader(open(path),delimiter=',')
    x = list(reader)
    feature_vectors_database = np.array(x).astype(dtype = np.float64)

    path = 'fname_databases.csv'
    reader = csv.reader(open(path))
    fname_database = list(reader)
    fname_database  = list(itertools.chain(*fname_database))

    path = 'labels.csv'
    reader = csv.reader(open(path),delimiter=',')
    x = list(reader)
    labels_database = np.array(x).astype(dtype = np.uint16)
    labels_database = labels_database.reshape(-1)
    feature_vectors_retrieval = np.concatenate((feature_vectors_database[0:5,:],feature_vectors_database[2001:2006,:]))

    feature_extraction_method = 'fotf'
    searching_method = 'lsh'
    retrieval_number = 10
    similarity_metric = 'ed'
    list_of_parameters = []
    path_output = ''

    result = searching(feature_vectors_database,feature_vectors_retrieval, labels_database,
                       fname_database, similarity_metric, retrieval_number, list_of_parameters,
                       feature_extraction_method,path_output,searching_method)
    a = 1
#teste_function()
