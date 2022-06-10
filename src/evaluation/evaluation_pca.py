'''
Created on 24 de out de 2016

@author: romuere
'''
from sklearn.neighbors import LSHForest
import os
import pickle 
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

def searching(feature_vectors_database,feature_vectors_retrieval, labels_database, retrieval_number):
    #using LSH
    #name to save the pickle file
    
    lshf = LSHForest()
    lshf.fit(feature_vectors_database)  
                     
    # Find closests pair for the first N points
    small_distances = []
    for id1,query in enumerate(feature_vectors_retrieval):
        _, indices = lshf.kneighbors(query, n_neighbors=retrieval_number)
        small_distances.append(indices[0])
                         
    result_labels = [] #labels
    for i in range(len(small_distances)):
        aux2 = []
        for k in range(retrieval_number):
            aux2.append(labels_database[small_distances[i][k]])
        result_labels.append(aux2)

    return result_labels


def precision_top_k_and_Map(labels_query,retrieval_images_names,K,jump):
    
    retrieval_images_accuracy = np.zeros((len(retrieval_images_names),len(retrieval_images_names[0])))
    MAP = np.zeros(len(retrieval_images_names))
    cont_row = 0
    
    for single_query_retrieval in retrieval_images_names:
        cont_column = 0
        cont_sucess = 0.0
        cont_total = 0.0
        sum_sucess = 0.0    
        for label_position in single_query_retrieval:
            cont_total+=1
            if(labels_query[cont_row] == label_position):
                cont_sucess+=1
                sum_sucess += np.double(np.double(cont_sucess)/np.double(cont_total))
                retrieval_images_accuracy[cont_row,cont_column] = 1
            cont_column+=1
            
        if(cont_sucess != 0):
            MAP[cont_row] = np.double(np.double(sum_sucess)/np.double(cont_sucess))
        else: 
            MAP[cont_row] = 0
    
        cont_row+=1
        
    #Each row represents an image and each column represents a k_
    average_precision_per_image = np.zeros((len(retrieval_images_names),len(K)))
    #Each column contains the average precision for a value of K
    average_precision_per_k = np.zeros(len(K))
    cont_index = 0
    for k_ in K:    
        average_precision_per_image[:,cont_index] = np.double(np.sum(retrieval_images_accuracy[:,0:k_],axis=1)/k_)
        average_precision_per_k[cont_index] = np.double(np.mean(average_precision_per_image[:,cont_index]))
        cont_index+=1
    
    return average_precision_per_image, average_precision_per_k, np.mean(np.asarray(MAP)),np.std(np.asarray(MAP))


def pca(train, test, n_comp):
    """
    PCA transformation for using a 'training' set and a 'testing' set
    """
    pca = PCA(n_components=n_comp)
    pca.fit(train[:,:-1],train[-1])
    transform = pca.transform(test[:,:-1])
    labels = test[:,-1].reshape(-1,1)
    return np.concatenate((transform,labels),axis=1)

def sets(file_train,label_train,file_test,label_test):
    
    reader = csv.reader(open(file_train),delimiter=',')
    x = list(reader)
    train = np.array(x).astype(dtype = np.float64)
    
    reader = csv.reader(open(label_train))
    x = list(reader)
    label = np.array(x).astype(dtype = np.uint8)
    
    train = np.concatenate((train,label),axis = 1)
    
    reader = csv.reader(open(file_test),delimiter=',')
    x = list(reader)
    test = np.array(x).astype(dtype = np.float64)
    
    reader = csv.reader(open(label_test))
    x = list(reader)
    label = np.array(x).astype(np.uint8)
    
    test = np.concatenate((test,label),axis = 1)
    
    
    return train,test
    
def main():
    
    file_train = '/Users/romuere/Dropbox/CBIR/pca/train.csv'
    label_train = '/Users/romuere/Dropbox/CBIR/pca/labels_training.csv'
    file_test = '/Users/romuere/Dropbox/CBIR/pca/test.csv'
    label_test = '/Users/romuere/Dropbox/CBIR/pca/labels_test.csv'
    train,test_ = sets(file_train,label_train,file_test,label_test)
    n_features = train.shape[1]-1
    
    """
    jump = 1
    labels = test_[:,-1].astype(dtype = np.uint16)
    query = test_[labels == 1,:]
    k = range(0,query.shape[0],jump)
    result = searching(test_, query, labels, query.shape[0])
    _, _, map_class,_ = precision_top_k_and_Map(query[:,-1], result,k,jump)
    
    print(map_class)
    """
    
    
    result = np.zeros(n_features)
    for nf in range(1,n_features+1):
        test = pca(train, test_, nf)
        n_class = np.unique(test[:,-1])
        labels = test[:,-1].astype(dtype = np.uint16)
        #test = test[:,:-1]
        map_class = np.zeros(len(n_class))
        for id_nc,nc in enumerate(n_class):
            jump = 1
            query = test[labels == nc,:]
            k = range(0,query.shape[0],jump)
            r = searching(test[:,:-1], query[:,:-1], labels, query.shape[0])
            _, _, map_class[id_nc],_ = precision_top_k_and_Map(query[:,-1], r,k,jump)
        result[nf] = np.mean(map_class)
        print('Component: ',nf)
        print('MAP: ',result[nf]) 
    
    fig, ax = plt.subplots(nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(result)
    
    plt.xlabel('Number of Components')
    plt.ylabel('MAP')
    plt.ylim(0,1.1)
    plt.show()
    
    return result
    #print('MAP normal: ',MAP_class0,'+-',MAP_class0_std,'\nMAP abnormal: ',MAP_class1,'+-',MAP_class1_std)
    
a = main()