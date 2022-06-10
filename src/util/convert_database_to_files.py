'''
Created on Sep 28, 2016

@author: flavio
'''
import glob
import os
import numpy as np

def remove_files(path_output,feature_extraction_method,distance_method,list_of_parameters):
    
    paths = []
    if(len(list_of_parameters) == 2 or not list_of_parameters):
        if feature_extraction_method == 'cnn_training' or feature_extraction_method == 'cnn' or feature_extraction_method == 'fotf':
            paths.append(path_output + '/feature_vectors_' + feature_extraction_method + '.csv')
            paths.append(path_output + '/image_filenames_' + feature_extraction_method + '.csv')
            paths.append(path_output + '/labels_' + feature_extraction_method + '.csv')
            paths.append(path_output + '/KDTree_' + feature_extraction_method + '_' + distance_method +'.pickle')
            paths.append(path_output + '/RTree_' + feature_extraction_method + '_' + distance_method +'.idx')
            paths.append(path_output + '/RTree_' + feature_extraction_method + '_' + distance_method +'.dat')
            paths.append(path_output + '/LSH_' + feature_extraction_method + '_' + distance_method +'.pickle')
        else:
            paths.append(path_output + '/feature_vectors_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '.csv') 
            paths.append(path_output + '/image_filenames_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '.csv')
            paths.append(path_output + '/labels_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '.csv')
            paths.append(path_output + '/KDTree_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + distance_method +'.pickle')
            paths.append(path_output + '/RTree_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + distance_method +'.idx')
            paths.append(path_output + '/RTree_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + distance_method +'.dat')
            paths.append(path_output + '/LSH_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + distance_method +'.pickle')
    elif(len(list_of_parameters) == 1):
        paths.append(path_output + '/feature_vectors_' + feature_extraction_method + '_' + list_of_parameters[0] + '.csv') 
        paths.append(path_output + '/image_filenames_' + feature_extraction_method + '_' + list_of_parameters[0] + '.csv')
        paths.append(path_output + '/labels_' + feature_extraction_method + '_' + list_of_parameters[0] + '.csv')
        paths.append(path_output + '/KDTree_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + distance_method +'.pickle')
        paths.append(path_output + '/RTree_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + distance_method +'.idx')
        paths.append(path_output + '/RTree_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + distance_method +'.dat')
        paths.append(path_output + '/LSH_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + distance_method +'.pickle')
    elif(len(list_of_parameters) == 3):
        paths.append(path_output + '/feature_vectors_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + list_of_parameters[2] +'.csv') 
        paths.append(path_output + '/image_filenames_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + list_of_parameters[2] + '.csv')
        paths.append(path_output + '/labels_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + list_of_parameters[2] + '.csv')
        paths.append(path_output + '/KDTree_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + list_of_parameters[2] + '_' + distance_method +'.pickle')
        paths.append(path_output + '/RTree_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + list_of_parameters[2] + '_' + distance_method +'.idx')
        paths.append(path_output + '/RTree_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + list_of_parameters[2] + '_' + distance_method +'.dat')
        paths.append(path_output + '/LSH_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + list_of_parameters[2] + '_' + distance_method +'.pickle')
    elif(len(list_of_parameters) == 4):
        paths.append(path_output + '/feature_vectors_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + list_of_parameters[2] + '_' + list_of_parameters[3] +'.csv') 
        paths.append(path_output + '/image_filenames_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + list_of_parameters[2] + '_' + list_of_parameters[3] + '.csv')
        paths.append(path_output + '/labels_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + list_of_parameters[2] + '_' + list_of_parameters[3] + '.csv')
        paths.append(path_output + '/KDTree_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + list_of_parameters[2] + '_' + list_of_parameters[3] + '_' + distance_method +'.pickle')
        paths.append(path_output + '/RTree_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + list_of_parameters[2] + '_' + list_of_parameters[3] + '_' + distance_method +'.idx')
        paths.append(path_output + '/RTree_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + list_of_parameters[2] + '_' + list_of_parameters[3] + '_' + distance_method +'.dat')
        paths.append(path_output + '/LSH_' + feature_extraction_method + '_' + list_of_parameters[0] + '_' + list_of_parameters[1] + '_' + list_of_parameters[2] + '_' + list_of_parameters[3] + '_' + distance_method +'.pickle')

    for p in paths:
        if(os.path.isfile(p)):
            os.remove(p)

def get_image_format(folder_images):
    name_images = os.listdir(folder_images)
    return name_images[1].split('.')[-1]

#remove of the list images that do not exists in the machine
def preprocessing(names, labels):
    names_checked = []
    labels_checked = []
    for i in range(len(names)):
        if(os.path.isfile(names[i])):
            names_checked.append(names[i])
            labels_checked.append(labels[i])
    
    return names_checked, np.asarray(labels_checked)
    
def get_name_labels(path_database,path_retrieval=''):
        
    folders = glob.glob(path_database + '*/')
    image_format = get_image_format(folders[0])
    
    #get the name and labels for the database
    cont = 0
    list_name_images_database = []
    labels_database = []
    for folder in folders:
        name_images = glob.glob(folder + "/*." + image_format)
        labels = np.zeros(len(name_images),dtype=np.int)
        labels[:] = cont
        
        list_name_images_database.extend(name_images)
        labels_database.extend(labels)
        cont+=1


    #get the name and labels for query images
    if( path_retrieval != ''):
        folders = glob.glob(path_retrieval + '*/')
        if len(folders) != 0:
            cont = 0
            list_name_images_query = []
            labels_query = []
            for folder in folders:
                name_images = glob.glob(folder + "/*." + image_format)
                labels = np.zeros(len(name_images), dtype=np.int)
                labels[:] = cont

                list_name_images_query.extend(name_images)
                labels_query.extend(labels)
                cont += 1
            return list_name_images_database, np.asarray(labels_database), list_name_images_query, labels_query
        else:
            list_name_images_query = glob.glob(path_retrieval + "/*." + image_format)
            labels_query = np.zeros(len(list_name_images_query),dtype=np.int)# all labels are 0 for query images
            return list_name_images_database, np.asarray(labels_database), list_name_images_query,labels_query
    else:
        return list_name_images_database, np.asarray(labels_database)