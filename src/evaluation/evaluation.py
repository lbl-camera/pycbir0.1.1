'''
Created on Oct 5, 2016

@author: flavio
'''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.run import run_command_line
from src.util import convert_database_to_files
import random

def accuracy(labels_query,retrieved_image_labels,k_accuracy):

    classes = np.unique(retrieved_image_labels[1])

    accuracy_per_class = np.zeros(len(classes))
    cont_row = 0
    cont_sucess_total = 0
    for single_query_label in retrieved_image_labels[1]:
        single_query_label = single_query_label[1:k_accuracy+1]#starts in 1 to remove the first image 
        cont_class = 0
        for class_ in classes:
            accuracy_per_class[cont_class] = np.sum(single_query_label[0:k_accuracy] == class_)
            cont_class+=1

        cont_sucess_query = 0
        for l in single_query_label:
            if(l == labels_query[cont_row]):
                cont_sucess_query+=1

        if(cont_sucess_query >= np.max(accuracy_per_class)):
            cont_sucess_total+=1
        cont_row+=1
    return (cont_sucess_total/cont_row)

def precision_top_k_and_Map(labels_query, retrieved_image_labels,K,jump):
    
    retrieval_images_accuracy = np.zeros((len(retrieved_image_labels[0]),len(retrieved_image_labels[0][0])))
    MAP = np.zeros(len(retrieved_image_labels[0]))
    cont_row = 0
    
    for single_query_retrieval in retrieved_image_labels[1]:
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
    average_precision_per_image = np.zeros((len(retrieved_image_labels[0]),len(K)))
    #Each column contains the average precision for a value of K
    average_precision_per_k = np.zeros(len(K))
    cont_index = 0
    for k_ in K:    
        average_precision_per_image[:,cont_index] = np.double(np.sum(retrieval_images_accuracy[:,0:k_],axis=1)/k_)
        average_precision_per_k[cont_index] = np.double(np.mean(average_precision_per_image[:,cont_index]))
        cont_index+=1
    
    return average_precision_per_image, average_precision_per_k, np.mean(np.asarray(MAP))

def get_name_percent_query(name_images_query,labels_query,class_,percent_query):
    
    #if(percent_query == 1):
    #    return name_images_query,labels_query
    
    names = [name for p,name in enumerate(name_images_query) if labels_query[p] == class_]
    random.shuffle(names)
    names = names[0:np.int(len(names)*percent_query)]
    
    labels = np.zeros(len(names))
    labels[:] = class_
    
    return names,labels 

def get_name_percent_database(name_images_database,labels_database,classes,percent_database):
    name_images_database_percent = []
    labels_database_percent = []
    
    #if(percent_database == 1):
    #    return name_images_database,labels_database
        
    for class_ in classes:  
        names = [name for p,name in enumerate(name_images_database) if labels_database[p] == class_]
        random.shuffle(names)
        names = names[0:np.int(len(names)*percent_database)]
        
        labels = np.zeros(len(names))
        labels[:] = class_
        
        name_images_database_percent.extend(names)
        labels_database_percent.extend(labels)
    
    return name_images_database_percent,labels_database_percent 

def evaluation(name_images_database, labels_database, name_images_query, labels_query,path_output,feature_extraction_method,distance,list_of_parameters,preprocessing_method,searching_method,percent_query=1,percent_database=1,path_cnn_pre_trained = '',path_save_cnn = '',jump = 10,k_accuracy=10):
    
    classes = np.unique(labels_database)
    name_images_database, labels_database = get_name_percent_database(name_images_database,labels_database,classes,percent_database)
    number_of_images = len(name_images_database)
    
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    average = []
    MAP = np.zeros(len(classes))
    ACCURACY = np.zeros(len(classes))
    cont_index = 0
    
    #remove_files(path_database, feature_extraction_method, distance, list_of_parameters)
    for class_ in classes:     
        print('class =',class_)
        k = range(0,number_of_images,jump) # from 0 to the number of images in the class
        
        #get the image names of the class 'class_'
        name_images_query_class_,label_class_ = get_name_percent_query(name_images_query, labels_query,class_,percent_query)
        
        retrieved_image_labels, _, _ = run_command_line(name_images_database,labels_database,name_images_query_class_,label_class_,path_cnn_pre_trained,path_save_cnn,path_output,feature_extraction_method,distance,number_of_images,list_of_parameters,preprocessing_method,searching_method,isEvaluation=True,do_searching_processing=True,save_csv=False)
                                    
        _, average_precision_per_k, MAP[cont_index] = precision_top_k_and_Map(label_class_, retrieved_image_labels,k,jump)
        ACCURACY[cont_index] = accuracy(label_class_,retrieved_image_labels,k_accuracy)
                 
        average.append(average_precision_per_k)
          
        cont_index+=1
        
        ax.plot(k,average_precision_per_k,label=class_)    
    #remove_files(path_database, feature_extraction_method, distance, list_of_parameters)

    plt.xlabel('Number of Retrieved Images')
    plt.ylabel('Average Precision')
    plt.legend(loc='best')
    plt.ylim(0,1.1)
        
    return MAP, ACCURACY, fig

def get_accuracy_using_list_k_accuracy(name_images_database, labels_database, name_images_query, labels_query,path_output,feature_extraction_method,distance,list_of_parameters,preprocessing_method,searching_method,list_k_accuracy,percent_query=1,percent_database=1,path_cnn_trained = '',jump = 10):
    
    classes = np.unique(labels_database)
    name_images_database, labels_database = get_name_percent_database(name_images_database,labels_database,classes,percent_database)
    number_of_images = len(name_images_database)
    
    list_accuracy = []
    for class_ in classes:             
        #get the image names of the class 'class_'
        name_images_query_class_,label_class_ = get_name_percent_query(name_images_query, labels_query,class_,percent_query)
        
        retrieved_image_labels, _, _ = run.run_command_line(name_images_database,labels_database,name_images_query_class_,label_class_,path_cnn_trained,path_output,feature_extraction_method,distance,number_of_images,list_of_parameters,preprocessing_method,searching_method,isEvaluation=True,do_searching_processing=True,save_csv=False)
        
        ACCURACY = np.zeros(len(list_k_accuracy))
        cont_index = 0
        for k_accuracy in list_k_accuracy:
            ACCURACY[cont_index] = accuracy(label_class_,retrieved_image_labels,k_accuracy)
            print('k =', k_accuracy, 'classe =',class_,'accuracy =',ACCURACY[cont_index])
            cont_index+=1
        list_accuracy.append(ACCURACY)
    
    list_accuracy = np.asarray(list_accuracy)
    list_accuracy = np.transpose(list_accuracy)
    return list_accuracy

def get_time_first_second_execution(name_images_database, labels_database, name_images_query, labels_query,path_cnn_trained,path_output,feature_extraction_method,distance,list_of_parameters,preprocessing_method,searching_method_all,percent_database=1):
    
    classes = np.unique(labels_database)
    name_images_database, labels_database = get_name_percent_database(name_images_database,labels_database,classes,percent_database)
    
    number_of_images = len(name_images_database)
    
    time_of_extraction_features = np.zeros((len(searching_method_all),2))
    time_of_retrieval_images = np.zeros((len(searching_method_all),2))
    cont = 0
    for searching_method in searching_method_all:     
        #convert_database_to_files.remove_files(path_output, feature_extraction_method, distance, list_of_parameters)
                
        #time for the first execution
        _, [_, time_of_extraction_features[cont,0]], time_of_retrieval_images[cont,0] = run.run_command_line(name_images_database,labels_database,name_images_query,labels_query,path_cnn_trained,path_output,feature_extraction_method,distance,number_of_images,list_of_parameters,preprocessing_method,searching_method,isEvaluation=True,do_searching_processing=True,save_csv=True)
        #time for the second execution
        _, [_, time_of_extraction_features[cont,1]], time_of_retrieval_images[cont,1] = run.run_command_line(name_images_database,labels_database,name_images_query,labels_query,path_cnn_trained,path_output,feature_extraction_method,distance,number_of_images,list_of_parameters,preprocessing_method,searching_method,isEvaluation=True,do_searching_processing=True,save_csv=True)
        
        time_of_retrieval_images[cont,0] = time_of_retrieval_images[cont,0]/len(name_images_query)
        time_of_retrieval_images[cont,1] = time_of_retrieval_images[cont,1]/len(name_images_query)
        
        cont+=1
    #convert_database_to_files.remove_files(path_output, feature_extraction_method, distance, list_of_parameters)
    
    return time_of_extraction_features, time_of_retrieval_images
