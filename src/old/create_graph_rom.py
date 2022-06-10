'''
Created on Nov 18, 2016

@author: flavio
'''

import util.convert_database_to_files
import evaluation
import run
import numpy as np
import glob
import os

def remove_files_pickle(path_output):
    #name_files = glob.glob(path_output + '*.ckpt')
    name_files = glob.glob(path_output + '*.pickle')
    for name in name_files:
        if(os.path.isfile(name)):
            os.remove(name)
            
def remove_files_cnn(path_output):
    name_files = glob.glob(path_output + '*.ckpt')
    #name_files = glob.glob(path_output + '*.pickle')
    for name in name_files:
        if(os.path.isfile(name)):
            os.remove(name)
    
def run_create_graph():
    #cnn
    path_database_train = '/home/users/romuere/Saxsgen/new_database_split/new_database_split_train/'
    path_database_test = '/home/users/romuere/Saxsgen/new_database_split/new_database_split_test/'
    path_retrieval = '/home/users/flavio/databases/fiberFlaRom/fiberFlaRom_train/query/'
    path_cnn_trained = '/home/users/romuere/Saxsgen/new_database_split/new_database_split_train/features/model.ckpt'
    path_output_train = path_database_train + 'features/'
    path_output_test = path_database_test + 'features/'
    
    preprocessing_method = 'log'
    distance = 'ed'
    searching_method = 'kd'
    percent_database = 1
    percent_query = 1
    number_of_images = 10
    feature_extraction_method = 'cnn_training'
    
    #jump_num_epoch = [1,4,5,10,20,30,30]#cells
    #learning_rate =[0.1,0.1,0.08,0.04,0.02,0.01,0.009]#cells

    #jump_num_epoch = [1,9,10,20,30,30,50,100,100]#fmd
    #learning_rate =[0.1,0.1,0.03,0.02,0.01,0.008,0.004,0.002,0.001]#fmd

    jump_num_epoch = [1,4,5,10,20,30,50,100]#fibers
    learning_rate =[0.1,0.1,0.1,0.1,0.08,0.06,0.04,0.04]#fibers    

    NUM_LEVEL = [0]
    
    for num_level in NUM_LEVEL:
        remove_files_cnn(path_output_train)
    
        list_train_time = []
        list_map = []
        #removing files
        remove_files_pickle(path_output_train)
        cont_index=0
        for num_epoch in jump_num_epoch:
            
            list_of_parameters = [str(learning_rate[cont_index]),str(num_epoch),str(num_level)]          
            
            #train
            #get the list of names and labels
            name_images_database, labels_database, name_images_query, labels_query = convert_database_to_files.get_name_labels(path_database_train,path_retrieval)
            _, train_time, _ = run.run_command_line(name_images_database,labels_database,name_images_query,labels_query,path_cnn_trained,path_output_train,feature_extraction_method,distance,number_of_images,list_of_parameters,preprocessing_method,searching_method, isEvaluation=True,do_searching_processing=False,save_csv=False)
            
            if(not list_train_time):
                list_train_time.append([num_epoch,train_time[0]])
            else:
                list_train_time.append([np.sum(jump_num_epoch[0:cont_index+1]),train_time[0] + list_train_time[-1][1]])
        
            #evaluation
            list_of_parameters = ['0.1','0',str(num_level)]
            name_images_database, labels_database = convert_database_to_files.get_name_labels(path_database_test)
            MAP, fig = evaluation.evaluation(name_images_database, labels_database, name_images_database, labels_database,path_output_test,feature_extraction_method,distance,list_of_parameters,preprocessing_method,searching_method,path_cnn_trained=path_cnn_trained,percent_query=percent_query,percent_database=percent_database)
            list_map.append([np.sum(jump_num_epoch[0:cont_index+1]),np.mean(MAP)])
      
            print('Num_epoch =', np.sum(jump_num_epoch[0:cont_index+1]),'train_time =', list_train_time[-1][1], 'MAP =',np.mean(MAP))
            
            #removing files
            remove_files_pickle(path_output_test)
            cont_index+=1
            
        np.savetxt(path_output_test + feature_extraction_method + '_train_time_' + preprocessing_method + '_' + str(num_level) + '_level' + '.csv', np.asarray(list_train_time),delimiter = ',')
        np.savetxt(path_output_test + feature_extraction_method + '_Map_' + preprocessing_method + '_' + str(num_level) + '_level' + '.csv', np.asarray(list_map),delimiter = ',')

run_create_graph()
        
