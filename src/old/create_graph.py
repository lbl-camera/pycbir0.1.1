'''
Created on Nov 18, 2016

@author: flavio
'''

import util.convert_database_to_files
from evaluation.evaluation import evaluation
import run
import numpy as np
import glob
import os
import math

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
            
def get_number_examples_per_class(labels_database):
    classes = np.unique(labels_database)
    number_per_class = np.zeros(len(classes))
    cont_index=0
    for class_ in classes:
        number_per_class[cont_index] = np.sum(labels_database == class_)
        cont_index+=1
        
    return number_per_class

def tan_sigmoid(x):
    return 2/(1+math.pow(math.e,(-2*x))) -1

def new_learning_new(c_learning,factor_dec,epoch,e_0,c_e,t):
    return (c_learning - (1-tan_sigmoid( (e_0-c_e)/t ))*factor_dec )
    
def run_create_graph_map():
    #cnn machine
    path_database_train = '/home/users/flavio/databases/fmd/fmd_train_resize_augmentation/'
    path_database_test = '/home/users/flavio/databases/cells/cells_test/'
    path_retrieval = '/home/users/flavio/databases/fiberFlaRom/fiberFlaRom_train/query/'
    path_cnn_trained = '/home/users/flavio/databases/fmd/fmd_train_resize_augmentation/features/model_test.ckpt'
    path_output_train = path_database_train + 'features/'
    path_output_test = path_database_test + 'features/'

    
    preprocessing_method = 'None'
    distance = 'ed'
    searching_method = 'kd'
    percent_database = 0.1
    percent_query = 0.001
    number_of_images = 10
    feature_extraction_method = 'cnn_training'
    
    #jump_num_epoch = [1,4,5,10,20,30,30]#cells
    #learning_rate =[0.1,0.1,0.08,0.04,0.02,0.01,0.009]#cells

    #jump_num_epoch = [1,9,10,20,30,30,50,100,100]#fmd
    #learning_rate =[0.1,0.1,0.03,0.02,0.01,0.008,0.004,0.002,0.001]#fmd

    #jump_num_epoch = [1,4,5,5,5]#fibers
    #learning_rate =[0.1,0.1,0.08,0.06,0.004]#fibers    

    #learning_rate =[0.1,0.1,0.1,0.08,0.06,0.02,0.008,0.006,0.004,0.001]#cells
    #learning_rate =[0.001,0.001,0.1,0.08,0.06,0.04,0.02,0.02,0.01,0.01]#cells
    
    NUM_LEVEL = [0]
    
    learning_rate_0 = 0.1
    factor_dec = 0.01
    learning_rate_f = 0.05
    
    for num_level in NUM_LEVEL:
        #remove_files_cnn(path_output_train)
    
        list_train_time = []
        list_map = []
        list_accuracy = []
        list_number_epoch = []
        list_error_total = []
        #removing files
        remove_files_pickle(path_output_train)
        cont_index=0
        #for num_epoch in jump_num_epoch:
        for num_epoch in range(1,61,1):
            
            if(num_epoch == 1 or num_epoch ==2): 
                new_learning_rate = learning_rate_0
                list_of_parameters = [str(new_learning_rate),str(1),str(num_level)]      
            else:                                   
                new_learning_rate = new_learning_new(new_learning_rate,factor_dec,num_epoch,list_error_total[-2][2],list_error_total[-1][2],num_epoch)
                if(new_learning_rate < learning_rate_f): 
                    new_learning_rate = learning_rate_f 
                list_of_parameters = [str(new_learning_rate),str(1),str(num_level)] 
                  
            #train
            #get the list of names and labels
            name_images_database, labels_database, name_images_query, labels_query = convert_database_to_files.get_name_labels(path_database_train,path_retrieval)
            _, train_time, _, error = run.run_command_line(name_images_database,labels_database,name_images_query,labels_query,path_cnn_trained,path_output_train,feature_extraction_method,distance,number_of_images,list_of_parameters,preprocessing_method,searching_method, isEvaluation=True,do_searching_processing=False,save_csv=False)
            
            if(not list_train_time):
                list_train_time.append(train_time[0])
            else:
                list_train_time.append(train_time[0] + list_train_time[-1])
            
            print('train time epoch', num_epoch, '=', list_train_time[-1]) 
               
            list_error_total.append([num_epoch, new_learning_rate, (error[0][1] + error[1][1])/2 ])
            print('Num_epoch =', list_error_total[-1][0],'Learning rate =', list_error_total[-1][1], 'Error =',list_error_total[-1][2])
        
            '''
            if(not list_train_time):
                list_train_time.append(train_time[0])
            else:
                list_train_time.append(train_time[0] + list_train_time[-1])
        
            
            #evaluation
            list_of_parameters = ['0.1','0',str(num_level)]
            name_images_database, labels_database = convert_database_to_files.get_name_labels(path_database_test)
            MAP, ACCURACY, fig = evaluation.evaluation(name_images_database, labels_database, name_images_database, labels_database,path_output_test,feature_extraction_method,distance,list_of_parameters,preprocessing_method,searching_method,path_cnn_trained=path_cnn_trained,percent_query=percent_query,percent_database=percent_database)
            list_number_epoch.append(np.sum(jump_num_epoch[0:cont_index+1]))
            list_map.append(MAP)
            list_accuracy.append(ACCURACY)
      
            print('Num_epoch =', list_number_epoch[-1],'train_time =', list_train_time[-1], 'MAP =',np.mean(MAP), 'Accuracy =',np.mean(ACCURACY))

            for i in range(len(list_map[-1])):
                print('Map for class ', i, list_map[-1][i])
                
            for i in range(len(list_map[-1])):
                print('Accuracy for class ', i, list_accuracy[-1][i])
            
            #removing files
            remove_files_pickle(path_output_test)
            cont_index+=1
            
        np.savetxt(path_output_test + feature_extraction_method + '_train_time_' + preprocessing_method + '_' + str(num_level) + '_level' + '.csv', np.asarray(list_train_time),delimiter = ',')
        np.savetxt(path_output_test + feature_extraction_method + '_Map_' + preprocessing_method + '_' + str(num_level) + '_level' + '.csv', np.asarray(list_map),delimiter = ',')
        np.savetxt(path_output_test + feature_extraction_method + '_number_epoch_' + preprocessing_method + '_' + str(num_level) + '_level' + '.csv', np.asarray(list_number_epoch),delimiter = ',')
        np.savetxt(path_output_test + feature_extraction_method + '_accuracy_' + preprocessing_method + '_' + str(num_level) + '_level' + '.csv', np.asarray(list_accuracy),delimiter = ',')
        '''
        np.savetxt(path_output_train + feature_extraction_method + '_learning_rate_error_' + '.csv', np.asarray(list_error_total),delimiter = ',')

def run_create_graph_loss_decay():
    #cnn machine
    path_database_train = '/home/users/flavio/databases/fmd/fmd_train_resize_augmentation/'
    path_database_test = '/home/users/flavio/databases/cells/cells_test/'
    path_retrieval = '/home/users/flavio/databases/fiberFlaRom/fiberFlaRom_train/query/'
    path_cnn_trained = '/home/users/flavio/databases/fmd/fmd_train_resize_augmentation/features/model_test.ckpt'
    path_output_train = path_database_train + 'features/'
    path_output_test = path_database_test + 'features/'

    
    preprocessing_method = 'None'
    distance = 'ed'
    searching_method = 'kd'
    percent_database = 0.1
    percent_query = 0.001
    number_of_images = 10
    feature_extraction_method = 'cnn_training'
    
    #jump_num_epoch = [1,4,5,10,20,30,30]#cells
    #learning_rate =[0.1,0.1,0.08,0.04,0.02,0.01,0.009]#cells

    #jump_num_epoch = [1,9,10,20,30,30,50,100,100]#fmd
    #learning_rate =[0.1,0.1,0.03,0.02,0.01,0.008,0.004,0.002,0.001]#fmd

    #jump_num_epoch = [1,4,5,5,5]#fibers
    #learning_rate =[0.1,0.1,0.08,0.06,0.004]#fibers    

    learning_rate =[0.1,0.1,0.09,0.09,0.08,0.08,0.07,0.07,0.06,0.06,0.05,0.05,0.04,0.04,0.03,0.03,0.02,0.02,0.01,0.01]
    #learning_rate =[0.001,0.001,0.1,0.08,0.06,0.04,0.02,0.02,0.01,0.01]#cells
    
    NUM_LEVEL = [0]
    
    #learning_rate_0 = 0.1
    #factor_dec = 0.01
    learning_rate_f = 0.01
    
    for num_level in NUM_LEVEL:
        #remove_files_cnn(path_output_train)
    
        list_train_time = []
        list_map = []
        list_accuracy = []
        list_number_epoch = []
        list_error_total = []
        #removing files
        remove_files_pickle(path_output_train)
        cont_index=0
        #for num_epoch in jump_num_epoch:
        for num_epoch in range(1,21,1):
            
            if(num_epoch == 1 or num_epoch ==2): 
                try:
                    new_learning_rate = learning_rate[num_epoch-1]
                except:
                    print('Learning rate', new_learning_rate)
                    new_learning_rate = learning_rate_f
                list_of_parameters = [str(new_learning_rate),str(1),str(num_level)]      
                  
            #train
            #get the list of names and labels
            name_images_database, labels_database, name_images_query, labels_query = convert_database_to_files.get_name_labels(path_database_train,path_retrieval)
            _, train_time, _, error = run.run_command_line(name_images_database,labels_database,name_images_query,labels_query,path_cnn_trained,path_output_train,feature_extraction_method,distance,number_of_images,list_of_parameters,preprocessing_method,searching_method, isEvaluation=True,do_searching_processing=False,save_csv=False)
            
            list_error_total.append([num_epoch, new_learning_rate, (error[0][1] + error[1][1])/2 ])
            print('Num_epoch =', list_error_total[-1][0],'Learning rate =', list_error_total[-1][1], 'Error =',list_error_total[-1][2])
        
            '''
            if(not list_train_time):
                list_train_time.append(train_time[0])
            else:
                list_train_time.append(train_time[0] + list_train_time[-1])
        
            
            #evaluation
            list_of_parameters = ['0.1','0',str(num_level)]
            name_images_database, labels_database = convert_database_to_files.get_name_labels(path_database_test)
            MAP, ACCURACY, fig = evaluation.evaluation(name_images_database, labels_database, name_images_database, labels_database,path_output_test,feature_extraction_method,distance,list_of_parameters,preprocessing_method,searching_method,path_cnn_trained=path_cnn_trained,percent_query=percent_query,percent_database=percent_database)
            list_number_epoch.append(np.sum(jump_num_epoch[0:cont_index+1]))
            list_map.append(MAP)
            list_accuracy.append(ACCURACY)
      
            print('Num_epoch =', list_number_epoch[-1],'train_time =', list_train_time[-1], 'MAP =',np.mean(MAP), 'Accuracy =',np.mean(ACCURACY))

            for i in range(len(list_map[-1])):
                print('Map for class ', i, list_map[-1][i])
                
            for i in range(len(list_map[-1])):
                print('Accuracy for class ', i, list_accuracy[-1][i])
            
            #removing files
            remove_files_pickle(path_output_test)
            cont_index+=1
            
        np.savetxt(path_output_test + feature_extraction_method + '_train_time_' + preprocessing_method + '_' + str(num_level) + '_level' + '.csv', np.asarray(list_train_time),delimiter = ',')
        np.savetxt(path_output_test + feature_extraction_method + '_Map_' + preprocessing_method + '_' + str(num_level) + '_level' + '.csv', np.asarray(list_map),delimiter = ',')
        np.savetxt(path_output_test + feature_extraction_method + '_number_epoch_' + preprocessing_method + '_' + str(num_level) + '_level' + '.csv', np.asarray(list_number_epoch),delimiter = ',')
        np.savetxt(path_output_test + feature_extraction_method + '_accuracy_' + preprocessing_method + '_' + str(num_level) + '_level' + '.csv', np.asarray(list_accuracy),delimiter = ',')
        '''
        np.savetxt(path_output_train + feature_extraction_method + '_learning_rate_error_decay' + '.csv', np.asarray(list_error_total),delimiter = ',')
                    
def run_create_graph_accuracy():
    #cnn
    path_database = '/home/users/flavio/databases/cells/cells_test/' #'/home/users/flavio/databases/new_database_split/new_database_split_test/'
    #path_cnn_trained = '/home/users/flavio/databases/fiberFlaRom/fiberFlaRom_train/features/model.ckpt' #'/home/users/flavio/databases/new_databa$
    path_cnn_trained = '/home/users/flavio/databases/inception_resnet_v2_2016_08_30.ckpt'
    path_output = path_database + 'features/'
    
    #flavio machine
    #path_database = '/Users/flavio/Desktop/cells/'
    #path_cnn_trained = '/Users/flavio/Desktop/cells/features/model.ckpt'
    #path_output = path_database + 'features/'
    
    preprocessing_method = 'None'
    distance = 'ed'
    searching_method = 'kd'
    percent_database = 1
    percent_query = 1
    feature_extraction_method = 'cnn'
    
    jump = 10
       
    #evaluation
    list_of_parameters = ['0.1','0','0']
    name_images_database, labels_database = convert_database_to_files.get_name_labels(path_database)
    
    list_k_accuracy = range(1,np.int(np.min(get_number_examples_per_class(labels_database))),jump)
    
    list_accuracy = evaluation.get_accuracy_using_list_k_accuracy(name_images_database, labels_database, name_images_database, labels_database,path_output,feature_extraction_method,distance,list_of_parameters,preprocessing_method,searching_method,list_k_accuracy,path_cnn_trained=path_cnn_trained,percent_query=percent_query,percent_database=percent_database)
        
    np.savetxt(path_output + feature_extraction_method + '_accuracy_per_class_' + preprocessing_method + '.csv', np.asarray(list_accuracy),delimiter = ',')
    np.savetxt(path_output + feature_extraction_method + '_list_k_accuracy_' + preprocessing_method + '.csv', np.asarray(list_k_accuracy),delimiter = ',')

run_create_graph_loss_decay()

        