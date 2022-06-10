'''
Created on Oct 6, 2016

@author: flavio
'''
import numpy as np
import src.run
from src.util import convert_database_to_files
from src.evaluation.evaluation import evaluation


def parameter_estimation(name_images_database, labels_database, name_images_query, labels_query,path_output,feature_extraction_method,distance,parameters_variation,searching_method,preprocessing_method,percent_evaluate):
    
    #get the parameters and the number of parameters
    if(len(parameters_variation) == 1):#lbp
        parameter1 = parameters_variation[0]
        parameter2 = ['']
        parameter3 = ['']
        parameter4 = ['']
        number_parameters = 1
    elif(len(parameters_variation) == 2):#glcm, hog
        parameter1 = parameters_variation[0]
        parameter2 = parameters_variation[1]
        parameter3 = ['']
        parameter4 = ['']
        number_parameters = 2
    elif(len(parameters_variation) == 3):
        parameter1 = parameters_variation[0]
        parameter2 = parameters_variation[1]
        parameter3 = parameters_variation[2]
        parameter4 = ['']
        number_parameters = 3
    elif(len(parameters_variation) == 4):#daisy
        parameter1 = parameters_variation[0]
        parameter2 = parameters_variation[1]
        parameter3 = parameters_variation[2]
        parameter4 = parameters_variation[3]
        number_parameters = 4
    
    best_Map = 0
    best_parameter = []
    list_parameters_map = []
    for p1 in parameter1:
        for p2 in parameter2:
            for p3 in parameter3:
                for p4 in parameter4:
            
                    if(number_parameters == 1):
                        list_of_parameters = [p1]
                    elif(number_parameters == 2):
                        list_of_parameters = [p1, p2]
                    elif(number_parameters == 3):
                        list_of_parameters = [p1, p2, p3]
                    elif(number_parameters == 4):
                        list_of_parameters = [p1, p2, p3, p4]
                        
                    try:
                        MAP,_ = evaluation.evaluation(name_images_database, labels_database, name_images_query, labels_query,path_output,feature_extraction_method,distance,list_of_parameters,preprocessing_method,searching_method,percent_evaluate)
                        
                        if(np.mean(MAP) > best_Map):
                            best_Map = np.mean(MAP)
                            best_parameter = [p1, p2, p3, p4]
                        
                        list_parameters_map.append([p1,p2,p3,p4,np.mean(MAP)])
                                                    
                        convert_database_to_files.remove_files(path_output, feature_extraction_method, distance, list_of_parameters)
                    except:
                        convert_database_to_files.remove_files(path_output, feature_extraction_method, distance, list_of_parameters)
                        print('Error', list_parameters_map[-1])
                        
    return best_Map, best_parameter,np.asarray(list_parameters_map)
