'''
Created on 30 de aug de 2016

@author: romuere
'''
import numpy as np
import csv
from glob import glob
import os
import itertools
import run  

def run_sorting():
    
    """
    This is just a test function, to avoid run the GUI every time.    
    """
    
    import csv
    import itertools
    
    """
    ##To run fibers/cells/fmd/dtd/...
    folders = ['/Users/romuere/Dropbox/CBIR/fibers/database/no_fibers/*','/Users/romuere/Dropbox/CBIR/fibers/database/yes_fibers/*']
    fname_database = []
    labels_database = np.empty(0)
    for id,f in enumerate(folders):
        files = glob.glob(f)
        labels_database = np.append(labels_database, np.zeros(len(files))+id)
        fname_database = fname_database+files
        print(files)
    print(len(fname_database))
    preprocessing_method = 'log'
    feature_extraction_method = 'glcm'
    searching_method = 'lsh'
    retrieval_number = 10
    similarity_metric = 'ed'
    path_output = '/Users/romuere/Dropbox/CBIR/fibers/results/'
    list_of_parameters = ['1','2']
    path_cnn_trained = ''
    
    fname_retrieval = fname_database[0:3] + fname_database[2001:2003] 
    labels_retrieval = np.concatenate((labels_database[0:3],labels_database[2001:2003]))
    """
    ##To run scattering images
    path = '/Users/romuere/Desktop/als/kyager_data_raw'
    files_database_class0 = '/Users/romuere/Desktop/als/kyager_data_raw/SAXS.txt'
    files_database_class1 = '/Users/romuere/Desktop/als/kyager_data_raw/WAXS.txt'
    files_retrieval_class0 = '/Users/romuere/Desktop/als/kyager_data_raw/SAXS_query.txt'
    files_retrieval_class1 = '/Users/romuere/Desktop/als/kyager_data_raw/WAXS_query.txt'
    
    #------#
    reader = csv.reader(open(files_database_class0))
    fname_database_class0 = list(reader)
    fname_database_class0  = list(itertools.chain(*fname_database_class0))
    labels_class_0 = np.zeros(len(fname_database_class0))
    
    reader = csv.reader(open(files_database_class1))
    fname_database_class1 = list(reader)
    fname_database_class1 = list(itertools.chain(*fname_database_class1))
    labels_class_1 = np.zeros(len(fname_database_class1))+1
   
    fname_database = fname_database_class0+fname_database_class1
    fname_database = [path+x for x in fname_database]
    labels_database = np.concatenate((labels_class_0,labels_class_1))
    #------#
    reader = csv.reader(open(files_retrieval_class0))
    fname_retrieval_class0 = list(reader)
    fname_retrieval_class0  = list(itertools.chain(*fname_retrieval_class0))
    labels_retrieval_class0 = np.zeros(len(fname_retrieval_class0))
    
    reader = csv.reader(open(files_retrieval_class1))
    fname_retrieval_class1 = list(reader)
    fname_retrieval_class1 = list(itertools.chain(*fname_retrieval_class1))
    labels_retrieval_class1 = np.zeros(len(fname_retrieval_class1))
    
    fname_retrieval = fname_retrieval_class0+fname_retrieval_class1
    fname_retrieval = [path+x for x in fname_retrieval]
    labels_retrieval = np.concatenate((labels_retrieval_class0,labels_retrieval_class1))
    #------#
    
    
    preprocessing_method = 'log'
    feature_extraction_method = 'lbp'
    searching_method = 'lsh'
    retrieval_number = 10
    similarity_metric = 'ed'
    path_output = '/Users/romuere/Desktop/als/output/'
    list_of_parameters = ['2']#['2','8','8']
    path_cnn_trained = ''
    
    
    
    run.run_command_line(fname_database,labels_database,fname_retrieval,labels_retrieval,path_cnn_trained,path_output,feature_extraction_method,similarity_metric,retrieval_number,list_of_parameters,preprocessing_method,searching_method, isEvaluation = False)
run_sorting()

def run_cubic():

    """
    This is just a test function, to avoid run the GUI every time.
    """

    path = '/home/users/romuere/Saxsgen/'
    folders = [x[0] for x in os.walk(path)][1:]
    fname_database = []
    labels_database = np.empty(0)
    fname_retrieval = [] 
    labels_retrieval = np.empty(0)
    
    cont = 0
    for id,f in enumerate(folders):
        test = f.split('/')
        if (not(test[-1][0] == '.') and not(test[-2][0]== '.')):
            print(f)
            files = glob.glob(f+'/*')
            labels_database = np.append(labels_database, np.zeros(len(files))+cont)
            fname_database = fname_database+files
            fname_retrieval += files[0]
            labels_retrieval = np.append(labels_retrieval,cont) 
            cont += 1

    preprocessing_method = 'log'
    feature_extraction_method = 'lbp'
    searching_method = 'lsh'
    retrieval_number = 10
    similarity_metric = 'ed'
    path_output = '/home/users/romuere/Saxsgen/'
    list_of_parameters = ['3']#['0.01','10000']
    path_cnn_trained = ''#/home/users/romuere/Saxsgen/model.ckpt'
    
    print(fname_retrieval)
    
    run_command_line(fname_database,labels_database,fname_retrieval,labels_retrieval,path_cnn_trained,path_output,feature_extraction_method,similarity_metric,retrieval_number,list_of_parameters,preprocessing_method,searching_method, isEvaluation = False)
#run_cubic()
