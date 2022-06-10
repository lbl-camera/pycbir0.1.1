'''
Created on 5 de jul de 2016

@author: flavio
'''
import numpy as np
import run

def run_test():
    
    FEATURE_EXTRACTION_METHOD = ['glcm', 'hog', 'lbp', 'fotf']
    DISTANCE = ['ED', 'CD', 'ID', 'CS', 'PCC', 'CSD', 'KLD', 'JD', 'KSD', 'CMD', 'EMD']
    
    path_database = "/Users/romuere/Dropbox/CBIR/fibers/database/"
    path_retrieval = "/Users/romuere/Dropbox/CBIR/fibers/retrieval/"
    #path_retrieval = ""
    path_image = ""
    #path_image = "/Users/flavio/Dropbox/Compartilhadas/Romuere/CBIR/fibers/retrieval/fiber_9_191_642.tif"
    extension_classes = ["tif", "tif"]
    #feature_extraction_method = "hog"
    #distance = "KLD"
    number_of_images = 10
    
    for feature_extraction_method in FEATURE_EXTRACTION_METHOD:
        for distance in DISTANCE:
            try:
                run.run(path_database,path_retrieval,path_image,extension_classes,feature_extraction_method,distance,number_of_images)
                print("Ok: ", feature_extraction_method, " ", distance)
            except: 
                print("Fail: ", feature_extraction_method, " ", distance, "--------------------------------------")
                
run_test()