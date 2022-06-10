'''
Created on 05 de out de 2016

@author: romuere
'''


from src.util import parallel
#import yourPreProcessing
from src.preprocessing import pp1
from src.preprocessing import pp2

'''
In this file the user can put any pre-processing (PP) method.  
Is is necessary that the PP method has the same structure of the ones already implemented.
Each PP has a identifier, str or int. 
'''


def preprocessing(collection,labels,preprocessing_method,**list_of_parameters):
    """
    Function to pre-process the images
    
    Parameters
    ----------
    image : numpy array
    preprocessing_method : string
        descriptor to compute the features
    list_of_parameters : numpy array or list
        parameters of the preprocessing method
        
    Returns
    -------
    images : numpy array
    """
        
    if preprocessing_method == 'simple':
        res = parallel.apply_parallel(collection, collection.files,labels, pp1.preprocessing)
    if preprocessing_method == 'log':
        res = parallel.apply_parallel(collection, collection.files,labels, pp2.preprocessing)
    elif preprocessing_method == 'other':
        res = parallel.apply_parallel(collection, labels, pp2.main)
    
    collection = []
    files = []
    for cont,e in enumerate(res):
        collection.append(e[0])
        files.append(e[1])
        labels[cont] = e[2]
        
    return collection, files, labels
