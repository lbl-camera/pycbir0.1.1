import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import warnings 
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, '..')

import matplotlib
matplotlib.use("Agg")
import numpy as np
import csv
from skimage.io import imread
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import timeit
import itertools

from src.cnn.Parameters import Parameters
from src.similarity import searching
from src.signatures import feature_extraction as fe
from src.cnn import keras_training
#---------------------------------------------------------------------------------------------------------------#
'''
pyCBIR

Developers: Romuere Silva, Flavio Araujo and Dani Ushizima
'''

def accuracy(result,retrieval_labels,retrieval_number, feature_extraction_method, probability_vector):
    """
    This is the function compute the accuracy based on the retrieval images.
    It is just the probability of the object to be of each class

    Parameters
    ----------
    small_distances:list of lists of str
        Each position corresponds to a list of paths. For each retrieval image it has an list of the retrieval sorted
    folder_classes: list of str
        The length of this list corresponds to the number of classes in the database
    k: int
        Number of image to retrieve for each query
    Returns
    -------
    probabilities: list
        Each position corresponds to a string with the probability of each object.
    """
    if(feature_extraction_method == 'cnn_training' or feature_extraction_method == 'fine_tuning_inception'):
        text = []
        cont = 0
        for probability_query in probability_vector:
            class_ = np.argmax(probability_query)

            format_str = ('%.2f class %d\n (GT class: %.2f)')
            text.append(format_str % (probability_query[class_], class_,int(retrieval_labels[cont])))
            cont+=1
        return text
    else:
        percent = []
        for i in result:
            un = np.unique(i)
            r = []
            for j in un:
                r.append(sum(i==j)/len(i)*100)
            percent.append((max(r),un[np.argmax(r)]))
        text = []
        for i in range(len(result)):
            text.append(str(percent[i][0])+' class '+ str(percent[i][1])+'\n (GT class: '+str(int(retrieval_labels[i]))+')')
        return text

def show_retrieval_indexing(fname_retrieval, result, labels_retrieval, retrieval_number, path_output, feature_extraction_method, similariry_metric, searching_method, log_vis=False, probability_vector = []):

    """
    This is the function to show and save the visual result of pyCBIR.

    Parameters
    ----------
    images_retrievael: list of numpy
        Retrieval images, each position of the list corresponds to an image
    small_distances:list of lists of str
        Each position corresponds to a list of paths. For each retrieval image it has an list of the retrieval sorted
    retrieval_number: int
        Number of image to retrieve for each query
    path_database: str
        Complete path of the database
    feature_extraction_method: str
        Feature extraction method that follows the README specification
    distace: str
        Similarity metric that follows the README specification
    folder_classes: list of str
        The length of this list corresponds to the number of classes in the database

    Returns
    -------

    """

    acc = accuracy(result[1],labels_retrieval,retrieval_number,feature_extraction_method,probability_vector)
    fig, ax = plt.subplots(len(fname_retrieval),retrieval_number+1, sharex=True, sharey=True)
    fig.set_size_inches(10,10)
    gs1 = GridSpec(len(fname_retrieval),retrieval_number+1)
    gs1.update(wspace=0.025, hspace=0.5) # set the spacing between axes.
    if len(fname_retrieval) > 1:
        cont = 0
        for cont2,i in enumerate(fname_retrieval):
            if log_vis == True:
                im = imread(i)
                a = (im == 0)
                im[a] = 1
                im = np.log(im)
                ax[cont2,0].imshow(im,cmap='viridis',interpolation = 'none')
            else:
                ax[cont2,0].imshow(imread(i),cmap='jet',interpolation = 'none')

            ax[cont2,0].set_adjustable('box')
            ax[cont2,0].set_yticks([])
            ax[cont2,0].set_xticks([])
            ax[cont2,0].set_ylabel(acc[cont2],fontsize = 5)
            cont += 1
            #for each retrieval image returns the k nearer images
            for j in range(retrieval_number):
                if log_vis == True:
                    im = imread(result[0][cont2][j])
                    a = (im == 0)
                    im[a] = 1
                    im = np.log(im)
                    ax[cont2,j+1].imshow(im,cmap='viridis',interpolation = 'none')
                else:
                    ax[cont2,j+1].imshow(imread(result[0][cont2][j]),cmap='jet',interpolation = 'none')

                if result[1][cont2][j] == labels_retrieval[cont2]:
                    color = 'green'
                else:
                    color = 'red'
                ax[cont2,j+1].set_xticks([])
                ax[cont2,j+1].set_yticks([])
                for axis in ['top','bottom','left','right']:
                    ax[cont2,j+1].spines[axis].set_linewidth(3)

                ax[cont2,j+1].spines['bottom'].set_color(color)
                ax[cont2,j+1].spines['top'].set_color(color)
                ax[cont2,j+1].spines['right'].set_color(color)
                ax[cont2,j+1].spines['left'].set_color(color)

                ax[cont2,j+1].set_adjustable('box')
                ax[cont2,j+1].set_yticks([])
                ax[cont2,j+1].set_xticks([])

                #shortName = result[0][cont2][j]
                #shortName = shortName.split('/')[-1]
                #shortName = shortName[0:10]
                #ax[cont2,j+1].set_title(shortName,fontsize=5)
                cont += 1

    else:
        im = imread(fname_retrieval[0])
        if log_vis == True:
            im = imread(i)
            a = (im == 0)
            im[a] = 1
            im = np.log(im)
            ax[0].imshow(im,cmap='viridis',interpolation = 'none')
        else:
            ax[0].imshow(im,cmap='viridis',interpolation = 'none')
        ax[0].set_adjustable('box')
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        ax[0].set_ylabel(acc[0],fontsize = 5)
        #for each retrieval image returns the k nearer images
        for j in range(retrieval_number):
            im = imread(result[0][0][j])
            if log_vis == True:
                    a = (im == 0)
                    im[a] = 1
                    im = np.log(im)
                    ax[j+1].imshow(im,cmap='viridis',interpolation = 'none')
            else:
                ax[j+1].imshow(im,cmap='viridis',interpolation = 'none')

            ax[j+1].set_adjustable('box')
            ax[j+1].set_yticks([])
            ax[j+1].set_xticks([])
            shortName = result[0][0][j]
            shortName = shortName.split('/')[-1]
            shortName = shortName[0:10]
            ax[j+1].set_title(shortName,fontsize=5)


    file = path_output + "result" + "_" + feature_extraction_method + "_" + similariry_metric + "_" + str(retrieval_number) + "_searching_method_" + searching_method +".png"
    #print(file)
    fig.savefig(file,bbox_inches='tight')   # save the figure to file   # save the figure to file
    #os.system(file)
    print("-------------------------------------------------------------")
    return file

def cnn_feature_extraction(name_images_database,labels_database,name_images_query,labels_query,path_cnn_pre_trained,path_save_cnn,path_output,feature_extraction_method,list_of_parameters,preprocessing_method,do_searching_processing,save_csv):

    """
    Documentation
    """
    train_time = 0
    batch_size = 16

    #Name of the files and paths to read or save the csv
    path_features = path_output + '/feature_vectors_' + feature_extraction_method + '.csv'
    path_filenames = path_output + '/image_filenames_' + feature_extraction_method + '.csv'
    path_labels = path_output + '/labels_' + feature_extraction_method + '.csv'

    if(feature_extraction_method[0:8] == 'training' or feature_extraction_method[0:11] == 'fine_tuning'):
        parameters = Parameters(batch_size,name_images_database,labels_database,path_output,path_cnn_pre_trained,path_save_cnn,list_of_parameters,preprocessing_method,feature_extraction_method)

        #if the train is necessary
        if(parameters.NUM_EPOCHS > 0):
            start = timeit.default_timer()

            if(feature_extraction_method[0:8] == 'training' or feature_extraction_method[0:11] == 'fine_tuning'):
                keras_training.train_model(parameters)

            stop = timeit.default_timer()
            train_time = (stop - start)

            #To generate the plot of the error decay
            if(parameters.LIST_ERROR):
                fig, ax = plt.subplots( nrows=1, ncols=1 )
                datas = np.asarray(parameters.LIST_ERROR)
                ax.plot(datas[:,0],datas[:,1])
                plt.xlabel('training epoch')
                plt.ylabel('error')
                plt.title('error')
                #plt.legend(loc='best')
                fig.savefig(path_output + 'error_training.png')

        #get the time of extraction of features for the whole database
        start = timeit.default_timer()
        if os.path.isfile(path_features) and os.path.isfile(path_filenames) and os.path.isfile(path_labels) and parameters.NUM_EPOCHS == 0:
            #read path_features
            reader = csv.reader(open(path_features),delimiter=',')
            x = list(reader)
            feature_vectors_database = np.array(x).astype('float16')

            #read path_filenames
            reader = csv.reader(open(path_filenames))
            fname_database = list(reader)
            fname_database  = list(itertools.chain(*fname_database))

            #read labels_database
            reader = csv.reader(open(path_labels),delimiter=',')
            x = list(reader)
            labels_database = np.array(x).astype(dtype = np.uint16)
            labels_database = labels_database.reshape(-1)
            #Calculating the time of the extraction of features for the whole database
            stop = timeit.default_timer()

        #To just train the cnn without do the retrieval process set do_searching_processing as False
        elif(do_searching_processing):

            if(feature_extraction_method[0:8] == 'training' or feature_extraction_method[0:11] == 'fine_tuning'):
                feature_vectors_database, fname_database, labels_database,probability_vector = keras_training.features_extraction(parameters)

            #Calculating the time of the extraction of features for the whole database
            stop = timeit.default_timer()

            if(save_csv):
                #save files
                np.savetxt(path_features, feature_vectors_database,delimiter = ',')
                np.savetxt(path_filenames, fname_database,fmt='%s')
                np.savetxt(path_labels, labels_database,fmt = '%d')

        #time
        time_to_extract_features = (stop - start)

        if(do_searching_processing):
            #calling the extraction of features for the retrieval images
            parameters.NAME_IMAGES = name_images_query
            parameters.LABELS = labels_query

            if(feature_extraction_method[0:8] == 'training' or feature_extraction_method[0:11] == 'fine_tuning'):
                feature_vectors_query, fname_query, labels_query,probability_vector = keras_training.features_extraction(parameters)

            return fname_database, feature_vectors_database, labels_database, fname_query, feature_vectors_query, labels_query, time_to_extract_features, train_time, parameters.LIST_ERROR, probability_vector
        return [], [], [], [], [], [], 0, train_time,parameters.LIST_ERROR

    #to use the inception resnet pretrained
    elif(feature_extraction_method[0:10] == 'pretrained'):

        parameters = Parameters(batch_size,name_images_database,labels_database,path_output,path_cnn_pre_trained,path_save_cnn,list_of_parameters,preprocessing_method,feature_extraction_method)

        #get the time of extraction of features for the whole database
        start = timeit.default_timer()

        if os.path.isfile(path_features) and os.path.isfile(path_filenames) and os.path.isfile(path_labels):
            #read path_features
            reader = csv.reader(open(path_features),delimiter=',')
            x = list(reader)
            feature_vectors_database = np.array(x).astype('float16')

            #read path_filenames
            reader = csv.reader(open(path_filenames,encoding = "ISO-8859-1"))
            fname_database = list(reader)
            fname_database  = list(itertools.chain(*fname_database))

            #read labels_database
            reader = csv.reader(open(path_labels),delimiter=',')
            x = list(reader)
            labels_database = np.array(x).astype(dtype = np.uint16)
            labels_database = labels_database.reshape(-1)

            #Calculating the time of the extraction of features for the whole database
            stop = timeit.default_timer()
        else:

            #if(feature_extraction_method == 'pretrained_lenet'):
            #    feature_vectors_database, fname_database, labels_database, _ = keras_training.features_extraction(parameters)
            #else:
            feature_vectors_database, fname_database, labels_database, _ = keras_training.features_extraction_from_pretrained_cnn(parameters)

            #Calculating the time of the extraction of features for the whole database
            stop = timeit.default_timer()

            np.savetxt(path_features, feature_vectors_database,delimiter = ',')
            np.savetxt(path_filenames, fname_database,fmt='%s')
            np.savetxt(path_labels, labels_database,fmt = '%d')
        #time
        time_of_extraction_features = (stop - start)
        parameters.NAME_IMAGES = name_images_query
        parameters.LABELS = labels_query

        #if (feature_extraction_method == 'pretrained_lenet'):
        #    feature_vectors_query, fname_query, labels_query, _ = keras_training.features_extraction(parameters)
        #else:
        feature_vectors_query, fname_query, labels_query, _  = keras_training.features_extraction_from_pretrained_cnn(parameters)

        return fname_database, feature_vectors_database, labels_database, fname_query, feature_vectors_query, labels_query, time_of_extraction_features, train_time,None, None

def run_command_line(fname_database,labels_database,fname_retrieval,labels_retrieval,path_cnn_pre_trained,path_save_cnn,path_output,feature_extraction_method,similarity_metric,retrieval_number,list_of_parameters,preprocessing_method, searching_method = 'bf', isEvaluation = False,do_searching_processing=True,save_csv=True):

    """
    This is the main function of the pycbir project, the interface will call
    this function. At the end of the process it will show the result and will
    save a .png file with the result. Also, it will return the 'result'
    variable to be used to compute the retrieval performance.

    Parameters
    ----------
    path_databse: str
        Complete path of the database folder
    path_retrieval:str
        Complete path of the retrieval images folder, if this value is '', then it will compute the retrieval for one image.
    path_image: str
        Complete path of a single retrievial image
    path_cnn_trained: str
        Complete path of a trained cnn
    feature_extraction_method: str
        Feature extraction method that follows the README specification
    distace: str
        Similarity metric that follows the README specification
    number_of_images: int
        Number of images to retrieve
    list_of_parameters: list
        Parameters for the feature extraction method chosed

    Returns
    -------
    result : list
        list of all points between start and end
    """

    probability_vector = []

    print('pyCBIR started!')
    train_time = 0
    if(feature_extraction_method[0:9] == 'training_' or feature_extraction_method[0:11] == 'fine_tuning' or feature_extraction_method[0:10] == 'pretrained'):
        fname_database,feature_vectors_database,labels_database,fname_retrieval,feature_vectors_retrieval,labels_retrieval, time_of_extraction_features, train_time, list_error, probability_vector = cnn_feature_extraction(fname_database,labels_database,fname_retrieval,labels_retrieval,path_cnn_pre_trained,path_save_cnn,path_output,feature_extraction_method,list_of_parameters,preprocessing_method,do_searching_processing,save_csv)

    else:
        #Get the csv file
        #check if there is a file computed for this descriptor-distance
        parameters_name = ""
        for parameter in list_of_parameters:
            parameters_name = parameters_name + "_" + parameter

        #get the time of extraction of features for the whole database
        start = timeit.default_timer()

        path_features = path_output + "feature_vectors" + "_" + feature_extraction_method + parameters_name + ".csv"
        path_filenames = path_output + "image_filenames" + "_" + feature_extraction_method + parameters_name + ".csv"
        path_labels = path_output + "labels" + "_" + feature_extraction_method + parameters_name + ".csv"

        if os.path.isfile(path_features) and os.path.isfile(path_filenames) and os.path.isfile(path_labels):
            #read path_features
            reader = csv.reader(open(path_features),delimiter=',')
            x = list(reader)
            feature_vectors_database = np.array(x).astype('float')

            #read path_filenames
            reader = csv.reader(open(path_filenames))
            fname_database = list(reader)
            fname_database  = list(itertools.chain(*fname_database))

            #read labels_database
            reader = csv.reader(open(path_labels),delimiter=',')
            x = list(reader)
            labels_database = np.array(x).astype(dtype = np.uint16)
            labels_database = labels_database.reshape(-1)


        else:
            #computing features for the database
            fname_database,feature_vectors_database,labels_database = fe.descriptor_all_database(fname_database,labels_database,feature_extraction_method,list_of_parameters,preprocessing_method)
            np.savetxt(path_features, feature_vectors_database,delimiter = ',')
            np.savetxt(path_filenames, fname_database,fmt='%s')
            np.savetxt(path_labels, labels_database,fmt = '%d')

        #Calculating the time of the extraction of features for the whole database
        stop = timeit.default_timer()
        time_of_extraction_features = (stop - start)

        #computing features for the retrieval image(s)
        fname_retrieval,feature_vectors_retrieval,labels_retrieval = fe.descriptor_all_database(fname_retrieval,labels_retrieval,feature_extraction_method,list_of_parameters,preprocessing_method)


    if(not do_searching_processing):
        return [], [train_time,time_of_extraction_features], [], list_error
    #get the time of the retrieval of the query image
    start = timeit.default_timer()
    print('Features Extracted!')
    #compute the ranked outputs
    result = searching.searching(feature_vectors_database,feature_vectors_retrieval, labels_database, fname_database,similarity_metric,retrieval_number,list_of_parameters,feature_extraction_method,path_output,searching_method)
    print('Searching Process!')
    #Calculating the time of the extraction of features for the whole database
    stop = timeit.default_timer()
    time_of_retrieval_images = (stop - start)

    if(not isEvaluation):
        #show the ranked output

        file = show_retrieval_indexing(fname_retrieval, result, labels_retrieval, retrieval_number, path_output, feature_extraction_method, similarity_metric, searching_method, probability_vector = probability_vector, log_vis = False)#log_vis = True

        return result, train_time, file

    else:
        return result, [train_time,time_of_extraction_features], time_of_retrieval_images

    print('Results Generated!')
