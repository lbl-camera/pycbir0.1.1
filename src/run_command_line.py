
from run import run_command_line
from util import convert_database_to_files
import timeit


path_database = '/Users/flavio/Desktop/cells_small/database/'
path_cnn_pre_trained = ''
#path_save_cnn = ''
path_save_cnn = '/Users/flavio/Desktop/cells_small/output/model.h5'
path_retrieval = '/Users/flavio/Desktop/cells_small/query_2/'
path_output = '/Users/flavio/Desktop/cells_small/output/'


#feature_extraction_method = 'pretrained_vgg16' #'lenet' to training the LeNet, 'pretrained_name' to use the network pretrained and 'fine_tuning_name' to fine-tuning the network. name can be inception_resnet, vgg, nasnet and inception_v4
feature_extraction_method = 'pretrained_vgg16'
preprocessing_method = 'None'
searching_method = 'bf' #bf is to use brute force and lsh is to use the lsh
distance = 'ed' #the distance used in the brute force
number_of_images = 10 #number of images retrived

list_of_parameters = ['0.0001','1'] #inicital learning rate and number of epochs

#get the list of names and labels
name_images_database, labels_database, name_images_query, labels_query = convert_database_to_files.get_name_labels(path_database,path_retrieval)


############################ Calling function ##################################
start = timeit.default_timer()

_, train_time, _ = run_command_line(name_images_database,labels_database,name_images_query,labels_query,path_cnn_pre_trained,path_save_cnn,path_output,feature_extraction_method,distance,number_of_images,list_of_parameters,preprocessing_method,searching_method, False)

stop = timeit.default_timer()

print("Total time = ", stop - start)
print("Train time = ", train_time)
