import sys, getopt

def main(argv):

    path_database = '' # -d
    path_output = '' # -o
    path_query = '' # -q

    feature_extraction_method = 'pretrained_vgg16' #optional -fe
    path_cnn_pre_trained = '' #optional -pt
    path_save_cnn = '/Users/flavio/Desktop/cells_small/output/model.h5' #optional - s
    preprocessing_method = 'None' #optional -pr
    searching_method = 'bf' #optional -sm
    distance = 'ed' #optional -di
    number_of_images = 5 #optional -n
    list_of_parameters = ['0.01','1'] #optional -p

    try:
      opts, args = getopt.getopt(argv,"hd:q:o:f:t:s:r:s:n:p:")
    except getopt.GetoptError:
      #print ('pycbir_cl.py -d <path_database> -c <path_cnn_trained> -r <path_folder_retrieval> -f <feature_extraction_method> -s <distance-similarity metric> -n <number_of_images> -m <list_of_parameters>')
      #Mandatory parameters
      print('Mandatory parameters: ')
      print ('pycbir_cl.py -d <path_database> -o <path_output> -q <path_query>')

      print('\nOptionals parameters: ')
      print ('-f <feature_extraction_method> -pt <path_cnn_pre_trained> -s <path_save_cnn> -pr <preprocessing_method> -sm <searching_method> -n <number_of_images> -p <list_of_parameters>')
      sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            #Describe all parameters
            print ("""
            Mandatory parameters:

            -d <path_database>:
                The path of the folder containing the images used in the retrieval process

            -q <path_query>:
                The path of the folder containing the query images

            -o <path_output>:
                The path of the folder where some files will be save

            Optional parameters:

            -f <feature_extraction_method>:
                The descriptor to extract features, values: 'glcm', 'fotf', 'lbp', 'hog', 'daisy','lenet', 'pretrained_lenet'
                , 'pretrained_vgg16', 'pretrained_vgg19', 'pretrained_xception', 'pretrained_resnet', 'pretrained_inception_resnet'
                , 'pretrained_nasnet', 'fine_tuning_lenet', 'fine_tuning_vgg16', 'fine_tuning_vgg19', 'fine_tuning_xception'
                , 'fine_tuning_resnet', 'fine_tuning_inception_resnet' and 'fine_tuning_nasnet'

            -t <path_cnn_pre_trained>:
                If the feature_extraction_method is 'lenet' or 'fine_tuning_*', this parameter is the path to a h5 file that will
                be used to initialize the network weights

            -s <path_save_cnn>:
                If the feature_extraction_method is 'lenet' or 'fine_tuning_*', this parameter is the path that the trained network
                will be saved

            -r <preprocessing_method>:

            -s <searching_method>:
                The searching method to calculate the similarity, values: 'bf' to brute force, 'kd' to kdTree and 'bt' to BallTree

            -n <number_of_images>:
                Number of images that will be returned for each query example

            -p <list_of_parameters>:

            """ )
            sys.exit()
        elif opt == '-d':
            path_database = arg
        elif opt == '-q':
            path_query = arg
        elif opt == '-o':
            path_output = arg
        elif opt == '-f':
            feature_extraction_method = arg
        elif opt == '-t':
            path_cnn_pre_trained = arg
        elif opt == '-s':
            path_save_cnn = arg
        elif opt == '-r':
            preprocessing_method = arg
        elif opt == '-s':
            searching_method = arg
        elif opt == '-n':
            number_of_images = int(float(arg))
        elif opt == '-p':
            print('parameters = ', arg)
            parameters = arg.split(',')
            list_of_parameters = []
            for i in parameters:
                list_of_parameters.append(i)

    from run import run_command_line
    from util import convert_database_to_files

    name_images_database, labels_database, name_images_query, labels_query = convert_database_to_files.get_name_labels(path_database,path_query)
    run_command_line(name_images_database,labels_database,name_images_query,labels_query,path_cnn_pre_trained,path_save_cnn,path_output,feature_extraction_method,distance,number_of_images,list_of_parameters,preprocessing_method,searching_method, False)

if __name__ == "__main__":
    main(sys.argv[1:])

'''
    try:
      opts, args = getopt.getopt(argv,"hd:c:r:f:s:p:n:m:")
    except getopt.GetoptError:
      print ('cbir_cl.py -d <path_database> -c <path_cnn_trained> -r <path_folder_retrieval> -f <feature_extraction_method> -s <distance-similarity metric> -n <number_of_images> -m <list_of_parameters>')
      sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ('cbir_cl.py -d <path_database> -c <path_cnn_trained> -r <path_folder_retrieval> -f <feature_extraction_method> -s <distance-similarity metric> -n <number_of_images> -m <list_of_parameters>')
            sys.exit()
        elif opt == '-d':
            path_database = arg
        elif opt == '-c':
            path_cnn_trained = arg
        elif opt == '-r':
            path_retrieval = arg
        elif opt == '-f':
            feature_extraction_method = arg
        elif opt == '-s':
            distance = arg
        elif opt == '-p':
            searching_method = arg
        elif opt == '-n':
            number_of_images = int(float(arg))
        elif opt == '-m':
            parameters = arg.split(',')
            list_of_parameters = []
            for i in parameters:
                list_of_parameters.append(i)
'''
