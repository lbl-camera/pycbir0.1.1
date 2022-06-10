# coding: utf-8
'''
Created on 1 de jul de 2016

@author: flavio
'''

import numpy as np
from tkinter import *
#from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename, askdirectory
import os
import glob
import run
import convert_database_to_files

def salve_values():
    global list_of_parameters
    list_of_parameters = []
    
    if(feature_extraction_method == "glcm"):
        list_of_parameters.append(input_distance_glcm.get())
        list_of_parameters.append(input_gray_levels_glcm.get())
        
    elif(feature_extraction_method == "hog"):
        list_of_parameters.append(input_block_hog.get())
        list_of_parameters.append(input_window_hog.get())

    elif(feature_extraction_method == "lbp"):
        list_of_parameters.append(input_radius_lbp.get())
    
    elif(feature_extraction_method == "cnn" or feature_extraction_method == "cnn_probability"):
        list_of_parameters.append(input_learning_rate_cnn.get())
        list_of_parameters.append(input_epochs_cnn.get())
        
    cancel()

def get_parameters():
    global feature_extraction_method
    
    feature_extraction_method = feature_extraction_method_name(var1.get())
    parameters_frame.title("pyCBIR - Parameters")
    parameters_frame.geometry('+500+250')
    #parameters_frame = Toplevel()
    
    if(feature_extraction_method == "glcm"):
        lb_distance_glcm.grid(row=0, column=0, sticky='E', padx=5, pady=2)
        input_distance_glcm.grid(row=0, column=1, sticky='E', padx=5, pady=2)
        lb_gray_levels_glcm.grid(row=1, column=0, sticky='E', padx=5, pady=2)
        input_gray_levels_glcm.grid(row=1, column=1, sticky='E', padx=5, pady=2)
        bt_parameters_cancel.grid(row=2, column=0, sticky='E', padx=5, pady=2)
        bt_ok.grid(row=2, column=1, sticky='E', padx=5, pady=2)
        parameters_frame.deiconify()
        
    elif(feature_extraction_method == "hog"):
        lb_block_hog.grid(row=0, column=0, sticky='E', padx=5, pady=2)
        input_block_hog.grid(row=0, column=1, sticky='E', padx=5, pady=2)
        lb_window_hog.grid(row=1, column=0, sticky='E', padx=5, pady=2)
        input_window_hog.grid(row=1, column=1, sticky='E', padx=5, pady=2)
        bt_parameters_cancel.grid(row=2, column=0, sticky='E', padx=5, pady=2)
        bt_ok.grid(row=2, column=1, sticky='E', padx=5, pady=2)
        parameters_frame.deiconify()
    
    elif(feature_extraction_method == "lbp"):
        lb_radius_lbp.grid(row=0, column=0, sticky='E', padx=5, pady=2)
        input_radius_lbp.grid(row=0, column=1, sticky='E', padx=5, pady=2)
        bt_parameters_cancel.grid(row=1, column=0, sticky='E', padx=5, pady=2)
        bt_ok.grid(row=1, column=1, sticky='E', padx=5, pady=2)
        parameters_frame.deiconify()

    elif(feature_extraction_method == "cnn" or feature_extraction_method == "cnn_probability"):
        lb_learning_rate_cnn.grid(row=0, column=0, sticky='E', padx=5, pady=2)
        input_learning_rate_cnn.grid(row=0, column=1, sticky='E', padx=5, pady=2)
        lb_epochs.grid(row=1, column=0, sticky='E', padx=5, pady=2)
        input_epochs_cnn.grid(row=1, column=1, sticky='E', padx=5, pady=2)
        check_button.grid(row=3, column=0, columnspan=3,sticky='N', padx=5, pady=2)
        bt_parameters_cancel.grid(row=4, column=0, sticky='E', padx=5, pady=2)
        bt_ok.grid(row=4, column=1, sticky='E', padx=5, pady=2)
        parameters_frame.deiconify()
        
def retrieval():
    global feature_extraction_method
    global distance
    global number_of_images
    global parameters_frame
    
    number_of_images = np.int16(input_number_of_images.get())
    feature_extraction_method = feature_extraction_method_name(var1.get())
    distance = distance_name(var2.get())
    
    #ajeitar para pegar esse parametro da interface
    #path_cnn_trained = '/Users/flavio/Dropbox/Compartilhadas/Romuere/CBIR/inception-2015-12-05/classify_image_graph_def.pb'
    #print(path_database,path_folder_retrieval,path_image,extension_classes,feature_extraction_method,distance,number_of_images,list_of_parameters)
    
    if( (feature_extraction_method == 'cnn' or feature_extraction_method == 'cnn_probability') and var3.get() == 0 ):
        feature_extraction_method = feature_extraction_method + '_training'
        
    #get the list of names and labels
    name_images_database, labels_database, name_images_query, labels_query = convert_database_to_files.get_name_labels(path_database,path_folder_retrieval)
    path_output = path_database + 'features/'
    path_cnn_trained = ''
    if(feature_extraction_method == 'cnn'):
        path_cnn_trained = path_output + 'inception_resnet_v2_2016_08_30.ckpt'
    elif(feature_extraction_method == 'cnn_training'):
        path_cnn_trained = path_output + 'model.ckpt'
        
    run.run_command_line(name_images_database, labels_database, name_images_query, labels_query, path_cnn_trained, path_output, feature_extraction_method,distance, number_of_images,list_of_parameters, 'simple', searching_method = 'bf', isEvaluation = False)
        
def feature_extraction_method_name(opt_numb):
    global list_of_parameters
    
    if(opt_numb == 1):
        return "glcm"
    elif(opt_numb == 2):
        return "hog"
    elif(opt_numb == 3):
        return "lbp"
    elif(opt_numb == 4):
        return "cnn"
    elif(opt_numb == 5):
        return "fotf"
    elif(opt_numb == 6):
        return "daisy"

def distance_name(opt_numb):
    if(opt_numb == 1):
        return "ed"
    elif(opt_numb == 2):
        return "cd"
    elif(opt_numb == 3):
        return "id"
    elif(opt_numb == 4):
        return "cs"
    elif(opt_numb == 5):
        return "pcc"
    elif(opt_numb == 6):
        return "csd"
    elif(opt_numb == 7):
        return "kld"
    elif(opt_numb == 8):
        return "jd"
    elif(opt_numb == 9):
        return "ksd"
    elif(opt_numb == 10):
        return "cmd"
    elif(opt_numb == 11):
        return "emd"
    
def update_classes(path):
    global extension_classes
    name_classes =[]
    extension_classes=[]
    list_classes.delete(0, END)
    folders = os.listdir(path)
    for i in folders:
        if(not i.count(".")):
            for ext in extension: 
                numb = len(glob.glob(path + i + "/" + "*" + ext))
                if(numb > 0):
                    extension_classes.append(ext[1:])
                    name_classes.append(i)
                    list_classes.insert(END,str(numb) + " images of the class \"" + i + "\"")
                    break
    
def read_path_database():
    global path_database
    input_path_database.delete(0,END)#limpando o campo de texto
    input_path_folder_retrieval.delete(0,END)
    
    path_database = askdirectory()
    path_database = path_database + "/"
    input_path_database.insert(0, path_database)
    
    update_classes(path_database + "database/")

def read_path_image():
    global path_image
    global path_folder_retrieval
    path_folder_retrieval = ""
    input_path_folder_retrieval.delete(0,END)
    input_path_image.delete(0, END)#limpando o campo de texto
    
    path_folder_retrieval = askopenfilename()
    input_path_image.insert(0, path_folder_retrieval)

def read_path_folder():
    global path_folder_retrieval
    global path_image
    path_image = ""
    input_path_image.delete(0,END)#limpando o campo de texto
    input_path_folder_retrieval.delete(0,END)
    
    path_folder_retrieval = askdirectory()
    path_folder_retrieval = path_folder_retrieval + "/"
    input_path_folder_retrieval.insert(0, path_folder_retrieval)

    #display the message processing
    #processing = Toplevel() 
    #label_processing = Label(processing, text = "processing.")
    #label_processing.grid(row=0,column=0)

    
    '''
    top = Toplevel()
    top.geometry('+170+200')
    Label(top, text='Result save in: ' + path_database + "result" + "_" + feature_extraction_method + "_" + distance + "_" + str(number_of_images) + ".png").grid(row=0, column=0, sticky='E', padx=5, pady=2)
    '''
    
def help_function():
    global help_frame
    help_frame = Toplevel() 
    help_frame.title("pyCBIR - Help")
    help_frame.geometry('+165+200')
    
    list_help = Listbox(help_frame,width=100,height=12)
    help_text = []
    help_text.append("\n")
    help_text.append("\n")
    help_text.append("    -  Click in the button “Load” inside the frame “Set path” and choice the folder that contains the database;")
    help_text.append("    -  The selected folder has to have a folder named database and inside it, each images of different class has to be in a different folder;")
    help_text.append("    -  If the structure of the database is correct, the name of all classes and the number of images will be shown in the classes frame;")
    help_text.append("    -  Select the feature extraction method and the distance;")
    help_text.append("    -  Click in the button “Load” of “Path image” in order to select only an image to the retrieval;")
    help_text.append("    -  In order to select more than one image to retrieval, click in the button “Load” of “Path folder”;")
    help_text.append("    -  Finally, fill the field “N. of images” with an integer number and click in the button “Retrieval”;")
    help_text.append("    -  Click in the button “Exit” to close the interface.")
    
    for i in help_text:
        list_help.insert(END,i)
    list_help.grid(row=0,column=0)
    #Label(top1, text= help_text).grid(row=0, column=0, sticky='E', padx=5, pady=2)
    bt_exit_top = Button(help_frame, height=2, width=10, text='Exit', command=close_window_top).grid(row=1, column=0, sticky='N', padx=5, pady=2)

def close_window_top(): 
    help_frame.destroy()

def cancel(): 
    
    if(feature_extraction_method == "glcm"):
        lb_distance_glcm.grid_forget()
        lb_gray_levels_glcm.grid_forget()
        input_distance_glcm.grid_forget()
        input_gray_levels_glcm.grid_forget()
        
    elif(feature_extraction_method == "hog"):
        lb_block_hog.grid_forget()
        lb_window_hog.grid_forget()
        input_block_hog.grid_forget()
        input_window_hog.grid_forget()

    elif(feature_extraction_method == "lbp"):
        lb_radius_lbp.grid_forget()
        input_radius_lbp.grid_forget()
    
    elif(feature_extraction_method == "cnn" or feature_extraction_method == "cnn_probability"):
        lb_learning_rate_cnn.grid_forget()
        lb_epochs.grid_forget()
        input_learning_rate_cnn.grid_forget()
        input_epochs_cnn.grid_forget()
        check_button.forget()
        
    parameters_frame.withdraw()
       
def exit_function():
    exit()
    
path_database = "" #variable containing the path to the database
path_image = "" #variable containing the path to the image
path_folder_retrieval = "" #variable containing the path to the folder of the images to retrieval
extension = [".jpg", ".JPG",".jpeg",".JPEG", ".tif",".TIF", ".bmp", ".BMP", ".png", ".PNG"]#extension that the system accept
extension_classes=[] #list containing the extension of each class
name_classes = [] #list containing the name of each class
feature_extraction_method = ""
distance = ""
number_of_images = 0
list_of_parameters =[]

window = Tk() 
window.title("pyCBIR") 
window.geometry('820x800+200+100') 

#creating the first frame
frame_set_path = LabelFrame(window, text=" Set path: ")
frame_feature_extraction = LabelFrame(window, text=" Feature extraction method: ")
frame_distance = LabelFrame(window, text=" Distance: ")
frame_classes = LabelFrame(window, text=" Classes: ")
frame_retrieval = LabelFrame(window, text=" Retrieval: ")

#Text input
input_path_database = Entry(frame_set_path,width=60) 
input_path_image = Entry(frame_retrieval, width=30)
input_path_folder_retrieval = Entry(frame_retrieval, width=30)
input_number_of_images = Entry(frame_retrieval, width=8)

#Buttons
bt_load_path_database = Button(frame_set_path, height=2, width=10, text='Load', command=read_path_database)
bt_load_path_image = Button(frame_retrieval, height=2, width=10, text='Load', command=read_path_image) 
bt_load_path_folder_retrieval = Button(frame_retrieval, height=2, width=10, text='Load', command=read_path_folder) 
bt_retrieval = Button(frame_retrieval, height=2, width=10, text='Retrieval', command=retrieval) 
bt_help = Button(window, height=2, width=10, text='Help', command=help_function) 
bt_exit = Button(window, height=2, width=10, text='Exit', command=exit_function) 

#Radio buttons
var1 = IntVar()
rb_feature1 = Radiobutton(frame_feature_extraction, text="Gray-Level Co-occurence Matrix", variable=var1, value=1, command=get_parameters)
rb_feature2 = Radiobutton(frame_feature_extraction, text="Histogram of Oriented Gradients", variable=var1, value=2, command=get_parameters)
rb_feature3 = Radiobutton(frame_feature_extraction, text="Local Binary Pattern", variable=var1, value=3, command=get_parameters)
rb_feature4 = Radiobutton(frame_feature_extraction, text="Convolutional Neural Network", variable=var1, value=4, command=get_parameters)
rb_feature5 = Radiobutton(frame_feature_extraction, text="Histogram (First Order Texture)", variable=var1, value=5)
rb_feature6 = Radiobutton(frame_feature_extraction, text="Daisy", variable=var1, value=6, command=get_parameters)


var2 = IntVar()
rb_distance1 = Radiobutton(frame_distance, text="Euclidean Distance", variable=var2, value=1)
rb_distance2 = Radiobutton(frame_distance, text="Cityblock Distance", variable=var2, value=2)
rb_distance3 = Radiobutton(frame_distance, text="Infinity Distance", variable=var2, value=3)
rb_distance4 = Radiobutton(frame_distance, text="Cosine Similarity", variable=var2, value=4)
rb_distance5 = Radiobutton(frame_distance, text="Pearson Correlation Coefficient", variable=var2, value=5)
rb_distance6 = Radiobutton(frame_distance, text="Chi-Square Dissimilarity", variable=var2, value=6)
rb_distance7 = Radiobutton(frame_distance, text="Kullback-Liebler Divergence", variable=var2, value=7)
rb_distance8 = Radiobutton(frame_distance, text="Jeffrey Divergence", variable=var2, value=8)
rb_distance9 = Radiobutton(frame_distance, text="Kolmogorov-Smirnov Divergence", variable=var2, value=9)
rb_distance10 = Radiobutton(frame_distance, text="Cramer-von Mises Divergence", variable=var2, value=10)
rb_distance11 = Radiobutton(frame_distance, text="Earth Movers Distance", variable=var2, value=11)

#List containing classes
list_classes = Listbox(frame_classes,width=25) 

#parameters
parameters_frame = Toplevel()
bt_ok = Button(parameters_frame, height=2, width=10, text='Ok', command=salve_values) 
bt_parameters_cancel = Button(parameters_frame, height=2, width=10, text='Cancel', command=cancel) 

#GLCM
lb_distance_glcm = Label(parameters_frame, text='Distance: ')
input_distance_glcm = Entry(parameters_frame, width=8)
lb_gray_levels_glcm = Label(parameters_frame, text='Gray levels: ')
input_gray_levels_glcm = Entry(parameters_frame, width=8)

#HOG
lb_block_hog = Label(parameters_frame, text='Cells: ')
lb_window_hog = Label(parameters_frame, text='Blocks: ')
input_block_hog = Entry(parameters_frame, width=8)
input_window_hog = Entry(parameters_frame, width=8)

#LBP
lb_radius_lbp = Label(parameters_frame, text='Radius: ')
input_radius_lbp = Entry(parameters_frame, width=8)

#CNN
lb_learning_rate_cnn = Label(parameters_frame, text='Learning rate: ')
lb_epochs = Label(parameters_frame, text='Number of epochs: ')
input_learning_rate_cnn = Entry(parameters_frame, width=8)
input_epochs_cnn = Entry(parameters_frame, width=8)
var3 = IntVar()
check_button = Checkbutton(parameters_frame, text="Use Inception",onvalue=1, offvalue=0, variable=var3)

#Closing the window
parameters_frame.withdraw()

#Frame of the database path
frame_set_path.grid(row=0, columnspan=7, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
Label(frame_set_path, text='Path database: ').grid(row=0, column=0, sticky='E', padx=5, pady=2)
input_path_database.grid(row=0,column=1,columnspan=7, sticky="WE", pady=3)
bt_load_path_database.grid(row=0,column=8,sticky='W', padx=5, pady=2)

#Frame of the feature extraction method
frame_feature_extraction.grid(row=2,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_feature1.grid(row=3,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_feature2.grid(row=4,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_feature5.grid(row=5,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_feature3.grid(row=6,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_feature4.grid(row=7,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_feature6.grid(row=8,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_feature2.select()

#Frame of the distance 
frame_distance.grid(row=2,column=3,columnspan=4, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_distance1.grid(row=3,column=3,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
#rb_distance2.grid(row=4,column=3,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_distance3.grid(row=4,column=3,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_distance4.grid(row=5,column=3,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_distance5.grid(row=6,column=3,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_distance6.grid(row=7,column=3,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_distance7.grid(row=3,column=5,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_distance8.grid(row=4,column=5,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_distance9.grid(row=5,column=5,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_distance10.grid(row=6,column=5,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_distance2.grid(row=7,column=5,columnspan=2, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
rb_distance1.select()

#Frame of the list of classes
frame_classes.grid(row = 9, column=0, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
list_classes.grid(row=9,column=0, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)

#Frame of the retrieval
frame_retrieval.grid(row = 9, column=3, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
Label(frame_retrieval, text='Path image: ').grid(row=0, column=0, sticky='E', padx=5, pady=2)
input_path_image.grid(row = 0, column=1, sticky='E')
bt_load_path_image.grid(row = 0, column=2, sticky='E')
Label(frame_retrieval, text='Path folder: ').grid(row=1, column=0, sticky='E', padx=5, pady=2)
input_path_folder_retrieval.grid(row = 1, column=1, sticky='E')
bt_load_path_folder_retrieval.grid(row = 1, column=2, sticky='E')

Label(frame_retrieval, text='N. of images: ').grid(row=2, column=0, sticky='E', padx=5, pady=2)
input_number_of_images.grid(row = 2, column=1, sticky='W')
input_number_of_images.insert(0, 10)#valor inicial
bt_retrieval.grid(row = 2, column=1, sticky='N')

bt_help.grid(row=10,column=3, sticky='NW')
bt_exit.grid(row=10,column=3, sticky='N')

window.mainloop()
