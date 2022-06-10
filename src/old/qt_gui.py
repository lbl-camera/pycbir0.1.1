# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pyCBIR.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel
from PyQt5.QtGui import QIcon, QPixmap
import run
import numpy as np
import glob
import csv

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from skimage.io import imread
import matplotlib

class Ui_pyCBIR(object):
    def setupUi(self, pyCBIR):
        pyCBIR.setObjectName("pyCBIR")
        pyCBIR.setEnabled(True)
        #pyCBIR.setFixedSize(1236, 700)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(pyCBIR.sizePolicy().hasHeightForWidth())
        pyCBIR.setSizePolicy(sizePolicy)
        pyCBIR.setWindowTitle("pyCBIR")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        pyCBIR.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(pyCBIR)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 10, 239, 153))
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox.setGeometry(QtCore.QRect(13, 28, 220, 18))
        self.checkBox.setObjectName("checkBox")
        self.checkBox_2 = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_2.setGeometry(QtCore.QRect(13, 47, 212, 18))
        self.checkBox_2.setCheckable(True)
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_3 = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_3.setGeometry(QtCore.QRect(13, 66, 188, 18))
        self.checkBox_3.setChecked(True)
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_4 = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_4.setGeometry(QtCore.QRect(13, 85, 142, 18))
        self.checkBox_4.setCheckable(True)
        self.checkBox_4.setObjectName("checkBox_4")
        self.checkBox_5 = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_5.setGeometry(QtCore.QRect(13, 104, 197, 18))
        self.checkBox_5.setCheckable(True)
        self.checkBox_5.setObjectName("checkBox_5")
        self.checkBox_6 = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_6.setGeometry(QtCore.QRect(13, 123, 113, 18))
        self.checkBox_6.setCheckable(True)
        self.checkBox_6.setObjectName("checkBox_6")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 160, 241, 211))
        self.groupBox_2.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.groupBox_2.setFlat(False)
        self.groupBox_2.setObjectName("groupBox_2")
        self.radioButton_7 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_7.setGeometry(QtCore.QRect(13, 29, 132, 18))
        self.radioButton_7.setChecked(True)
        self.radioButton_7.setObjectName("radioButton_7")
        self.radioButton_8 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_8.setGeometry(QtCore.QRect(13, 46, 117, 18))
        self.radioButton_8.setObjectName("radioButton_8")
        self.radioButton_9 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_9.setGeometry(QtCore.QRect(13, 63, 129, 18))
        self.radioButton_9.setObjectName("radioButton_9")
        self.radioButton_10 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_10.setGeometry(QtCore.QRect(13, 80, 141, 18))
        self.radioButton_10.setObjectName("radioButton_10")
        self.radioButton_16 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_16.setGeometry(QtCore.QRect(13, 97, 164, 18))
        self.radioButton_16.setObjectName("radioButton_16")
        self.radioButton_11 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_11.setGeometry(QtCore.QRect(13, 114, 187, 18))
        self.radioButton_11.setObjectName("radioButton_11")
        self.radioButton_13 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_13.setGeometry(QtCore.QRect(13, 131, 134, 18))
        self.radioButton_13.setObjectName("radioButton_13")
        self.radioButton_12 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_12.setGeometry(QtCore.QRect(13, 148, 213, 18))
        self.radioButton_12.setObjectName("radioButton_12")
        self.radioButton_15 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_15.setGeometry(QtCore.QRect(13, 165, 138, 18))
        self.radioButton_15.setObjectName("radioButton_15")
        self.radioButton_14 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_14.setGeometry(QtCore.QRect(13, 182, 213, 17))
        self.radioButton_14.setObjectName("radioButton_14")
        #self.frame = QtWidgets.QFrame(self.centralwidget)
        #self.frame.setEnabled(False)
        #self.frame.setGeometry(QtCore.QRect(270, 30, 951, 791))
        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        #self.frame.setSizePolicy(sizePolicy)
        #self.frame.setAcceptDrops(True)
        #self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        #self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        #self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(710, 10, 101, 16))
        self.label.setObjectName("label")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(20, 480, 241, 381))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.label_2.setGeometry(QtCore.QRect(10, 30, 111, 20))
        self.label_2.setObjectName("label_2")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit.setGeometry(QtCore.QRect(120, 30, 51, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_6.setGeometry(QtCore.QRect(0, 50, 241, 141))
        self.groupBox_6.setObjectName("groupBox_6")
        self.lineEdit_8 = QtWidgets.QLineEdit(self.groupBox_6)
        self.lineEdit_8.setGeometry(QtCore.QRect(10, 110, 221, 21))
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.lineEdit_9 = QtWidgets.QLineEdit(self.groupBox_6)
        self.lineEdit_9.setGeometry(QtCore.QRect(10, 50, 221, 21))
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_7.setEnabled(True)
        self.pushButton_7.setGeometry(QtCore.QRect(40, 20, 141, 32))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_8.setEnabled(True)
        self.pushButton_8.setGeometry(QtCore.QRect(40, 80, 141, 32))
        self.pushButton_8.setObjectName("pushButton_8")
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_5.setGeometry(QtCore.QRect(0, 200, 241, 141))
        self.groupBox_5.setObjectName("groupBox_5")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.groupBox_5)
        self.lineEdit_4.setGeometry(QtCore.QRect(10, 110, 221, 21))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.groupBox_5)
        self.lineEdit_6.setGeometry(QtCore.QRect(10, 50, 221, 21))
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_4.setEnabled(True)
        self.pushButton_4.setGeometry(QtCore.QRect(40, 20, 141, 32))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_5.setEnabled(True)
        self.pushButton_5.setGeometry(QtCore.QRect(40, 80, 141, 32))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_6.setGeometry(QtCore.QRect(60, 200, 110, 32))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton.setEnabled(True)
        self.pushButton.setGeometry(QtCore.QRect(70, 350, 110, 32))
        self.pushButton.setObjectName("pushButton")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(20, 370, 241, 111))
        self.groupBox_4.setObjectName("groupBox_4")
        self.radioButton_17 = QtWidgets.QRadioButton(self.groupBox_4)
        self.radioButton_17.setGeometry(QtCore.QRect(13, 29, 93, 18))
        self.radioButton_17.setChecked(True)
        self.radioButton_17.setObjectName("radioButton_17")
        self.radioButton_18 = QtWidgets.QRadioButton(self.groupBox_4)
        self.radioButton_18.setGeometry(QtCore.QRect(13, 46, 64, 18))
        self.radioButton_18.setObjectName("radioButton_18")
        self.radioButton_19 = QtWidgets.QRadioButton(self.groupBox_4)
        self.radioButton_19.setGeometry(QtCore.QRect(13, 63, 73, 18))
        self.radioButton_19.setObjectName("radioButton_19")
        self.radioButton_20 = QtWidgets.QRadioButton(self.groupBox_4)
        self.radioButton_20.setGeometry(QtCore.QRect(13, 80, 54, 18))
        self.radioButton_20.setObjectName("radioButton_20")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(290, 840, 491, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(False)
        self.progressBar.setObjectName("progressBar")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(300, 830, 271, 16))
        self.label_4.setObjectName("label_4")
        self.label.raise_()
        self.groupBox_4.raise_()
        self.progressBar.raise_()
        self.label_4.raise_()
        #self.frame.raise_()
        self.groupBox_3.raise_()
        self.groupBox_2.raise_()
        self.groupBox.raise_()
        pyCBIR.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(pyCBIR)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1236, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuFeatures = QtWidgets.QMenu(self.menubar)
        self.menuFeatures.setObjectName("menuFeatures")
        pyCBIR.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(pyCBIR)
        self.statusbar.setObjectName("statusbar")
        pyCBIR.setStatusBar(self.statusbar)
        self.actionAbout_pyCBIR = QtWidgets.QAction(pyCBIR)
        self.actionAbout_pyCBIR.setObjectName("actionAbout_pyCBIR")
        self.actionHow_to_Use = QtWidgets.QAction(pyCBIR)
        self.actionHow_to_Use.setObjectName("actionHow_to_Use")
        self.actionOpen_File = QtWidgets.QAction(pyCBIR)
        self.actionOpen_File.setObjectName("actionOpen_File")
        self.actionOpen_Database = QtWidgets.QAction(pyCBIR)
        self.actionOpen_Database.setObjectName("actionOpen_Database")
        self.menuFile.addAction(self.actionOpen_File)
        self.menuFile.addAction(self.actionOpen_Database)
        self.menuHelp.addAction(self.actionHow_to_Use)
        self.menuHelp.addAction(self.actionAbout_pyCBIR)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuFeatures.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(pyCBIR)
        QtCore.QMetaObject.connectSlotsByName(pyCBIR)

        self.pushButton_7.clicked.connect(self.loadDatabaseFile)
        self.pushButton_8.clicked.connect(self.loadRetrievalFile)

        self.pushButton_4.clicked.connect(self.loadDatabasePath)
        self.pushButton_5.clicked.connect(self.loadRetrievalPath)

        #self.pushButton.clicked.connect(self.returnInformation)
        self.pushButton.clicked.connect(self.returnInformation)
        
        
        
    def loadDatabaseFile(self,pyCBIR):
        file = QFileDialog.getOpenFileName(None,'Open file', '/Users/romuere/Dropbox/CBIR/cells2/databases','CSV Files (*.csv)')
        self.lineEdit_9.setText(str(file[0]))

    def loadRetrievalFile(self,pyCBIR):
        file = QFileDialog.getOpenFileName(None,'Open file', '/Users/romuere/Dropbox/CBIR/cells2/retrieval','CSV Files (*.csv)')
        self.lineEdit_8.setText(str(file[0]))
        self.returnPathOutput(self)
        
    def loadDatabasePath(self,pyCBIR):
        file = QFileDialog.getExistingDirectory(None,'Open path', '/Users/romuere/Dropbox/CBIR/cells2/databases')
        self.lineEdit_6.setText(str(file+'/'))

    def loadRetrievalPath(self,pyCBIR):
        file = QFileDialog.getExistingDirectory(None,'Open path', '/Users/romuere/Dropbox/CBIR/cells2/retrieval')
        self.lineEdit_4.setText(str(file+'/'))
        self.returnPathOutput(self)
        
    def retranslateUi(self, pyCBIR):
        _translate = QtCore.QCoreApplication.translate
        self.groupBox.setTitle(_translate("pyCBIR", "Feature Extraction Methods"))
        self.checkBox.setText(_translate("pyCBIR", "Gray-Level Co-Occurrence Matrix"))
        self.checkBox_2.setText(_translate("pyCBIR", "Histogram of Oriented Gradients"))
        self.checkBox_3.setText(_translate("pyCBIR", "First Order Texture Features"))
        self.checkBox_4.setText(_translate("pyCBIR", "Local Binary Pattern"))
        self.checkBox_5.setText(_translate("pyCBIR", "Convolutional Neural Network"))
        self.checkBox_6.setText(_translate("pyCBIR", "Daisy Features"))
        self.groupBox_2.setTitle(_translate("pyCBIR", "Similarity Metrics"))
        self.radioButton_7.setText(_translate("pyCBIR", "Euclidian Distance"))
        self.radioButton_8.setText(_translate("pyCBIR", "Infinity Distance"))
        self.radioButton_9.setText(_translate("pyCBIR", "Cossine Similarity"))
        self.radioButton_10.setText(_translate("pyCBIR", "Pearson Correlation"))
        self.radioButton_16.setText(_translate("pyCBIR", "Chi-Square Dissimilarity"))
        self.radioButton_11.setText(_translate("pyCBIR", "Kullback-Leibler Divergence"))
        self.radioButton_13.setText(_translate("pyCBIR", "Jeffrey Divergence"))
        self.radioButton_12.setText(_translate("pyCBIR", "Kolmogorov-Smirnov Divergence"))
        self.radioButton_15.setText(_translate("pyCBIR", "Cramer Divergence"))
        self.radioButton_14.setText(_translate("pyCBIR", "Earth Movers Distance"))
        self.label.setText(_translate("pyCBIR", "Retrieval Result"))
        self.groupBox_3.setTitle(_translate("pyCBIR", "Retrieval Options"))
        self.label_2.setText(_translate("pyCBIR", "Retrieval Number:"))
        self.lineEdit.setText(_translate("pyCBIR", "10"))
        self.groupBox_6.setTitle(_translate("pyCBIR", "By CSV File"))
        self.lineEdit_8.setText(_translate("pyCBIR", ""))
        self.lineEdit_9.setText(_translate("pyCBIR", ""))
        self.pushButton_7.setText(_translate("pyCBIR", "Load Database File"))
        self.pushButton_8.setText(_translate("pyCBIR", "Load Retrieval File"))
        self.groupBox_5.setTitle(_translate("pyCBIR", "By Path"))
        self.lineEdit_4.setText(_translate("pyCBIR", ""))
        self.lineEdit_6.setText(_translate("pyCBIR", ""))
        self.pushButton_4.setText(_translate("pyCBIR", "Load Database Path"))
        self.pushButton_5.setText(_translate("pyCBIR", "Load Retrieval Path"))
        self.pushButton_6.setText(_translate("pyCBIR", "Run pyCBIR"))
        self.pushButton.setText(_translate("pyCBIR", "Run pyCBIR"))
        self.groupBox_4.setTitle(_translate("pyCBIR", "Searching Methods"))
        self.radioButton_17.setText(_translate("pyCBIR", "Brute Force"))
        self.radioButton_18.setText(_translate("pyCBIR", "R Tree"))
        self.radioButton_19.setText(_translate("pyCBIR", "KD Tree"))
        self.radioButton_20.setText(_translate("pyCBIR", "LSH "))
        self.label_4.setText(_translate("pyCBIR", "Progress Status (computing segnatures)"))
        self.menuFile.setTitle(_translate("pyCBIR", "File"))
        self.menuHelp.setTitle(_translate("pyCBIR", "Help"))
        self.menuFeatures.setTitle(_translate("pyCBIR", "Features"))
        self.actionAbout_pyCBIR.setText(_translate("pyCBIR", "About pyCBIR"))
        self.actionHow_to_Use.setText(_translate("pyCBIR", "How to Use"))
        self.actionOpen_File.setText(_translate("pyCBIR", "Open File"))
        self.actionOpen_Database.setText(_translate("pyCBIR", "Open Database"))



    def returnPathOutput(self,pyCBIR):
        msg = QMessageBox.information(None,'Path Output','Now, you have to set the directory where the output files will be saved!')

        path_output = QFileDialog.getExistingDirectory(None,'Select the path output', '/Users/romuere/Dropbox/CBIR/cells2/output')
        self.path_output = path_output+'/'

    def returnInformation(self,pyCBIR):
        
        #path_output = self.returnPathOutput(self)

        feature_extraction_method = ''
        if self.checkBox.isChecked():
            feature_extraction_method = 'glcm'
        if self.checkBox_2.isChecked():
            feature_extraction_method = 'hog'
        if self.checkBox_3.isChecked():
            feature_extraction_method = 'fotf'
        if self.checkBox_4.isChecked():
            feature_extraction_method = 'lbp'
        if self.checkBox_5.isChecked():
            feature_extraction_method = 'cnn'
        if self.checkBox_6.isChecked():
            feature_extraction_method = 'daisy'

        similarity_metric = ''
        if self.radioButton_7.isChecked():
            similarity_metric = 'ed'
        elif self.radioButton_8.isChecked():
            similarity_metric = 'id'
        elif self.radioButton_9.isChecked():
            similarity_metric = 'cs'
        elif self.radioButton_10.isChecked():
            similarity_metric = 'pcc'
        elif self.radioButton_16.isChecked():
            similarity_metric = 'csd'
        elif self.radioButton_11.isChecked():
            similarity_metric = 'kld'
        elif self.radioButton_13.isChecked():
            similarity_metric = 'ld'
        elif self.radioButton_12.isChecked():
            similarity_metric = 'ksd'
        elif self.radioButton_15.isChecked():
            similarity_metric = 'cmd'
        elif self.radioButton_14.isChecked():
            similarity_metric = 'cd'

        searching_method = ''
        if self.radioButton_17.isChecked():
            searching_method = 'bf'
        elif self.radioButton_18.isChecked():
            searching_method = 'r'
        elif self.radioButton_19.isChecked():
            searching_method = 'kd'
        elif self.radioButton_20.isChecked():
            searching_method = 'lsh'

        retrieval_number = int(self.lineEdit.text())
        preprocessing_method = 'simple'
        path_cnn_trained = self.path_output + 'model.ckpt'
        list_of_parameters = ['0','0','0']

        if (self.lineEdit_6.text() != ""):
            fname_database,labels_database = informationPath(self.lineEdit_6.text())
            fname_retrieval,labels_retrieval = informationPath(self.lineEdit_4.text())
        elif (self.lineEdit_9.text() != ""):
            fname_database,labels_database = informationFile(self.lineEdit_9.text())
            fname_retrieval,labels_retrieval = informationFile(self.lineEdit_8.text())

        _,_,file = run.run_command_line(fname_database,labels_database,fname_retrieval,labels_retrieval,path_cnn_trained,self.path_output,feature_extraction_method,similarity_metric,retrieval_number,list_of_parameters,preprocessing_method,searching_method, isEvaluation = False)

        print(file)
        #file = '/Users/romuere/Dropbox/new_Database/todas.jpg'
        # Create widget
        self.w = QtWidgets.QWidget(self.centralwidget)
        self.w.setGeometry(QtCore.QRect(270, 30, 951, 791))
        self.w.setEnabled(True)
        self.w.setAcceptDrops(True)
        self.w.setObjectName("image")
        label = QLabel(self.w)
        pixmap = QPixmap(file)
        if (pixmap.height() < 700):
            pixmap = pixmap.scaled(900,pixmap.height())
        else:
            pixmap = pixmap.scaled(900,800)
        label.setPixmap(pixmap)
        self.w.show()
        #self.w.show()
        #self.resize(pixmap.width(),pixmap.height())

        #m = PlotCanvas(self, width=10, height=10)
        #m.move(50,50)

"""
        self.figure = matplotlib.figure.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.plot2(self)

    def plot2(self,pyCBIR):
        self.figure.clf()
        ax3 = self.figure.add_subplot(111)
        ax3.imshow(imread(self.file))
        self.canvas.draw()
"""

def informationPath(folder):
    folders = glob.glob(folder+"*/")
    fname_database = []
    labels_database = np.empty(0)
    cont = 0
    if len(folders) != 0:
        for id,f in enumerate(folders):
            test = f.split('/')
            if not(test[-2][0]== '.'):
                files = glob.glob(f+'/*')
                labels_database = np.append(labels_database, np.zeros(len(files))+cont)
                fname_database = fname_database+files
                cont += 1
    else:
        fname_database = glob.glob(folder+'*')
        labels_database = np.append(labels_database, np.zeros(len(fname_database)))
    return fname_database,labels_database

def informationFile(folder):
    reader = csv.reader(open(folder))
    fname = list(reader)
    fname = np.array(fname,dtype = np.str)
    labels = fname[:,1].astype(np.int)
    fname = list(fname[:,0])
    return fname,labels

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=10, height=10, dpi=100,f= ''):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        ax = self.figure.add_subplot(111)
        ax.imshow(imread(f))
        self.draw()
