'''
Created on 15 de sep de 2016
Last Modified on 26 de set de 2019

@author: romuere, flavio, dani
'''
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pycbir.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel, QComboBox
from PyQt5.QtGui import QIcon, QPixmap
import numpy as np
import glob
import csv
import os

class Ui_pyCBIR(object):

    def setupUi(self, pyCBIR):
        pyCBIR.setObjectName("pyCBIR")
        pyCBIR.resize(946, 730)
        self.centralwidget = QtWidgets.QWidget(pyCBIR)
        self.centralwidget.setObjectName("centralwidget")
        """
        label = QLabel(self.centralwidget)
        pixmap = QPixmap('logo_2.png')
        label.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())
        #self.logo = QtWidgets.QLabel(label)
        """

        self.w = QtWidgets.QWidget(self.centralwidget)
        self.w.setGeometry(QtCore.QRect(4, 550, 951, 791))
        self.w.setEnabled(True)
        self.w.setAcceptDrops(True)
        self.w.setObjectName("image")
        label = QLabel(self.w)
        pixmap = QPixmap("interface/logo_pycbir.png")
        pixmap = pixmap.scaledToHeight(70,QtCore.Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        self.w.show()


        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(3, 0, 131, 141))
        self.groupBox.setObjectName("groupBox")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(3, 20, 101, 20))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QtCore.QRect(3, 40, 97, 18))
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_3.setGeometry(QtCore.QRect(3, 60, 141, 18))
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_3.setChecked(True)
        self.radioButton_4 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_4.setGeometry(QtCore.QRect(3, 80, 97, 18))
        self.radioButton_4.setObjectName("radioButton_4")
        self.radioButton_5 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_5.setGeometry(QtCore.QRect(3, 100, 97, 18))
        self.radioButton_5.setObjectName("radioButton_5")
        #self.radioButton_5.setChecked(True)
        self.radioButton_6 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_6.setGeometry(QtCore.QRect(3, 120, 97, 18))
        self.radioButton_6.setObjectName("radioButton_6")

        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(3, 140, 131, 88))
        self.groupBox_2.setObjectName("groupBox_2")
        self.radioButton_7 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_7.setGeometry(QtCore.QRect(3, 20, 101, 20))
        self.radioButton_7.setObjectName("radioButton_7")
        self.radioButton_7.setChecked(True)
        self.radioButton_8 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_8.setGeometry(QtCore.QRect(3, 40, 97, 20))
        self.radioButton_8.setObjectName("radioButton_8")
        self.radioButton_9 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_9.setGeometry(QtCore.QRect(3, 60, 97, 20))
        self.radioButton_9.setObjectName("radioButton_9")

        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(3, 230, 131, 471))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setGeometry(QtCore.QRect(3, 20, 111, 16))
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit.setGeometry(QtCore.QRect(42, 40, 50, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_4.setGeometry(QtCore.QRect(3, 130, 131, 141))
        self.groupBox_4.setObjectName("groupBox_4")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.groupBox_4)
        self.lineEdit_3.setGeometry(QtCore.QRect(0, 50, 131, 21))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_3.setGeometry(QtCore.QRect(0, 20, 130, 32))
        self.pushButton_3.setObjectName("pushButton_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.groupBox_4)
        self.lineEdit_4.setGeometry(QtCore.QRect(0, 110, 131, 21))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_4.setGeometry(QtCore.QRect(0, 80, 130, 32))
        self.pushButton_4.setObjectName("pushButton_4")
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_5.setGeometry(QtCore.QRect(-1, 280, 131, 141))
        self.groupBox_5.setObjectName("groupBox_5")
        #self.lineEdit_5 = QtWidgets.QLineEdit(self.groupBox_5)
        #self.lineEdit_5.setGeometry(QtCore.QRect(3, 50, 131, 21))
        #self.lineEdit_5.setObjectName("lineEdit_5")
        #self.lineEdit_6 = QtWidgets.QLineEdit(self.groupBox_5)
        #self.lineEdit_6.setGeometry(QtCore.QRect(0, 110, 131, 21))
        #self.lineEdit_6.setObjectName("lineEdit_6")
        #self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_5)
        #self.pushButton_5.setGeometry(QtCore.QRect(0, 20, 130, 32))
        #self.pushButton_5.setObjectName("pushButton_5")
        #self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_5)
        #self.pushButton_6.setGeometry(QtCore.QRect(0, 80, 130, 32))
        #self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton.setGeometry(QtCore.QRect(10, 431, 110, 40))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 70, 110, 32))
        self.pushButton_2.setObjectName("pushButton_2")
        self.lineEdit.setText("10")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_2.setGeometry(QtCore.QRect(0, 100, 131, 21))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.line_2 = QtWidgets.QFrame(self.groupBox_3)
        self.line_2.setGeometry(QtCore.QRect(0, 120, 131, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.groupBox_3)
        self.line_3.setGeometry(QtCore.QRect(0, 273, 131, 10))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(self.groupBox_3)
        self.line_4.setGeometry(QtCore.QRect(0, 420, 131, 10))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(130, 0, 20, 681))
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        pyCBIR.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(pyCBIR)
        self.statusbar.setObjectName("statusbar")
        pyCBIR.setStatusBar(self.statusbar)

        self.retranslateUi(pyCBIR)
        QtCore.QMetaObject.connectSlotsByName(pyCBIR)

        #variables
            #defining defaults
        self.feature_extraction_method = 'fotf'
        self.similarity_metric = 'ed'
        self.list_of_parameters = []
        self.path_cnn_pre_trained = ''
        self.path_cnn_trained = ''
        self.path_save_cnn = ''

        #features
        self.radioButton.clicked.connect(self.radio_clicked)
        self.radioButton_2.clicked.connect(self.radio2_clicked)
        self.radioButton_4.clicked.connect(self.radio4_clicked)
        self.radioButton_5.clicked.connect(self.radio5_clicked)
        self.radioButton_6.clicked.connect(self.radio6_clicked)

        #metrics
        #self.radioButton_7.clicked.connect(self.radio7_clicked)

        #output
        self.pushButton_2.clicked.connect(self.returnPathOutput)

        #data
        self.pushButton_3.clicked.connect(self.loadDatabasePath)
        self.pushButton_4.clicked.connect(self.loadRetrievalPath)
        #self.pushButton_5.clicked.connect(self.loadDatabaseFile)
        #self.pushButton_6.clicked.connect(self.loadRetrievalFile)

        #run pyCBIR
        self.pushButton.clicked.connect(self.returnInformation)

        #show results


#----------------------Output------------------------------#
    def returnPathOutput(self,pyCBIR):
        cwd = os.getcwd()
        file = QFileDialog.getExistingDirectory(None,'Select the path output', cwd)
        self.lineEdit_2.setText(str(file+'/'))

#----------------------Load data---------------------------#
    def loadDatabaseFile(self,pyCBIR):
        cwd = self.lineEdit_2.text()
        file = QFileDialog.getOpenFileName(None,'Open file', cwd,'CSV Files (*.csv)')
        self.lineEdit_5.setText(str(file[0]))

    def loadRetrievalFile(self,pyCBIR):
        file = QFileDialog.getOpenFileName(None,'Open file', self.lineEdit_5.text(),'CSV Files (*.csv)')
        self.lineEdit_6.setText(str(file[0]))

    def loadDatabasePath(self,pyCBIR):
        cwd = self.lineEdit_2.text()
        file = QFileDialog.getExistingDirectory(None,'Open path', cwd)
        self.lineEdit_3.setText(str(file+'/'))

    def loadRetrievalPath(self,pyCBIR):
        file = QFileDialog.getExistingDirectory(None,'Open path', self.lineEdit_2.text())
        self.lineEdit_4.setText(str(file+'/'))

#----------------------GLCM Parameters---------------------#
    def radio_clicked(self,pyCBIR):
        self.mySubwindow=subwindow()
        self.mySubwindow.createWindow(200,120)
        self.mySubwindow.setWindowFlags(QtCore.Qt.CustomizeWindowHint)

        self.label1_glcm = QtWidgets.QLabel(self.mySubwindow)
        self.label1_glcm.setGeometry(QtCore.QRect(55, 0, 100, 16))
        self.label1_glcm.setObjectName("label1")
        self.label1_glcm.setText("GLCM options")

        #GLCM parameters
        self.label2_glcm = QtWidgets.QLabel(self.mySubwindow)
        self.label2_glcm.setGeometry(QtCore.QRect(0, 30, 70, 16))
        self.label2_glcm.setObjectName("label2")
        self.label2_glcm.setText("Distance:")
        self.lineEdit1_glcm = QtWidgets.QLineEdit(self.mySubwindow)
        self.lineEdit1_glcm.setGeometry(QtCore.QRect(75, 30, 30, 16))
        self.lineEdit1_glcm.setObjectName("lineEdit")
        self.lineEdit1_glcm.setText("1")


        self.label3_glcm = QtWidgets.QLabel(self.mySubwindow)
        self.label3_glcm.setGeometry(QtCore.QRect(0, 50, 110, 16))
        self.label3_glcm.setObjectName("label3")
        self.label3_glcm.setText("GrayLevels_old:")
        self.lineEdit2_glcm = QtWidgets.QLineEdit(self.mySubwindow)
        self.lineEdit2_glcm.setGeometry(QtCore.QRect(115, 50, 30, 16))
        self.lineEdit2_glcm.setObjectName("lineEdit")
        self.lineEdit2_glcm.setText("8")

        self.label4_glcm = QtWidgets.QLabel(self.mySubwindow)
        self.label4_glcm.setGeometry(QtCore.QRect(0, 70, 110, 16))
        self.label4_glcm.setObjectName("label4")
        self.label4_glcm.setText("GrayLevels_new:")
        self.lineEdit3_glcm = QtWidgets.QLineEdit(self.mySubwindow)
        self.lineEdit3_glcm.setGeometry(QtCore.QRect(115, 70, 30, 16))
        self.lineEdit3_glcm.setObjectName("lineEdit")
        self.lineEdit3_glcm.setText("8")


        self.buttom_glcm = QtWidgets.QPushButton(self.mySubwindow)
        self.buttom_glcm.setText("Ok")
        self.buttom_glcm.setGeometry(QtCore.QRect(50, 100, 100, 16))
        self.buttom_glcm.clicked.connect(self.b_glcm)

        self.mySubwindow.show()

    def b_glcm(self):
        self.list_of_parameters = [self.lineEdit1_glcm.text(),self.lineEdit2_glcm.text(),self.lineEdit3_glcm.text()]
        self.mySubwindow.close()


#----------------------HOG Parameters---------------------#
    def radio2_clicked(self,pyCBIR):
        self.mySubwindow=subwindow()
        self.mySubwindow.createWindow(200,90)
        self.mySubwindow.setWindowFlags(QtCore.Qt.CustomizeWindowHint)

        self.label1_hog = QtWidgets.QLabel(self.mySubwindow)
        self.label1_hog.setGeometry(QtCore.QRect(55, 0, 100, 16))
        self.label1_hog.setObjectName("label1")
        self.label1_hog.setText("HOG options")

        self.label2_hog = QtWidgets.QLabel(self.mySubwindow)
        self.label2_hog.setGeometry(QtCore.QRect(0, 30, 70, 16))
        self.label2_hog.setObjectName("label2")
        self.label2_hog.setText("Cells:")
        self.lineEdit1_hog = QtWidgets.QLineEdit(self.mySubwindow)
        self.lineEdit1_hog.setGeometry(QtCore.QRect(75, 30, 30, 16))
        self.lineEdit1_hog.setObjectName("lineEdit")
        self.lineEdit1_hog.setText("3")


        self.label3_hog = QtWidgets.QLabel(self.mySubwindow)
        self.label3_hog.setGeometry(QtCore.QRect(0, 50, 110, 16))
        self.label3_hog.setObjectName("label3")
        self.label3_hog.setText("Blocks:")
        self.lineEdit2_hog = QtWidgets.QLineEdit(self.mySubwindow)
        self.lineEdit2_hog.setGeometry(QtCore.QRect(75, 50, 30, 16))
        self.lineEdit2_hog.setObjectName("lineEdit")
        self.lineEdit2_hog.setText("3")


        self.buttom_hog = QtWidgets.QPushButton(self.mySubwindow)
        self.buttom_hog.setText("Ok")
        self.buttom_hog.setGeometry(QtCore.QRect(50, 70, 100, 16))
        self.buttom_hog.clicked.connect(self.b_hog)

        self.mySubwindow.show()

    def b_hog(self):
        try:
            celulas = int(self.lineEdit1_hog.text())
            if celulas <= 2:
                QMessageBox.information(None, 'pyCBIR', 'Number of cells must be greater than 2!')
            else:
                self.list_of_parameters = [self.lineEdit1_hog.text(), self.lineEdit2_hog.text()]
                self.mySubwindow.close()
        except ValueError:
            QMessageBox.information(None, 'pyCBIR', 'Number of cells must be greater than 2!')

#----------------------LBP Parameters---------------------#
    def radio4_clicked(self,pyCBIR):
        self.mySubwindow=subwindow()
        self.mySubwindow.createWindow(200,90)
        self.mySubwindow.setWindowFlags(QtCore.Qt.CustomizeWindowHint)

        self.label1_lbp = QtWidgets.QLabel(self.mySubwindow)
        self.label1_lbp.setGeometry(QtCore.QRect(55, 0, 100, 16))
        self.label1_lbp.setObjectName("label1")
        self.label1_lbp.setText("LBP options")

        self.label2_lbp = QtWidgets.QLabel(self.mySubwindow)
        self.label2_lbp.setGeometry(QtCore.QRect(0, 30, 70, 16))
        self.label2_lbp.setObjectName("label2")
        self.label2_lbp.setText("Neighbors:")
        self.lineEdit1_lbp = QtWidgets.QLineEdit(self.mySubwindow)
        self.lineEdit1_lbp.setGeometry(QtCore.QRect(75, 30, 30, 16))
        self.lineEdit1_lbp.setObjectName("lineEdit")
        self.lineEdit1_lbp.setText("16")

        self.label3_lbp = QtWidgets.QLabel(self.mySubwindow)
        self.label3_lbp.setGeometry(QtCore.QRect(0, 50, 110, 16))
        self.label3_lbp.setObjectName("label3")
        self.label3_lbp.setText("Radio:")
        self.lineEdit2_lbp = QtWidgets.QLineEdit(self.mySubwindow)
        self.lineEdit2_lbp.setGeometry(QtCore.QRect(75, 50, 30, 16))
        self.lineEdit2_lbp.setObjectName("lineEdit")
        self.lineEdit2_lbp.setText("2")


        self.buttom_lbp = QtWidgets.QPushButton(self.mySubwindow)
        self.buttom_lbp.setText("Ok")
        self.buttom_lbp.setGeometry(QtCore.QRect(50, 70, 100, 16))
        self.buttom_lbp.clicked.connect(self.b_lbp)

        self.mySubwindow.show()

    def b_lbp(self):
        self.list_of_parameters = [self.lineEdit1_lbp.text(),self.lineEdit2_lbp.text()]
        self.mySubwindow.close()


#----------------------CNN Parameters---------------------#
    def radio5_clicked(self,pyCBIR):
        self.mySubwindow=subwindow()
        self.mySubwindow.createWindow(400,200)
        self.mySubwindow.setWindowFlags(QtCore.Qt.CustomizeWindowHint)

        self.groupBox_ = QtWidgets.QGroupBox(self.mySubwindow)
        self.groupBox_.setGeometry(QtCore.QRect(0, 20, 400, 20))
        self.groupBox_.setObjectName("groupBox_")
        self.rb1 = QtWidgets.QRadioButton(self.groupBox_)
        self.rb1.setGeometry(QtCore.QRect(0, 0, 100, 20))
        self.rb1.setObjectName("rb1")
        self.rb1.setChecked(True)
        self.rb2 = QtWidgets.QRadioButton(self.groupBox_)
        self.rb2.setGeometry(QtCore.QRect(120, 0, 150, 20))
        self.rb2.setObjectName("rb2")
        self.rb3 = QtWidgets.QRadioButton(self.groupBox_)
        self.rb3.setGeometry(QtCore.QRect(270, 0, 150, 20))
        self.rb3.setObjectName("rb3")

        self.rb1.clicked.connect(self.rb1_clicked)

        self.rb2.clicked.connect(self.rb2_clicked)

        self.rb3.clicked.connect(self.rb3_clicked)


        self.rb1.setText("Train CNN")
        self.rb2.setText("Fine-Tuning CNN")
        self.rb3.setText("Pre-Trained CNN")

        self.label_cnn_type = QtWidgets.QLabel(self.mySubwindow)
        self.label_cnn_type.setGeometry(QtCore.QRect(0, 55, 150, 16))
        self.label_cnn_type.setObjectName("label1")
        self.label_cnn_type.setText("CNN Architecture: ")

        self.comboBox = QComboBox(self.mySubwindow)
        self.comboBox.addItem("lenet")
        self.comboBox.addItem("nasnet")
        self.comboBox.addItem("inception_resnet")
        self.comboBox.addItem("vgg16")
        self.comboBox.addItem("inception_v4")
        self.comboBox.setGeometry(QtCore.QRect(130, 50, 120, 25))


        self.label1 = QtWidgets.QLabel(self.mySubwindow)
        self.label1.setGeometry(QtCore.QRect(180, 0, 100, 16))
        self.label1.setObjectName("label1")
        self.label1.setText("CNN options")

        #CNN parameters
        self.label2 = QtWidgets.QLabel(self.mySubwindow)
        self.label2.setGeometry(QtCore.QRect(0, 100, 50, 16))
        self.label2.setObjectName("label2")
        self.label2.setText("Epochs:")
        self.lineEdit1 = QtWidgets.QLineEdit(self.mySubwindow)
        self.lineEdit1.setGeometry(QtCore.QRect(55, 100, 30, 16))
        self.lineEdit1.setObjectName("lineEdit")
        self.lineEdit1.setText("1")


        self.label3 = QtWidgets.QLabel(self.mySubwindow)
        self.label3.setGeometry(QtCore.QRect(120, 100, 100, 16))
        self.label3.setObjectName("label3")
        self.label3.setText("Learning Rate:")
        self.lineEdit2 = QtWidgets.QLineEdit(self.mySubwindow)
        self.lineEdit2.setGeometry(QtCore.QRect(210, 100, 50, 16))
        self.lineEdit2.setObjectName("lineEdit")
        self.lineEdit2.setText("0.01")

        self.label4 = QtWidgets.QLabel(self.mySubwindow)
        self.label4.setGeometry(QtCore.QRect(290, 100, 70, 16))
        self.label4.setObjectName("label4")
        self.label4.setText("Decay:")
        self.lineEdit3 = QtWidgets.QLineEdit(self.mySubwindow)
        self.lineEdit3.setGeometry(QtCore.QRect(340, 100, 50, 16))
        self.lineEdit3.setObjectName("lineEdit")
        self.lineEdit3.setText("0.04")

        self.buttom_ok = QtWidgets.QPushButton(self.mySubwindow)
        self.buttom_ok.setText("Ok")
        self.buttom_ok.setGeometry(QtCore.QRect(180, 150, 70, 50))
        self.buttom_ok.clicked.connect(self.b_cnn)

        self.mySubwindow.show()

    def rb1_clicked(self,pyCBIR):
        self.lineEdit1.show()
        self.lineEdit2.show()
        self.lineEdit3.show()

    def rb2_clicked(self,pyCBIR):
        self.lineEdit1.show()
        self.lineEdit2.show()
        self.lineEdit3.show()

    def rb3_clicked(self,pyCBIR):

        self.lineEdit1.hide()
        self.lineEdit2.hide()
        self.lineEdit3.hide()

    def b_cnn(self):

        if self.rb1.isChecked(): #treinar cnn
            #self.feature_extraction_method = self.comboBox.currentText()
            self.feature_extraction_method = 'training_' + self.comboBox.currentText()
            cwd = os.getcwd()
            lr =  self.lineEdit2.text()
            lr = lr.replace('.', '')
            file_name = self.comboBox.currentText()+'_epochs_'+self.lineEdit1.text()+'_learningRate_'+lr+'.h5'
            #file_name = 'model.ckpt'
            try:
                epocas = int(self.lineEdit1.text())
                if epocas <= 0:
                    QMessageBox.information(None,'pyCBIR', 'Invalid number of epochs!')
                    #self.buttom_ok.clicked.connect(self.radio5_clicked)
                else:
                #    if self.feature_extraction_method == 'lenet':
                    QMessageBox.information(None,'pyCBIR', 'Now you have to choose the place to save the trained model.')
                    self.path_save_cnn = QFileDialog.getSaveFileName(None,'Save File',file_name)[0]
                    self.list_of_parameters = [self.lineEdit2.text(),self.lineEdit1.text()]#learning rate and epochs
                    self.mySubwindow.close()
            except ValueError:
                QMessageBox.information(None,'pyCBIR', 'Invalid number of epochs!')
            #if self.lineEdit1.text() is '0':
            #    self.path_cnn_trained = QFileDialog.getOpenFileName(None,'Select the file of the pre-trained CNN: ', cwd,"Model Files (*.h5)")
            #else:
            #    self.path_cnn_trained = QFileDialog.getSaveFileName(None,'Save File',file_name,filter = 'h5 (*.h5)')[0]

        elif self.rb2.isChecked():#fine tuning
            self.feature_extraction_method = 'fine_tuning_'+self.comboBox.currentText()

            buttonReply = QMessageBox.question(None,'pyCBIR', 'Do you want to load a .h5 file?',QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                cwd = os.getcwd()
                self.path_cnn_pre_trained = QFileDialog.getOpenFileName(None,'Select a .h5 file!',cwd,filter = 'h5 (*.h5)')[0]
            else:
                self.path_cnn_pre_trained = ''

            QMessageBox.information(None,'pyCBIR', 'Now you have to choose the place to save the fine tuning model.')
            self.path_save_cnn = QFileDialog.getExistingDirectory(None,'Open path')

            self.path_save_cnn = self.path_save_cnn + '/model_fine_tuning.h5'
                        #if self.feature_extraction_method == 'fine_tuning_lenet':
            #    QMessageBox.information(None,'pyCBIR', 'Now you have to choose the pre-trained file.')
            #    cwd = os.getcwd()
            #    self.path_cnn_pre_trained = QFileDialog.getOpenFileName(None,'Select the file of the pre-trained CNN: ', cwd,"Model Files (*.ckpt)")
            self.list_of_parameters = [self.lineEdit2.text(),self.lineEdit1.text()]#learning rate and epochs
            self.mySubwindow.close()
        else: #pre-treinada
            self.feature_extraction_method = 'pretrained_'+self.comboBox.currentText()
            if self.feature_extraction_method == 'pretrained_lenet':
                QMessageBox.information(None,'pyCBIR', 'Now you have to choose the pre-trained file.')
                cwd = os.getcwd()
                self.path_cnn_pre_trained = QFileDialog.getOpenFileName(None,'Select the file of the pre-trained CNN: ', cwd,"Model Files (*.h5)")[0]
                #self.feature_extraction_method = 'pretrained_'+self.comboBox.currentText()

            else:
                buttonReply = QMessageBox.question(None,'pyCBIR', 'Do you want to use imageNet weights?',QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if buttonReply == QMessageBox.No:
                    cwd = os.getcwd()
                    QMessageBox.information(None,'pyCBIR', 'Now you have to choose the pre-trained file.')
                    self.path_cnn_pre_trained = QFileDialog.getOpenFileName(None,'Select a .h5 file!',cwd,filter = 'h5 (*.h5)')[0]
                else:
                    self.path_cnn_pre_trained = ''

            self.list_of_parameters = [self.lineEdit2.text(),self.lineEdit1.text()]#learning rate and epochs
            self.mySubwindow.close()

        #print(self.feature_extraction_method)




#----------------------Daisy Parameters---------------------#
    def radio6_clicked(self,pyCBIR):
        self.mySubwindow=subwindow()
        self.mySubwindow.createWindow(200,140)
        self.mySubwindow.setWindowFlags(QtCore.Qt.CustomizeWindowHint)

        self.label1_daisy = QtWidgets.QLabel(self.mySubwindow)
        self.label1_daisy.setGeometry(QtCore.QRect(55, 0, 100, 16))
        self.label1_daisy.setObjectName("label1")
        self.label1_daisy.setText("Daisy options")

        #GLCM parameters
        self.label2_daisy = QtWidgets.QLabel(self.mySubwindow)
        self.label2_daisy.setGeometry(QtCore.QRect(0, 30, 70, 16))
        self.label2_daisy.setObjectName("label2")
        self.label2_daisy.setText("Step:")
        self.lineEdit1_daisy = QtWidgets.QLineEdit(self.mySubwindow)
        self.lineEdit1_daisy.setGeometry(QtCore.QRect(45, 30, 30, 16))
        self.lineEdit1_daisy.setObjectName("lineEdit")
        self.lineEdit1_daisy.setText("4")


        self.label3_daisy = QtWidgets.QLabel(self.mySubwindow)
        self.label3_daisy.setGeometry(QtCore.QRect(0, 50, 110, 16))
        self.label3_daisy.setObjectName("label3")
        self.label3_daisy.setText("Rings:")
        self.lineEdit2_daisy = QtWidgets.QLineEdit(self.mySubwindow)
        self.lineEdit2_daisy.setGeometry(QtCore.QRect(45, 50, 30, 16))
        self.lineEdit2_daisy.setObjectName("lineEdit")
        self.lineEdit2_daisy.setText("3")

        self.label4_daisy = QtWidgets.QLabel(self.mySubwindow)
        self.label4_daisy.setGeometry(QtCore.QRect(0, 70, 110, 16))
        self.label4_daisy.setObjectName("label4")
        self.label4_daisy.setText("Histogram:")
        self.lineEdit3_daisy = QtWidgets.QLineEdit(self.mySubwindow)
        self.lineEdit3_daisy.setGeometry(QtCore.QRect(85, 70, 30, 16))
        self.lineEdit3_daisy.setObjectName("lineEdit")
        self.lineEdit3_daisy.setText("2")

        self.label5_daisy = QtWidgets.QLabel(self.mySubwindow)
        self.label5_daisy.setGeometry(QtCore.QRect(0, 90, 110, 16))
        self.label5_daisy.setObjectName("label4")
        self.label5_daisy.setText("Orientations:")
        self.lineEdit4_daisy = QtWidgets.QLineEdit(self.mySubwindow)
        self.lineEdit4_daisy.setGeometry(QtCore.QRect(85, 90, 30, 16))
        self.lineEdit4_daisy.setObjectName("lineEdit")
        self.lineEdit4_daisy.setText("8")

        self.buttom_daisy = QtWidgets.QPushButton(self.mySubwindow)
        self.buttom_daisy.setText("Ok")
        self.buttom_daisy.setGeometry(QtCore.QRect(50, 115, 100, 16))
        self.buttom_daisy.clicked.connect(self.b_daisy)

        self.mySubwindow.show()

    def b_daisy(self):
        self.list_of_parameters = [self.lineEdit1_daisy.text(),self.lineEdit2_daisy.text(),self.lineEdit3_daisy.text(),self.lineEdit4_daisy.text()]
        self.mySubwindow.close()

#-----------------Brute Force-----------------#
    """
    def radio7_clicked(self,pyCBIR):
        self.mySubwindow=subwindow()
        self.mySubwindow.createWindow(200,230)
        self.mySubwindow.setWindowFlags(QtCore.Qt.CustomizeWindowHint)

        self.label1_bf = QtWidgets.QLabel(self.mySubwindow)
        self.label1_bf.setGeometry(QtCore.QRect(55, 0, 120, 16))
        self.label1_bf.setObjectName("label1")
        self.label1_bf.setText("Similarity Metrics")

        self.gp = QtWidgets.QGroupBox(self.mySubwindow)
        self.radioButton_bf_1 = QtWidgets.QRadioButton(self.gp)
        self.radioButton_bf_1.setGeometry(QtCore.QRect(13, 29, 132, 18))
        self.radioButton_bf_1.setChecked(True)
        self.radioButton_bf_2 = QtWidgets.QRadioButton(self.gp)
        self.radioButton_bf_2.setGeometry(QtCore.QRect(13, 46, 132, 18))
        self.radioButton_bf_3 = QtWidgets.QRadioButton(self.gp)
        self.radioButton_bf_3.setGeometry(QtCore.QRect(13, 63, 129, 18))
        self.radioButton_bf_4 = QtWidgets.QRadioButton(self.gp)
        self.radioButton_bf_4.setGeometry(QtCore.QRect(13, 80, 141, 18))
        self.radioButton_bf_5 = QtWidgets.QRadioButton(self.gp)
        self.radioButton_bf_5.setGeometry(QtCore.QRect(13, 97, 164, 18))
        self.radioButton_bf_6 = QtWidgets.QRadioButton(self.gp)
        self.radioButton_bf_6.setGeometry(QtCore.QRect(13, 114, 187, 18))
        self.radioButton_bf_7 = QtWidgets.QRadioButton(self.gp)
        self.radioButton_bf_7.setGeometry(QtCore.QRect(13, 131, 150, 18))
        self.radioButton_bf_8 = QtWidgets.QRadioButton(self.gp)
        self.radioButton_bf_8.setGeometry(QtCore.QRect(13, 148, 213, 18))
        self.radioButton_bf_9 = QtWidgets.QRadioButton(self.gp)
        self.radioButton_bf_9.setGeometry(QtCore.QRect(13, 165, 138, 18))
        self.radioButton_bf_10 = QtWidgets.QRadioButton(self.gp)
        self.radioButton_bf_10.setGeometry(QtCore.QRect(13, 182, 213, 17))

        self.radioButton_bf_1.setText("Euclidean Distance")
        self.radioButton_bf_2.setText("Infinity Distance")
        self.radioButton_bf_3.setText("Cossine Similarity")
        self.radioButton_bf_4.setText("Pearson Correlation")
        self.radioButton_bf_5.setText("Chi-Square")
        self.radioButton_bf_6.setText("Kullback Divergence")
        self.radioButton_bf_7.setText("Jeffrey Divergence")
        self.radioButton_bf_8.setText("Kolmogorov Divergence")
        self.radioButton_bf_9.setText("Cramer Divergence")
        self.radioButton_bf_10.setText("Earth Movers Distance")


        self.buttom_bf = QtWidgets.QPushButton(self.mySubwindow)
        self.buttom_bf.setText("Ok")
        self.buttom_bf.setGeometry(QtCore.QRect(50, 210, 100, 16))
        self.buttom_bf.clicked.connect(self.b_bf)

        self.mySubwindow.show()

    def b_bf(self):

        if self.radioButton_bf_1.isChecked():
            self.similarity_metric = 'ed'
        elif self.radioButton_bf_2.isChecked():
            self.similarity_metric = 'id'
        elif self.radioButton_bf_3.isChecked():
            self.similarity_metric = 'cs'
        elif self.radioButton_bf_4.isChecked():
            self.similarity_metric = 'pcc'
        elif self.radioButton_bf_5.isChecked():
            self.similarity_metric = 'csd'
        elif self.radioButton_bf_6.isChecked():
            self.similarity_metric = 'kld'
        elif self.radioButton_bf_7.isChecked():
            self.similarity_metric = 'ld'
        elif self.radioButton_bf_8.isChecked():
            self.similarity_metric = 'ksd'
        elif self.radioButton_bf_9.isChecked():
            self.similarity_metric = 'cmd'
        elif self.radioButton_bf_10.isChecked():
            self.similarity_metric = 'cd'

        self.mySubwindow.close()
    """
#-------------------Information---------------#
    def returnInformation(self,pyCBIR):
        #import sys
        #sys.path.insert(0, '../src')
        import run

        if self.radioButton.isChecked():
            self.feature_extraction_method = 'glcm'
        if self.radioButton_2.isChecked():
            self.feature_extraction_method = 'hog'
        if self.radioButton_3.isChecked():
            self.feature_extraction_method = 'fotf'
        if self.radioButton_4.isChecked():
            self.feature_extraction_method = 'lbp'
        if self.radioButton_6.isChecked():
            self.feature_extraction_method = 'daisy'

        searching_method = ''

        if self.radioButton_7.isChecked():
            searching_method = 'bf'
            self.similarity_metric = 'ed'
        elif self.radioButton_8.isChecked():
            searching_method = 'kd'
        elif self.radioButton_9.isChecked():
            searching_method = 'bt'

        retrieval_number = int(self.lineEdit.text())

        preprocessing_method = 'simple'


        if (self.lineEdit_3.text() != ""):
            fname_database,labels_database = informationPath(self.lineEdit_3.text())
            fname_retrieval,labels_retrieval = informationPath(self.lineEdit_4.text())
        #elif (self.lineEdit_5.text() != ""):
        #    fname_database,labels_database = informationFile(self.lineEdit_5.text())
        #    fname_retrieval,labels_retrieval = informationFile(self.lineEdit_6.text())

        print(self.path_cnn_trained)
        _,_,file = run.run_command_line(fname_database,labels_database,fname_retrieval,labels_retrieval,self.path_cnn_pre_trained, self.path_save_cnn, self.lineEdit_2.text(),self.feature_extraction_method,self.similarity_metric,retrieval_number,self.list_of_parameters,preprocessing_method,searching_method, isEvaluation = False)
        self.w = QtWidgets.QWidget(self.centralwidget)
        self.w.setGeometry(QtCore.QRect(140, 0, 951, 791))
        self.w.setEnabled(True)
        self.w.setAcceptDrops(True)
        self.w.setObjectName("image")
        label = QLabel(self.w)
        pixmap = QPixmap(file)
        if (pixmap.height() < 680):
            pixmap = pixmap.scaled(800,pixmap.height())
        else:
            pixmap = pixmap.scaled(800,680)
        label.setPixmap(pixmap)
        self.w.show()


    def retranslateUi(self, pyCBIR):
        _translate = QtCore.QCoreApplication.translate
        pyCBIR.setWindowTitle(_translate("pyCBIR", "pyCBIR"))
        self.groupBox.setTitle(_translate("pyCBIR", "Signatures"))
        self.radioButton.setText(_translate("pyCBIR", "GLCM"))
        self.radioButton_2.setText(_translate("pyCBIR", "HOG"))
        self.radioButton_3.setText(_translate("pyCBIR", "Histogram"))
        self.radioButton_4.setText(_translate("pyCBIR", "LBP"))
        self.radioButton_5.setText(_translate("pyCBIR", "CNN"))
        self.radioButton_6.setText(_translate("pyCBIR", "Daisy"))
        self.groupBox_2.setTitle(_translate("pyCBIR", "Searching"))
        self.radioButton_7.setText(_translate("pyCBIR", "Brute Force"))
        self.radioButton_8.setText(_translate("pyCBIR", "KDTree"))
        self.radioButton_9.setText(_translate("pyCBIR", "BallTree"))
        self.groupBox_3.setTitle(_translate("pyCBIR", "Retrieval Options"))
        self.label.setText(_translate("pyCBIR", "Retrieval Number:"))
        self.groupBox_4.setTitle(_translate("pyCBIR", "By Folder"))
        self.pushButton_3.setText(_translate("pyCBIR", "Database Folder"))
        self.pushButton_4.setText(_translate("pyCBIR", "Query Folder"))

        #CSV
        #self.groupBox_5.setTitle(_translate("pyCBIR", "By CSV"))
        #self.pushButton_5.setText(_translate("pyCBIR", "Database File"))
        #self.pushButton_6.setText(_translate("pyCBIR", "Query File"))
        self.pushButton.setText(_translate("pyCBIR", "pyCBIR"))
        self.pushButton_2.setText(_translate("pyCBIR", "Output Folder"))


class subwindow(QtWidgets.QWidget):
    def createWindow(self,WindowWidth,WindowHeight):
       parent=None
       super(subwindow,self).__init__(parent)
       self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
       self.resize(WindowWidth,WindowHeight)


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
