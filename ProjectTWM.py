
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# from model import Model
import os
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from decimal import Decimal
from tkinter import *
from PIL import Image, ImageTk

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


qtcreator_file = "Project.ui"
ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)



class MyWindow(QtWidgets.QMainWindow, ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        ui_MainWindow.__init__(self)
        # self.model = Model()
        self.setupUi(self)
        self.btnResult.clicked.connect(self.GetResult)
        self.label_2.setStyleSheet("background-image : url(workplace-2303849_1280.jpg);")

        # setting label text
    def GetResult(self):
        document = self.txtInput.toPlainText()

        modelscorev2 = joblib.load('model.pkl', mmap_mode='r')
        count_vect = CountVectorizer()
        tfidf_transformer = TfidfTransformer()
        document = [str(document)]
        count_vect.fit_transform(document)
        X_new_counts = count_vect.fit_transform(document)
        # We call transform instead of fit_transform because it's already been fit
        X_new_tfidf = tfidf_transformer.fit_transform(X_new_counts)
        result = modelscorev2.predict(document)
        listKQ = ["Đây là bình luận tiêu cực", "Đây là bình luận trung lập", "Đây là bình luận tích cực"]
        if result[0] == -1:
            self.refreshAll()
            self.lbResult.setText(str(listKQ[0]))
        elif result[0] == 0:
            self.refreshAll()
            self.lbResult.setText(str(listKQ[1]))
        elif result[0] == 1:
            self.refreshAll()
            self.lbResult.setText(str(listKQ[2]))


    def refreshAll(self):
        self.lbResult.setText(" ")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
