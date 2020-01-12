# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:35 2020

@author: my, myself and I
"""

# -*- coding: utf-8 -*-

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO
from IPython.display import Image
import pydotplus

"""
Data Engineering and Analysis
"""
#Load the dataset

AH_data = pd.read_csv("breast-cancer.csv")
data_clean = AH_data.dropna()
data_clean.describe()
"""
Modeling and Prediction
"""
#Split into training and testing sets

predictors = data_clean[['age','menopause','tumor-size','inv-nodes','node-caps','deg-malig',
'breast','breast-quad','irradiat']]

targets = data_clean.recurrenceEvents

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.4)

#Build model on training data
classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

out = StringIO()
tree.export_graphviz(classifier, out_file=out)

graph=pydotplus.graph_from_dot_data(out.getvalue())

Image(graph.create_png())

graph.write_pdf("result.pdf")




