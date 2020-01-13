# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:26:46 2015

@author: jrose01
"""

#from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLarsCV
from sklearn import preprocessing
 
#Load the dataset
data = pd.read_csv("../../breast-cancer.csv")

#upper-case all DataFrame column names
data.columns = map(str.lower, data.columns)
print(data.columns)
# Data Management
data_clean = data.dropna()

#select predictor variables and target variable as separate data sets  
predvar= data_clean[['age','menopause','tumor-size','inv-nodes','node-caps','deg-malig',
'breast','breast-quad','irradiat']]

target = data_clean.recurrenceevents
 
# standardize predictors to have mean=0 and sd=1
predictors=predvar.copy()
'''
predictors['age']=preprocessing.scale(predictors['age'].astype('float32'))
predictors['menopause']=preprocessing.scale(predictors['menopause'].astype('float32'))
predictors['tumor-size']=preprocessing.scale(predictors['tumor-size'].astype('float32'))
predictors['inv-nodes']=preprocessing.scale(predictors['inv-nodes'].astype('float32'))
predictors['node-caps']=preprocessing.scale(predictors['node-caps'].astype('float32'))
predictors['deg-malig']=preprocessing.scale(predictors['deg-malig'].astype('float32'))
predictors['breast']=preprocessing.scale(predictors['breast'].astype('float32'))
predictors['breast-quad']=preprocessing.scale(predictors['breast-quad'].astype('float32'))
predictors['irradiat']=preprocessing.scale(predictors['irradiat'].astype('float32'))'''
# split data into train and test sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, 
                                                              test_size=.3, random_state=123)

# specify the lasso regression model
model=LassoLarsCV(cv=5, precompute=False).fit(pred_train,tar_train)

# print variable names and regression coefficients
print(dict(zip(predictors.columns, model.coef_)))

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.mse_path_, ':')
plt.plot(m_log_alphascv, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
plt.show()

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)