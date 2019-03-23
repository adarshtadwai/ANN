# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 01:29:37 2019

@author: Adarsh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data Preprocessing Part
DataSet = pd.read_csv('Churn_Modelling.csv')
X = DataSet.iloc[: , 3:13].values
Y = DataSet.iloc[: , 13].values

#ENCODING CATEGORICAL DATA
#1.Encoding Independent variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_encoder_X1 = LabelEncoder()
X[:,1] = label_encoder_X1.fit_transform(X[:,1])
label_encoder_X2 = LabelEncoder()
X[:,2] = label_encoder_X2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#Removal of Dummy Variable[Without getting fallen into the trap]
X = X[: , 1:]
#2.Dependent Variables are already Encoded Properly

#Splitting the data into train set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Using ANN for prediction
import keras
from keras.models import Sequential #Initialize ANN
from keras.layers import Dense #Create Layers in ANN
classifier = Sequential() # Initializing an object having sequence of Layers
#Adding The Input layer and first hidden layer
classifier.add(Dense(input_dim=11,output_dim=6,activation='relu',init='uniform'))
#Adding The Second Input Layer
classifier.add(Dense(output_dim=6,activation='relu',init='uniform'))
#Adding the Output Layer
classifier.add(Dense(output_dim=1,activation='sigmoid',init='uniform'))

#Making the predictions and evaluating the model

#APPLYING STOCHASTIC GRADIENT DESCENT TO ANN, i.e., COMPILING THE ANN
#adam is one of the most efficient form of gradient descent algorithm
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrices=['accuracy'])

#  Fitting ANN to the training set
classifier.fit(X_train,Y_train,epochs=10, batch_size=32)

#Predicting the test_set results

y_pred = classifier.predict(X_test)






  

