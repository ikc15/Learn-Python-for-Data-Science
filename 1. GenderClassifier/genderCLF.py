# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:52:10 2018

@author: ikc15

Gender classifier using sklearn classifiers
"""
import numpy as np
from sklearn import tree #tree classifier
from sklearn.neighbors import KNeighborsClassifier # K-nearest neighbours classifier
from sklearn.neural_network import MLPClassifier # neural net classifier
from sklearn.cross_validation import train_test_split

# Data [height, weight, shoe size]
X = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],
     [166,65,40],[190,90,47],[175,64,39],[177,70,40],
     [171,75,42],[181,85,43]]

# Labels 
Y = ['male', 'female', 'female', 'female', 'male', 'male',
     'male', 'female', 'male', 'female']

# Split data into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= .2) 

# define classifiers using dictionary
classifier = {'Tree':tree.DecisionTreeClassifier(), 'KNN':KNeighborsClassifier(), 
              'NN':MLPClassifier()}

# train classifiers on train data
for clf in classifier:
    classifier[clf].fit(X_train,Y_train)

# Test the classifiers using test data
def predict(data, label):
    for clf in classifier:
        counter = 0
        pred = classifier[clf].predict(data)
        for i in range(len(label)):
            if pred[i] == label[i]:
                counter+=1
        print ('%s: accuracy = %f'%(clf,counter/len(data)))

predict(X_test, Y_test)

