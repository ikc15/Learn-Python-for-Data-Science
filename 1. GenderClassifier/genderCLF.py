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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= .5) 

# define classifiers using dictionary
classifier = {'Tree':[tree.DecisionTreeClassifier()], 'KNN':[KNeighborsClassifier()], 
              'NN':[MLPClassifier()]}

# train classifiers on train data
for clf in classifier:
    classifier[clf][0].fit(X_train,Y_train)

# Predict new data using the trained classifiers
def predict(clf, data):
    return classifier[clf][0].predict(data)

# Compare the 3 SciKit-learn Models     
for clf in classifier:
    pred = predict(clf, X_test)
    match = np.where(np.array(pred)==np.array(Y_test))
    accuracy = len(match[0])/len(X_test)
    classifier[clf].append(accuracy) # append accuracy to dictionary 
    print ('%s: accuracy = %f'%(clf,accuracy))

# find best model
score = 0 
key = None
for clf in classifier:
    if classifier[clf][1] > score: 
        score, key= classifier[clf][1], clf    
print ('Best model is %s'%(key))
