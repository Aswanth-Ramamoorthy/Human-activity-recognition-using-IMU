#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 21:58:26 2020

@author: aswanthramamoorthy
"""


from sklearn.svm import SVC
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
import numpy as np


# This is to load a given dataset from a given filepath and returns a NumPy array
def loadData(filename):
    df = np.loadtxt(filename)
    return df
  
# load each group of dataset i.e. train and test
def load_dataset_group(group, prefix=''):
	# load input data
	X = loadData(prefix + group + '/X_'+group+'.txt')
	# load class output
	y = loadData(prefix + group + '/y_'+group+'.txt')
	return X, y
 
# This loads the dataset, returns X,y for train and test elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'UCI HAR Dataset/')
	
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'UCI HAR Dataset/')
	
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

trainX, trainy, testX, testy = load_dataset()

#This to give the count of each activity and entered into NumPy array
def total_count(y):
    tot_count = np.zeros(6)
    for i in range(y.size):
        if (y[i] == 1):
            tot_count[0] += 1
        if (y[i] == 2):
            tot_count[1] += 1
        if (y[i] == 3):
            tot_count[2] += 1
        if (y[i] == 4):
            tot_count[3] += 1
        if (y[i] == 5):
            tot_count[4] += 1
        if (y[i] == 6):
            tot_count[5] += 1
    return tot_count

y_test_tot_count = total_count(testy)

#Confusion matrix to determine how correct our covariance
def confusion_matrix(y_test, y_pred):
    c_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            for k in range(y_pred.size):
                if (y_pred[k] == j + 1 and y_test[k] == i + 1):
                    c_matrix[i][j] += 1
    return c_matrix


# This determines the precision percentage for each activity
def pre_rec(c_matrix, count):
    precision = []
    for i in range(6):
        precision.append(float("{0:.2f}".format(c_matrix[i][i] / count[i])))
    return precision
 
#This determine overall accuracy       
def overall_accuracy(cm, y_test):
    sum = 0
    for i in range(6):
        sum += cm[i][i]
    return float("{0:.2f}".format(sum * 100.0 / y_test.size))    
        
#----------------------------------------------------------------------------------

#Plotting the  distribution of the test cases using principal component analysis        
pca2 = PCA(n_components = 2)
pca2.fit(trainX)
X_train_2 = pca2.transform(trainX)
X_test_2 = pca2.transform(testX)

x_11 = []
x_12 = []
x_21 = []
x_22 = []
x_31 = []
x_32 = []
x_41 = []
x_42 = []
x_51 = []
x_52 = []
x_61 = []
x_62 = []

for i in range(len(testy)):
    if (testy[i] == 1):
        x_11.append(X_test_2[i][0])
        x_12.append(X_test_2[i][1])
    elif (testy[i] == 2):
        x_21.append(X_test_2[i][0])
        x_22.append(X_test_2[i][1])
    elif (testy[i] == 3):
        x_31.append(X_test_2[i][0])
        x_32.append(X_test_2[i][1])
    elif (testy[i] == 4):
        x_41.append(X_test_2[i][0])
        x_42.append(X_test_2[i][1])
    elif (testy[i] == 5):
        x_51.append(X_test_2[i][0])
        x_52.append(X_test_2[i][1])
    else:
        x_61.append(X_test_2[i][0])
        x_62.append(X_test_2[i][1])


plt.figure() 
plt.plot(x_11, x_12, 'xc', label = 'Walking')
plt.plot(x_21, x_22, 'xb', label = 'Upstairs')
plt.plot(x_31, x_32, 'xy', label = 'Downstairs')
plt.plot(x_41, x_42, 'xr', label = 'Sitting')
plt.plot(x_51, x_52, 'xm', label = 'Standing')     
plt.plot(x_61, x_62, 'xg', label = 'Laying')
plt.legend(loc='lower left')
plt.show()


#-------------------------------------------------------------------------------
#SVM algorithm        
model = SVC(probability = True, gamma ='auto')
model.fit(trainX, trainy)
yhat = model.predict(testX)

#cunts the number of activites gotten correctly
y_pred_count = total_count(yhat)

#confusion matrix
cmatrix_svm = confusion_matrix(testy, yhat)

print ("\nSVM with kernel = 'rbf':")
print (cmatrix_svm)
print ("")

#recall calculated:
recall = pre_rec(cmatrix_svm, y_test_tot_count)

#precision calculated
precision= pre_rec(cmatrix_svm, y_pred_count)

#acuracy calculated
accuracy = overall_accuracy(cmatrix_svm, testy)


print ("Precision for SVM: ")
print (precision)

print ("Recall for SVM: ")
print (recall)

print ("Accuracy for SVM: ")
print (accuracy)

