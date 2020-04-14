#!/usr/bin/env python
# coding: utf-8

# In[26]:


import cv2
import numpy as np
import mahotas
import os
from sklearn.svm import SVC
from sklearn import svm
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler


def fd_hu_moments(image):
    
    """ 
    Original algorithm from DataTurks: https://medium.com/@dataturks/understanding-svms-for-image-classification-cf4f01232700 
    Adapted for the purposes of this Lab
    fd_hu_moments: image -> flattened vectors with hu moments
    Purpose: Receives an image and returns a flattened vector with the image's Hu Moments, which is a
    feature that can be used for SVM Classification
    Example: def(cat.01.jpg) -> vector with hu moments of cat.01.jpg
    """
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    
    """ 
    Original algorithm from DataTurks: https://medium.com/@dataturks/understanding-svms-for-image-classification-cf4f01232700 
    Adapted for the purposes of this Lab
    fd_haralick: image -> flattened vector with a haralick feature vector
    Purpose: Receives an image and returns a flattened vector with the image's haralick texture feature, which is a
    feature that can be used for SVM Classification
    Example: def(cat.01.jpg) -> vector with haralick texture feature of cat.01.jpg
    """
        
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
 
def fd_histogram(image):
    
    """ 
    Original algorithm from DataTurks: https://medium.com/@dataturks/understanding-svms-for-image-classification-cf4f01232700 
    Adapted for the purposes of this Lab
    fd_histogram: image -> flattened vector with a haralick feature vector
    Purpose: Receives an image and returns a flattened vector with the image's haralick texture feature, which is a
    feature that can be used for SVM Classification
    Example: def(cat.01.jpg) -> vector with haralick texture feature of cat.01.jpg
    """
        
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [5, 5, 5], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    hist = hist.flatten()
    return hist
    
def myFeatureExtract(fileList, numImages):
    
    """ myFeatureExtract: array + int -> training and testing sets of features for SVM classification
    Purpose: Receive an array with the images to be used and their class labels to return training and testing arrays with features and class
    Example: def(fileList, 1000) -> returns testing set and training set for SVM classification
    """
        
    trainFeat = []
    testFeat = []
    
    x = np.arange(0, len(fileList[:,0]), 1, dtype=int)
    nTrain = int(numImages*.80)
    subset = np.random.choice(x, numImages)
    training = subset[:int(nTrain)]
    test = subset[int(nTrain):]

#we extract features of each image file and add them to a huge array that we will use for classification
    
    for i in range(len(training)):
            klass = fileList[int(training[i]),1]
            image = cv2.imread(fileList[int(training[i]),0])
            histo = fd_histogram(image)
            haralick = fd_haralick(image)
            huMoms = fd_hu_moments(image)
            trainFeat.append(np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image), int(klass)]))  
    
    for i in range(len(test)):
            klass=fileList[int(test[i]),1]
            image = cv2.imread(fileList[int(test[i]),0])
            histo = fd_histogram(image)
            haralick = fd_haralick(image)
            huMoms = fd_hu_moments(image)
            testFeat.append(np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image), int(klass)]))    
    
    return trainFeat, testFeat 




def myKittyClassifier(numImages):

    pth = 'D:\\Dataset'


    fileList = []
    xTrain = []
    xTest = []
    yTrain = []
    yTest = []

    #Listing files from the Dataset, and storing their paths for automation of feature extraction
    #Done by adapting algorithm from https://note.nkmk.me/en/python-os-getcwd-chdir/
    # r=root, d=directories, f = files
    for r, d, f in os.walk(pth):
        for file in f:
            if ('.jpg' in file) & ('cat' in file):
                fileList.append([os.path.join(r, file),0])
            elif ('.jpg' in file) & ('dog' in file):
                fileList.append([os.path.join(r, file),1])

    fileList = np.array(fileList)


    # We specify the # of Images we want to analyze in our Dataset. It will
    # be distributed in a 80% train 20% test distribution. "0" for cats and "1" for dogs
    xTrain, xTest = myFeatureExtract(fileList, numImages)
    xTest = np.array(xTest)
    xTrain = np.array(xTrain)

    #we re-scale
    scaler = MinMaxScaler(feature_range=(0, 1))

    yTrain = xTrain[:,145]
    yTest = xTest[:,145]
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.fit_transform(xTest)


    mySVC = sk.svm.SVC(C = 1000, kernel = 'linear') 
    mySVC.fit(xTrain[:,0:144], yTrain)

    #Predicting with our model 
    yPred = mySVC.predict(xTest[:,0:144])

    #Runnning Reports
    myAccu = sk.metrics.accuracy_score(yTest, yPred, normalize = False)
    myReport  = classification_report(yTest, yPred, target_names = ['Cats', 'Dogs'])
    myConfMat = confusion_matrix(yTest,yPred)

    print("This is the Accuracy \n", myAccu)
    print("This is the Classification Report \n", myReport)
    print("This is the Confusion matrix \n", myConfMat)

myKittyClassifier(5000)


# In[ ]:




