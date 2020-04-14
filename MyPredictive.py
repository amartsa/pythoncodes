# -*- coding: utf-8 -*-
from sklearn.svm import SVC
from sklearn import svm
import sklearn as sk
import pandas as pd
import datetime#imported like this in order to be able to use datetime.dateime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  


#def myPredictive():

#Formatting the dates and adding together both datasets
myStdData = pd.read_csv("C:\\Users\\Ariel Martinez Salas\\Google Drive\\UNB MSC of Data Science\\1st Trimester\\GGE 6505 Intro to Data Science\\Assignment Instructions\\Assignment 2\\430pm_Standard2.csv", header=None, names=['TimeStamp', 'ChPointID', 'ChType','Status'], sep= ' ')#Reads the StandardData.csv file and incorporates corresponding variable names on the columns
myMgdData = pd.read_csv("C:\\Users\\Ariel Martinez Salas\\Google Drive\\UNB MSC of Data Science\\1st Trimester\\GGE 6505 Intro to Data Science\\Assignment Instructions\\Assignment 2\\430pm_Managed2.csv", header=None, names=['TimeStamp', 'ChPointID', 'ChType','Status'], sep= ' ')#Reads the StandardData.csv file and incorporates corresponding variable names on the columns
myWhlData = pd.concat([myStdData,myMgdData], ignore_index=True)#Joins the two datasets for analysis analysis together
for myDtPt in range(0, myWhlData.shape[0]):#starts the loop that will change each date into the format que want
    myWhlData.iloc[myDtPt,0] = datetime.datetime.strptime(str(myWhlData.iloc[myDtPt,0]),'%Y%m%d%H%M').strftime("%H:%M")#Getting the time stamp in a format we can use. I used time hour and minute instead of day because all the measures are from 1 day only, and the only discriminating timestamp we could use was this one.
#myWhlData.to_csv("C:\\Users\\Ariel Martinez Salas\\Google Drive\\UNB MSC of Data Science\\1st Trimester\\GGE 6505 Intro to Data Science\\Assignment Instructions\\Assignment 2\\myWhlData.csv", index=False, header=True) # Just used to make sure the date and data were correctly formatted and added together

#Creating the Unique Array
unique = {}
for myLsCt in myWhlData:
    unique[myLsCt] = (set(myWhlData[myLsCt]))
for myLsCt in myWhlData:
    myWhlData[myLsCt].replace(unique[myLsCt], range(len(unique[myLsCt])), inplace = True)

myX = myWhlData.drop('Status', axis=1)
myY = myWhlData['Status']

#Creating the training and test datasets
Ctrain, Ctest, Ytrain, Ytest = train_test_split(myX, myY, test_size=0.4)#Ctrain and Ctest variables,as well as their labels (status) (syntax on where each value goes is standard)

#Testing the datasets
#mySVC = SVC(kernel='linear')  
mySVC = sk.svm.SVC() 
mySVC.fit(Ctrain, Ytrain)

#Predicting with our model 
YPred = mySVC.predict(Ctest)

#Runnning Reports
myAccu = sk.metrics.accuracy_score(Ytest, YPred)
myReport  = classification_report(Ytest, YPred, target_names = ['Free', 'Occupied'])
myConfMat = confusion_matrix(Ytest,YPred)

#Printing Reports
print(myAccu)
print(myReport)
print(myConfMat)
#if __name__ == "__myPredictive__":
    #myPredictive()