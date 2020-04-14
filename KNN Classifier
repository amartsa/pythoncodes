"""
Provides digit classification from the digits Dataset using KNN.

"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,confusion_matrix)
import cv2
import sklearn as sk

# load MNIST images
mnist = datasets.load_digits()
images = mnist.images

# pre-processing images
# convert 2D image array to 1D
data_size = len(images)
images = images.reshape(data_size, -1)
labels = mnist.target

# we split the dataset into trainng and testing
xTrain, xTest, yTrain, yTest = train_test_split(images, labels, test_size = 0.20, random_state = 100)

# training data and test data
model = KNeighborsClassifier(n_neighbors=1)
model.fit(xTrain, yTrain)   

# testing the data
yPred = model.predict(xTest)                               
myAccu = sk.metrics.accuracy_score(yTest, yPred, normalize = False)
myReport  = classification_report(yTest, yPred, target_names = ['0','1','2','3','4','5','6','7','8','9'])
myConfMat = confusion_matrix(yTest,yPred, labels = [0,1,2,3,4,5,6,7,8,9])
                             
print("This is the Accuracy \n", myAccu)
print("This is the Classification Report \n", myReport)
print("This is the Confusion matrix \n", myConfMat)


"""We get can see the precision of our KNN algorithm at 1 nearest neighbor is almost perfect.
The confusion matrix confirms this assessment. We are classifying most digits properly 100% of the time
with only a few mistakes in digits 1 and 3 of the dataset. KNN seems to be a good image classifier -  and the original data is impeccable too

"""

