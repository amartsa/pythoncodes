from matplotlib import pyplot as plt
from PIL import Image
"""
Purpose: Opens image provided by instructore and creates a thumbnail of the exact same size of the original image
Example: Not available as solution is specific to instructions by lecturer
"""

#Opens the image to be used and stores it into the myImage variable. 
#Also creates a plot called myPlot that we will use to store the images
#Extracts also the size of the opened image for later use of this information
myPlot = plt.figure()
myImage = Image.open("img_0.png")
myWidth, myHeight = myImage.size

#Adds img_0.png to the plot as the first element of the plot
myPlot.add_subplot(1, 2, 1)
plt.imshow(myImage)

#Adds the 'resized' thumbnail of 'myImage' to the plot as the second element of the plot. 
#This is achieved by making the thumbnail have the EXACT same size as the original image
myPlot.add_subplot(1, 2, 2)
myImage.thumbnail((myWidth, myHeight))
plt.imshow(myImage)
plt.show()






