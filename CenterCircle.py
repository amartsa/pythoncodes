"""
Finds the center of mass of different objects in an image and plots a red dot in them
"""

from matplotlib import pyplot as plt
from PIL import Image as img
import numpy as np
from skimage import io
from scipy import ndimage as ndi

#We read the image and turn it into a binary array (0 for black, 1 for white)
myImage = img.open("circles.png")
myImage = myImage.convert(mode="1")
myImage = np.array(myImage)

#Using 'label' we create a tuple that has: 1) location of different features (circles)
#and 2) the "name" attached to them. We store this in 'centroid'
lbl = ndi.label(myImage)[0]
labeled_array, num_features = ndi.label(myImage)
centroid = ndi.measurements.center_of_mass(myImage, lbl, np.unique(labeled_array))

#We extract the x,y coordinates from "centroid" using the "zip" function
#We plot them on top of the original image
im = plt.imread("circles.png")
implot = plt.imshow(im)
y, x = zip(*centroid)
plt.scatter(x, y, c='r')
plt.savefig('Center.png')
plt.show()
