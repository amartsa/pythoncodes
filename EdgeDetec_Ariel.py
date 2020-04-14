#!/usr/bin/env python
# coding: utf-8

# In[3]:


from matplotlib import pyplot as plt
from PIL import Image as img
import numpy as np
from skimage import io
from scipy import ndimage

def myCannyEdge(image, desired):

    """ myCannyEdge: image * 'gaussian' -> Blurred Image
        myCannyEdge: image * 'sobel' -> Applies Sobel Filter
        myCannyEdge: image * 'full' -> Applies Sobel Filter
    Purpose: Receive an image and the desired filter type to create the outputs needed for lab 2.
    Example: def(York House_Fredericton.jpg, gaussian) -> Image blurred by a 5x5 Gaussian kernel applied"
    """
    if desired == 'gaussian':    
        myImage = img.open("York House_Fredericton.jpg")
        myWidth, myHeight = myImage.size
        myImage = myImage.convert(mode="L")
        myWidth, myHeight = myImage.size
        myImage = np.array(myImage, dtype = np.float32)

        #We declare the size of the new array where we'll store post-Gaussian pixels
        newWt = myWidth - 4
        newHt = myHeight - 4
        newArray = np.zeros((newHt, newWt), dtype = int)

        #We create the 5x5 Gaussian mask with sigma = 1
        sigma = 1
        x, y = np.mgrid[-2:3, -2:3]
        normal = 1 / (2.0 * np.pi * sigma**2)
        gaussian =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

        #We apply the mask to the image
        for i in  list(range (newHt)):
            for j in list(range (newWt)):
                tempArray = myImage[i:i+5,j:j+5]
                tempArray = np.multiply(tempArray,gaussian*7500)
                mySum = np.sum(tempArray)
                newPixel = mySum/25
                newArray[i,j] = newPixel

        #We rebuild the image from the array
        io.imsave("EdgeDetec_Smooth.jpg",newArray*25000)

    if desired == 'sobel':
        myCannyEdge(image, 'gaussian')
        myImage = img.open("EdgeDetec_Smooth.jpg")
        myImage = np.array(myImage, dtype = np.float32)
        Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        Ix = ndimage.filters.convolve(myImage, Sx)
        Iy = ndimage.filters.convolve(myImage, Sy)
        sobelled = np.hypot(Ix, Iy)
        sobelled = sobelled / sobelled.max() * 255
        sobelled = sobelled.astype(int)
        io.imsave("EdgeDetec_Sobel.jpg", sobelled)
        
    if desired == 'full':
        myCannyEdge(image, 'sobel')
        myImage = img.open("EdgeDetec_Smooth.jpg")
        myImage = np.array(myImage, dtype = np.float32)
        Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        Ix = ndimage.filters.convolve(myImage, Sx)
        Iy = ndimage.filters.convolve(myImage, Sy)
        sobelled = np.hypot(Ix, Iy)
        myMag = sobelled / sobelled.max() * 255
        gradDir = np.arctan2(Iy, Ix)
        gradDir = np.rad2deg(gradDir)

        #We apply Non-Max suppression, store in "Z"
        H, W = myMag.shape
        Z = np.zeros((H,W), dtype=np.int32)
        gradDir += 180

        for i in range(1,H-1):
            for j in range(1,W-1):
                try:
                    q = 255
                    r = 255

                    if (0 <= gradDir[i,j] < 22.5) or (157.5 <= gradDir[i,j] <= 180):
                        q = myMag[i, j+1]
                        r = myMag[i, j-1]

                    elif (22.5 <= gradDir[i,j] < 67.5):
                        q = myMag[i+1, j-1]
                        r = myMag[i-1, j+1]

                    elif (67.5 <= gradDir[i,j] < 112.5):
                        q = myMag[i+1, j]
                        r = myMag[i-1, j]

                    elif (112.5 <= gradDir[i,j] < 157.5):
                        q = myMag[i-1, j-1]
                        r = myMag[i+1, j+1]

                    if (myMag[i,j] >= q) and (myMag[i,j] >= r):
                        Z[i,j] = myMag[i,j]
                    else:
                        Z[i,j] = 0
                except: 
                    print("Error")
        nMaxSup = Z
        #io.imsave("EdgeDetec_nMaxSupress.jpg", Z)
        #io.imshow(rebuilt)
        #io.imshow(Z)
        
        
        #We apply Thresholding, save in Z
        
        Z = np.zeros((H,W), dtype=np.int32)
        strong = np.int32(255)
        weak = np.int32(50)
    
        strong_i, strong_j = np.where(nMaxSup >= strong)
        weak_i, weak_j = np.where((nMaxSup <= strong) & (nMaxSup >= weak))
    
        Z[strong_i, strong_j] = strong
        Z[weak_i, weak_j] = weak
                     
        hyster = Z
        io.imsave("EdgeDetec_Final.jpg", Z)
        
        Z = np.zeros((H,W), dtype=np.int32)
        
        '''
        
        Below is the code for the Hystheresis filter. However, the image is just too dark to apply it, this is why I have refrained from using the code in the final product
        #We apply Hysteresis to Z, with strong pixel = 255 and weak pixel = 50. However a very low contrast ima
        for i in range(1, H-1):
            for j in range(1, W-1):
                if (hyster[i,j] == weak):
                    try:
                        if ((hyster[i+1, j-1] == strong) or (hyster[i+1, j] == strong) or (hyster[i+1, j+1] == strong)
                            or (hyster[i, j-1] == strong) or (hyster[i, j+1] == strong)
                            or (hyster[i-1, j-1] == strong) or (hyster[i-1, j] == strong) or (hyster[i-1, j+1] == strong)):
                            Z[i, j] = strong * 5000
                        else:
                            Z[i, j] = 0
                    except:
                        print("Error")
        io.imsave("EdgeDetec_Hyster.jpg", Z)
        #io.imshow(rebuilt)
        #io.imshow(Z)
        ''' 
        
myCannyEdge("York House_Fredericton.jpg", 'full')


# In[ ]:




