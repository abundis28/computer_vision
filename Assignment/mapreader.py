#!/usr/bin/env python3
# This is template for your program, for you to expand with all the correct 
# functionality.

import cv2, sys, numpy as np
from matplotlib.pyplot import get

from sympy import im

#-------------------------------------------------------------------------------
# Function declarations.
def blurImage(image):
    kernel = np.ones((5,5),np.float32)/25
    image = cv2.filter2D(image,-1,kernel)
    return image

def checkPositionGreenArrow(image):
    return True

def getPositionRA(image):
    return 0.5, 0.5

def getContours(image):
    # Find internal and external contours.
    contours, hierarchy = cv2.findContours (image, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

    rectangle = []   # list of dice contours
    shapes = []   # list of spot contours

    # Find dice contours, drawing the contours as we process them.
    for (i, c) in enumerate(hierarchy[0]):
        if c[3] == -1:
            rectangle.append (i)
            cv2.drawContours (image, contours, i, (0, 0, 255), 8)

    # Find spot contours, drawing them as we process them.
    for (i, c) in enumerate(hierarchy[0]):
        if c[3] in rectangle:
            shapes.append (i)
            cv2.drawContours (image, contours, i, (0, 255, 0), 8)
    
    return image, rectangle, shapes

def getDirectionRA(image):
    return 45

def getImageChannel(image, channel):
    temp = image.copy()
    if channel == 'r':
        temp[:,:,0] = 0
        temp[:,:,1] = 0
    elif channel == 'b':
        temp[:,:,2] = 0
        temp[:,:,1] = 0
    elif channel == 'g':
        temp[:,:,0] = 0
        temp[:,:,2] = 0
    grey = cv2.cvtColor (temp, cv2.COLOR_BGR2GRAY) 
    print("Shape after gray cvr: "+str(grey.shape))   
    return grey

def removeNoise(image):
    kernel = np.ones ((9,9), np.uint8)
    image = cv2.dilate (image, kernel, iterations=1)
    return image

def threshold(image):
    image_copy = getImageChannel(image, 'b')
    t, binary = cv2.threshold (image_copy, 0, 255, cv2.THRESH_BINARY
                                    + cv2.THRESH_OTSU)
    return binary, t

####     MAIN PROGRAM     ####
if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[1], file=sys.stderr)
    exit (1)
print ("The filename to work on is %s." % sys.argv[1])

image = cv2.imread (sys.argv[1])

threshold_image, thre = threshold(image)
print("Threshold: "+str(thre))
print("Shape after threshold: "+str(threshold_image.shape))

reduced_image = removeNoise(threshold_image)

contours_image, rect, shapes = getContours(reduced_image)
print("Num of rect: "+str(len(rect))+" // Num of shapes: "+str(len(shapes)))

xpos, ypos = getPositionRA(contours_image)
hdg = getDirectionRA(contours_image)

# Output the position and bearing in the form required by the test harness.
print ("POSITION %.3f %.3f" % (xpos, ypos))
print ("BEARING %.1f" % hdg)

#-------------------------------------------------------------------------------

cv2.namedWindow (sys.argv[0], cv2.WINDOW_NORMAL)
ny, nx = contours_image.shape
cv2.resizeWindow (sys.argv[0], nx//2, ny//2)
cv2.imshow (sys.argv[0], contours_image)
cv2.waitKey (0)
