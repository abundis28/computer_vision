#!/usr/bin/env python3
# This is template for your program, for you to expand with all the correct
# functionality.

import cv2, sys, numpy as np
from matplotlib.pyplot import get

from sympy import im

#-------------------------------------------------------------------------------
# Main program.
def checkPositionGreenArrow(image):
    return

def getImageChannel(image, channel):
    if channel == 'r':
        return image[:,:,2]
    elif channel == 'b':
        return image[:,:,0]
    elif channel == 'g':
        return image[:,:,1]

def threshold(image):
    image_copy = getImageChannel(image, 'b')
    # grey = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    t, binary = cv2.threshold (image_copy, 0, 255, cv2.THRESH_BINARY
                                    + cv2.THRESH_OTSU)
    return binary, t
    # binary = cv2.adaptiveThreshold (grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
    #                                 cv2.THRESH_BINARY, 31, 4)
    # return binary

# Ensure we were invoked with a single argument.

if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[1], file=sys.stderr)
    exit (1)
print ("The filename to work on is %s." % sys.argv[1])

image = cv2.imread (sys.argv[1])
print("Unprocessed image's shape: "+str(image.shape))

t_image, thre = threshold(image)
print("Threshold: "+str(thre))
# t_image = threshold(image)
# t_image = getImageChannel(image, 'b')

print("Processed image's shape: "+str(t_image.shape))

xpos = 0.5
ypos = 0.5
hdg = 45.1

# Output the position and bearing in the form required by the test harness.
print ("POSITION %.3f %.3f" % (xpos, ypos))
print ("BEARING %.1f" % hdg)

#-------------------------------------------------------------------------------

cv2.namedWindow (sys.argv[0], cv2.WINDOW_NORMAL)
ny, nx, nc = image.shape
cv2.resizeWindow (sys.argv[0], nx//2, ny//2)
cv2.imshow (sys.argv[0], t_image)
cv2.waitKey (0)
