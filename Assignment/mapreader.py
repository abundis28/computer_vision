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
    blur = cv2.GaussianBlur (image, (7, 7), 0)
    kernel = np.ones ((15,15), np.uint8)
    image = cv2.dilate (blur, kernel, iterations=1)
    return image

def hsvFilter(image):
    # image_copy = getImageChannel(image, 'b')
    # t, binary = cv2.threshold (image_copy, 0, 255, cv2.THRESH_BINARY
    #                                 + cv2.THRESH_OTSU)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([179, 171, 226], np.uint8)
    binary = cv2.inRange(hsv, lower, upper)

    return binary

def getExternalContours(image, reduced_image):
    # Find internal and external contours.
    canny = cv2.Canny(reduced_image,100,150)
    contours, hierarchy = cv2.findContours (canny, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

    rectangle = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rectangle)
    box = np.int0(box)
    cv2.drawContours(image,[box],0,(0,0,255),2)
    
    return image, rectangle, box

def crop_rect(image, rectangle, box):
    # All points are in format [cols, rows]
    pt_A = box[0]
    pt_B = box[1]
    pt_C = box[2]
    pt_D = box[3]

    # Here, I have used L2 norm. You can use L1 also.
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))
    
    
    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])
	
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    out = cv2.warpPerspective(image,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)

    return out

####     MAIN PROGRAM     ####
if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[1], file=sys.stderr)
    exit (1)
print ("The filename to work on is %s." % sys.argv[1])

image = cv2.imread (sys.argv[1])

reduced_image = removeNoise(image)

filtered_image = hsvFilter(reduced_image)

contours_image, rect, box_points = getExternalContours(image, filtered_image)
print(box_points)
# cropped_img, rotated_image = crop_rect(image, rect)
cropped_image = crop_rect(image, rect, box_points)

xpos, ypos = getPositionRA(filtered_image)
hdg = getDirectionRA(filtered_image)

# Output the position and bearing in the form required by the test harness.
print ("POSITION %.3f %.3f" % (xpos, ypos))
print ("BEARING %.1f" % hdg)

#-------------------------------------------------------------------------------

cv2.namedWindow ("processed_image", cv2.WINDOW_NORMAL)
ny, nx = filtered_image.shape
cv2.resizeWindow ("processed_image", nx//2, ny//2)
cv2.imshow ("processed_image", filtered_image)
cv2.waitKey (0)

cv2.namedWindow ("contours_image", cv2.WINDOW_NORMAL)
ny, nx, nz = contours_image.shape
cv2.resizeWindow ("contours_image", nx//2, ny//2)
cv2.imshow ("contours_image", contours_image)
cv2.waitKey (0)

cv2.namedWindow ("cropped_image", cv2.WINDOW_NORMAL)
ny, nx, nz = cropped_image.shape
cv2.resizeWindow ("cropped_image", nx//2, ny//2)
cv2.imshow ("cropped_image", cropped_image)
cv2.waitKey (0)