#!/usr/bin/env python3
# This is template for your program, for you to expand with all the correct 
# functionality.

import cv2, sys, numpy as np

#-------------------------------------------------------------------------------

########    FUNCTION DECLARATIONS   ########
def isGreenArrowCorrectPosition(image, cropped_image, contours, indexList):
    for index in indexList:
        if cv2.contourArea(contours[index]) > 2000:
            moment = cv2.moments(contours[index])
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])
            b, g, r = cropped_image[cy,cx]
            if g > b + 20 and g > r + 20:
                cv2.drawContours (cropped_image, contours, index, (0, 0, 255), 5)
                return True, cropped_image
    return False, cropped_image

def getPositionRA(image):
    return 0.5, 0.5

def getDirectionRA(image):
    return 45

def blurImage(image):
    kernel = np.ones((5,5),np.float32)/25
    image = cv2.filter2D(image,-1,kernel)
    return image

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
    return grey

def removeNoise(image, sizeGaussian, sizeDilate):
    blur = cv2.GaussianBlur(image, (sizeGaussian, sizeGaussian), 0)
    kernel = np.ones((sizeDilate, sizeDilate), np.uint8)
    image = cv2.erode(blur, kernel, iterations=1)
    return image

def hsvFilter(image, hMax, sMax, vMax):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([hMax, sMax, vMax], np.uint8)
    binary = cv2.inRange(hsv, lower, upper)

    return binary

def getExternalContours(image, reduced_image):
    # Find internal and external contours.
    canny = cv2.Canny(reduced_image,100,150)
    contours, _ = cv2.findContours (canny, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

    rectangle = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rectangle)
    box = np.int0(box)
    
    return image, box

def getInternalContours(image, thres_image):
    # canny = cv2.Canny(thres_image,100,150)
    contours, hierarchy = cv2.findContours (thres_image, cv2.RETR_CCOMP,
                                        cv2.CHAIN_APPROX_SIMPLE)

    external = []
    internal = []
    for (i, c) in enumerate(hierarchy[0]):
        if c[3] == -1:
            external.append (i)
            # cv2.drawContours (image, contours, i, (0, 0, 255), 5)

    # Find spot contours, drawing them as we process them.
    for (i, c) in enumerate(hierarchy[0]):
        if c[3] in external:
            internal.append (i)
            # cv2.drawContours (image, contours, i, (0, 255, 0), 5)

    print(len(external), len(internal))
    return image, contours, external, internal

def crop_rect(image, filtered_image):
    _, box_points = getExternalContours(image, filtered_image)

    # Taken from the entry: "Perspective Transformation" by 
    # Kang & Atul in The AI Learner: 
    # https://theailearner.com/tag/cv2-getperspectivetransform/

    # All points are in format [cols, rows]
    pt_A = box_points[0]
    pt_B = box_points[1]
    pt_C = box_points[2]
    pt_D = box_points[3]

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

def threshold(image):
    filtered = hsvFilter(image, 179, 120, 210)
    reduced_image = removeNoise(filtered, 3, 3)

    binary = cv2.adaptiveThreshold(reduced_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    substracted = binary-reduced_image
    # substracted = cv2.bitwise_not(substracted)
    substracted_reduced = removeNoise(substracted, 5, 5)
    return substracted_reduced

########     MAIN PROGRAM     ########
if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[1], file=sys.stderr)
    exit (1)
print ("The filename to work on is %s." % sys.argv[1])

image = cv2.imread (sys.argv[1])

reduced_image = removeNoise(image, 9, 11)

filtered_image = hsvFilter(reduced_image, 179, 171, 226)

cropped_image = crop_rect(image, filtered_image)

thres_image  = threshold(cropped_image)

contour_image, internal_contours, external_list, internal_list = getInternalContours(cropped_image, thres_image)

boolFLag, cont_image = isGreenArrowCorrectPosition(image, cropped_image, internal_contours, external_list)

xpos, ypos = getPositionRA(filtered_image)
hdg = getDirectionRA(filtered_image)

# Output the position and bearing in the form required by the test harness.
print ("POSITION %.3f %.3f" % (xpos, ypos))
print ("BEARING %.1f" % hdg)

#-------------------------------------------------------------------------------

cv2.namedWindow ("thres_image", cv2.WINDOW_NORMAL)
ny, nx = thres_image.shape
cv2.resizeWindow ("thres_image", nx//2, ny//2)
cv2.imshow ("thres_image", thres_image)
cv2.waitKey (0)

cv2.namedWindow ("contour_image", cv2.WINDOW_NORMAL)
ny, nx, nz = cont_image.shape
cv2.resizeWindow ("contour_image", nx//2, ny//2)
cv2.imshow ("contour_image", cont_image)
cv2.waitKey (0)

# cv2.namedWindow ("substracted", cv2.WINDOW_NORMAL)
# ny, nx = substracted.shape
# cv2.resizeWindow ("substracted", nx//2, ny//2)
# cv2.imshow ("substracted", substracted)
# cv2.waitKey (0)