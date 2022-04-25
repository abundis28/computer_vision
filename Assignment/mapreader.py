#!/usr/bin/env python3

# Name: Andres Abundis Correa
# Reg Number: aa2100995

# Program name: mapreader.py
# Purpose:  These program will take images of a museum-provided map
#           that is placed on top of a blue background. The program
#           will then indicate the position (scaled to a previously
#           determined grid) of the pointer of a red arrow that is
#           present on all maps and the direction that it is pointing
#           towards in degrees (from the north direction and in a 
#           clockwise direction).

import cv2, sys, math, numpy as np

#-------------------------------------------------------------------------------

########    FUNCTION DECLARATIONS   ########

def removeNoise(image, sizeGaussian, sizeDilate):
    # Utility function that is used in several other functions to
    # reduce noise to facilitate thresholding.

    blur = cv2.GaussianBlur(image, (sizeGaussian, sizeGaussian), 0)
    kernel = np.ones((sizeDilate, sizeDilate), np.uint8)
    eroded_image = cv2.erode(blur, kernel, iterations=1)

    return eroded_image

def hsvFilter(image, hMax, sMax, vMax):
    # Segment different parts of an image depending on the hue,
    # saturation and value parameters that are passed.

    # Change image to the HSV colorspace.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range of hue, saturation and value to filter according to.
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([hMax, sMax, vMax], np.uint8)
    # Filter
    binary = cv2.inRange(hsv, lower, upper)

    return binary

def getExternalContours(image, reduced_image):

    # Adapted from the article "Edges and Contours Basics with OpenCV" by
    # Thiago Carvalho in Towards Data Science:
    # https://bit.ly/towards-datascience-edges-and-contours

    # Find internal and external contours.
    # The canny method of cv2 was used to improve the contours.
    canny = cv2.Canny(reduced_image,100,150)
    contours, _ = cv2.findContours (canny, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    # Define minimum bounding box.
    rectangle = cv2.minAreaRect(contours[0])
    # Get four corners of min bounding box.
    box = cv2.boxPoints(rectangle)
    box = np.int0(box)

    return image, box

def getInternalContours(thres_image):

    # Adapted from the starter code from experiment 3 of CE866 designed by
    # Dr. Adrian Clark.

    # Find internal and external contours.
    contours, hierarchy = cv2.findContours(thres_image, cv2.RETR_CCOMP,
                                        cv2.CHAIN_APPROX_SIMPLE)
    # Separate the indexes of external and internal contours.
    external = []
    internal = []
    for (i, c) in enumerate(hierarchy[0]):
        if c[3] == -1:
            external.append (i)
    for (i, c) in enumerate(hierarchy[0]):
        if c[3] in external:
            internal.append (i)

    return contours, external, internal

def crop_rect(image, filtered_image):
    _, box_points = getExternalContours(image, filtered_image)

    # Taken from the entry: "Perspective Transformation" by 
    # Kang & Atul in The AI Learner: 
    # https://theailearner.com/tag/cv2-getperspectivetransform/

    # Define the four corners of the map.
    pt_A = box_points[0]
    pt_B = box_points[1]
    pt_C = box_points[2]
    pt_D = box_points[3]

    # Calculate the lengths of the sides of the images.
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))
    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    # Prepare input and output parameters for the transformation.
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])
    # Crop the image to the map.
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    cropped_image = cv2.warpPerspective(image,M,(maxWidth, maxHeight),
                    flags=cv2.INTER_LINEAR)
    
    return cv2.flip(cropped_image, 0)

def threshold(image, sizeGaussian1, sizeDilate1, sizeGaussian2, sizeDilate2):

    # Adapted from the starter code from experiment 3 of CE866 designed by
    # Dr. Adrian Clark.

    # Filter through HSV color space. The HSV values were obtained through trial
    # and error.
    filtered = hsvFilter(image, 179, 120, 210)
    # Reduce noise
    reduced_image = removeNoise(filtered, sizeGaussian1, sizeDilate1) 
    binary = cv2.adaptiveThreshold( 
                                    reduced_image,255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                    cv2.THRESH_BINARY,11,2
                                  )
    # Substract the binary obtained by the adaptive thresholding so that only
    # desired values remain.
    substracted = binary-reduced_image
    # Remove noise one last time.
    substracted_reduced = removeNoise(substracted, sizeGaussian2, sizeDilate2)
    return substracted_reduced

def getCentroid(image, contours, indexList, color):
    for index in indexList:
        # Adapted from OpenCV Contour Features tutorial:
        # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        if cv2.contourArea(contours[index]) > 2000:
            moment = cv2.moments(contours[index])
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])
            b, g, r = image[cy,cx]
            # Get centroid of either green or red figures in the image as
            # determined by the color parameter.
            if color == 'g':
                if g > b + 20 and g > r + 20:
                    break
            if color == 'r':
                if r > b + 20 and r > g + 20:
                    break
    return cx, cy, index

def correctOrientation(image, x_centroid, y_centroid):
    # Get half limits of the image to divide it into quadrants.
    height, width, _ = image.shape
    h_limit, w_limit = height//2, width//2
    # Check that the passed centroid coordinates fall in the top left quadrant.
    if y_centroid > h_limit and x_centroid < w_limit:
        corrected = cv2.flip(image, -1)
    else:
        corrected = image
    return corrected

def getRepeatedCoordinate(coord_1_1, coord_1_2, coord_2_1, coord_2_2):
    # Four points are passed (starting and ending points of 2 vectors).
    # The possible combinations are checked to see which starting point is
    # shared by both vectors.
    if list(coord_1_1) == list(coord_2_1) or list(coord_1_1) == list(coord_2_2):
        return coord_1_1
    if list(coord_1_2) == list(coord_2_1) or list(coord_1_2) == list(coord_2_2):
        return coord_1_2

def getTopCorner(t_corners):
    # The function gets the top corner of an iscoceles triangle.
    # The length of the three sides are calculated.
    dist1 = math.dist(t_corners[0][0], t_corners[1][0])
    dist2 = math.dist(t_corners[1][0], t_corners[2][0])
    dist3 = math.dist(t_corners[0][0], t_corners[2][0])
    # Check which two sides are the largest and return their common vertex.
    if dist1 > dist3 and dist2 > dist3:
        return getRepeatedCoordinate(   t_corners[0][0], t_corners[1][0], 
                                        t_corners[1][0], t_corners[2][0]
                                    )
    if dist2 > dist1 and dist3 > dist1:
        return getRepeatedCoordinate(   t_corners[1][0], t_corners[2][0],
                                        t_corners[0][0], t_corners[2][0]
                                    )
    if dist1 > dist2 and dist3 > dist2:
        return getRepeatedCoordinate(   t_corners[0][0], t_corners[1][0],
                                        t_corners[0][0], t_corners[2][0]
                                    )

def getPositionRA(image, internal_contours, external_list):
    # The function gets the position of the pointer of the red arrow.
    # Get the centroid of the red arrow.
    x_pixels, y_pixels, i_red_contour = getCentroid(image, 
                                                    internal_contours, 
                                                    external_list, 'r'
                                                   )
    # Obtain the min enclosing triangle.
    red_pointer_approx = cv2.minEnclosingTriangle(
                                        internal_contours[i_red_contour]
                                    )
    # Find the top corner of the iscoceles triangle.
    coord_corner = np.int32(getTopCorner(red_pointer_approx[1]))
    # Scale pointer coordinates.
    xpos = 1.0 - ((coord_corner[1] * 1.0)/image.shape[0])
    ypos = ((coord_corner[0] * 1.0)/image.shape[1])
    scaled_coord = [xpos, ypos]
    coord = [x_pixels, y_pixels]
    return scaled_coord, coord, coord_corner, i_red_contour
    
def getDirectionRA(correct_image, c_center, c_corner):
    # Get angle in radians between vertical northern line and direction of the
    # red arrow pointer.
    angle = math.atan2((c_center[0]-c_corner[0]), (c_center[1]-c_corner[1]))
    angle = np.rad2deg(angle)
    if angle < 0:
        angle *= -1
    else:
        angle = 360-angle
    return angle
    
    
########     MAIN PROGRAM     ########
if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[1], file=sys.stderr)
    exit (1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
original_image = cv2.imread (sys.argv[1])
image = original_image.copy()

# Prepare the image and threshold through HSV to crop the blue background out.
reduced_image = removeNoise(image, 9, 11)
filtered_image = hsvFilter(reduced_image, 179, 171, 226)
crop_image = crop_rect(image, filtered_image)

# Threshold and get contours of red and green arrow inside the map.
thres_image1  = threshold(crop_image, 3, 3, 5, 5)
i_contours, e_list, _ = getInternalContours(thres_image1)
# Get centroid of the green arrow and rotate 180 degrees if the arrow is not in
# the top right quadrant.
x_centroid, y_centroid, _ = getCentroid(crop_image, i_contours, e_list, 'g')
correct_image = correctOrientation(crop_image, x_centroid, y_centroid)

# Threshold and get contours of the green and red arrows.
thres_image2  = threshold(correct_image, 3, 1, 5, 9)
i_contours, e_list, _ = getInternalContours(thres_image2)
# Get position of of red arrow.
coord_trans, coord_center, coord_corner, _ = getPositionRA( correct_image, 
                                                            i_contours,
                                                            e_list
                                                          )
# Get direction of the pointer of the red arrow.
hdg = getDirectionRA(correct_image, coord_center, coord_corner)

# Output the position and bearing in the form required by the test harness.
print ("POSITION %.3f %.3f" % (coord_trans[1], coord_trans[0]))
print ("BEARING %.1f" % hdg)

#-------------------------------------------------------------------------------
