#!/usr/bin/env python3
# This is template for your program, for you to expand with all the correct 
# functionality.

import cv2, sys, math, numpy as np

#-------------------------------------------------------------------------------

########    FUNCTION DECLARATIONS   ########

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

def getInternalContours(thres_image):
    # Adapted from the starter code from experiment 3 of CE866 designed by
    # Dr. Adrian Clark.
    contours, hierarchy = cv2.findContours(thres_image, cv2.RETR_CCOMP,
                                        cv2.CHAIN_APPROX_SIMPLE)
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

    # Adapted from the entry: "Perspective Transformation" by 
    # Kang & Atul in The AI Learner: 
    # https://theailearner.com/tag/cv2-getperspectivetransform/

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
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    cropped_image = cv2.warpPerspective(image,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
    
    return cv2.flip(cropped_image, 0)

def threshold(image, sizeGaussian1, sizeDilate1, sizeGaussian2, sizeDilate2):
    filtered = hsvFilter(image, 179, 120, 210)
    reduced_image = removeNoise(filtered, sizeGaussian1, sizeDilate1)
    binary = cv2.adaptiveThreshold(reduced_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    substracted = binary-reduced_image
    # substracted = cv2.bitwise_not(substracted)
    substracted_reduced = removeNoise(substracted, sizeGaussian2, sizeDilate2)
    return substracted_reduced

def getCentroid(image, contours, indexList, color):
    for index in indexList:
        if cv2.contourArea(contours[index]) > 2000:
            # Adapted from OpenCV Contour Features tutorial:
            # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
            moment = cv2.moments(contours[index])
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])
            b, g, r = image[cy,cx]
            if color == 'g':
                if g > b + 20 and g > r + 20:
                    break
            if color == 'r':
                if r > b + 20 and r > g + 20:
                    # cv2.drawContours(image, contours, index, (0, 0, 255), 5)
                    # print(cx, cy)
                    break
    return cy, cx, index

def correctOrientation(image, y_centroid, x_centroid):
    height, width, _ = image.shape
    h_limit, w_limit = height//2, width//2
    if x_centroid > h_limit and y_centroid < w_limit:
        corrected = cv2.flip(image, -1)
    else:
        corrected = image
    return corrected

def getRepeatedCoordinate(coord_1_1, coord_1_2, coord_2_1, coord_2_2):
    if coord_1_1.all() == coord_2_1.all() or coord_1_1.all() == coord_2_2.all():
        return coord_1_1
    elif coord_1_2.all() == coord_2_1.all() or coord_1_2.all() == coord_2_2.all():
        return coord_1_2

def getTopCorner(t_corners):
    print(t_corners)
    dist1 = math.dist(t_corners[0][0], t_corners[1][0])
    dist2 = math.dist(t_corners[1][0], t_corners[2][0])
    dist3 = math.dist(t_corners[0][0], t_corners[2][0])
    if dist1 > dist3 and dist2 > dist3:
        print(1)
        repeated = getRepeatedCoordinate(t_corners[0][0], t_corners[1][0], t_corners[1][0], t_corners[2][0])
    if dist2 > dist1 and dist3 > dist1:
        print(2)
        repeated = getRepeatedCoordinate(t_corners[1][0], t_corners[2][0], t_corners[0][0], t_corners[2][0])
    if dist1 > dist2 and dist3 > dist2:
        print(3)
        repeated = getRepeatedCoordinate(t_corners[0][0], t_corners[1][0], t_corners[0][0], t_corners[2][0])
    print(repeated)
    return repeated

def getPositionRA(image, internal_contours, external_list):
    x_pixels, y_pixels, index_red_contour = getCentroid(image, internal_contours, external_list, 'r')
    red_pointer_approx = cv2.minEnclosingTriangle(internal_contours[index_red_contour])
    #print(x_pixels, y_pixels)
    #print(red_pointer_approx)
    coord_corner = np.int32(getTopCorner(red_pointer_approx[1]))
    image = cv2.polylines(image, np.int32([red_pointer_approx[1]]), True, (0,0,255), 4)
    image = cv2.circle(image, (y_pixels,x_pixels), 9, (0,255,0), -1)
    # image = cv2.circle(image, coord_corner, 8, (0,255,0), -1)
    xpos = 1.0 - ((x_pixels * 1.0)/image.shape[0])
    ypos = ((y_pixels * 1.0)/image.shape[1])
    return [xpos, ypos], [x_pixels, y_pixels], coord_corner, image, internal_contours, index_red_contour
    
def getDirectionRA(correct_image, c_center, c_corner):
    line_image = cv2.line(correct_image, (c_center[1], 0), (c_center[1], c_center[0]), (0, 0, 0), thickness=3)
    line_image = cv2.line(line_image, (c_center[1], c_center[0]), (c_corner[0], c_corner[1]), (0, 0, 0), thickness=3)
    return _, line_image
    
    
########     MAIN PROGRAM     ########
if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[1], file=sys.stderr)
    exit (1)

original_image = cv2.imread (sys.argv[1])
image = original_image.copy()

reduced_image = removeNoise(image, 9, 11)
filtered_image = hsvFilter(reduced_image, 179, 171, 226)
crop_image = crop_rect(image, filtered_image)

thres_image1  = threshold(crop_image, 3, 3, 5, 5)
internal_contours, external_list, internal_list = getInternalContours(thres_image1)
x_centroid, y_centroid, _ = getCentroid(crop_image, internal_contours, external_list, 'g')
correct_image = correctOrientation(crop_image, y_centroid, x_centroid)

thres_image2  = threshold(correct_image, 3, 1, 5, 9)
internal_contours, external_list, internal_list = getInternalContours(thres_image2)
coord_trans, coord_center, coord_corner, red_image, cnts, index_red = getPositionRA(correct_image, internal_contours, external_list)
hdg, line_image = getDirectionRA(correct_image, coord_center, coord_corner)

# Output the position and bearing in the form required by the test harness.
print ("POSITION %.3f %.3f" % (coord_trans[0], coord_trans[1]))
print ("BEARING %.1f" % 0)

#-------------------------------------------------------------------------------

cv2.namedWindow ("original_image", cv2.WINDOW_NORMAL)
ny, nx, nz = original_image.shape
cv2.resizeWindow ("original_image", nx//2, ny//2)
cv2.imshow ("original_image", original_image)
cv2.waitKey (0)

cv2.namedWindow ("red_image", cv2.WINDOW_NORMAL)
ny, nx, nz = red_image.shape
cv2.resizeWindow ("red_image", nx//2, ny//2)
cv2.imshow ("red_image", correct_image)
cv2.waitKey (0)

cv2.namedWindow ("line_image", cv2.WINDOW_NORMAL)
ny, nx, nz = line_image.shape
cv2.resizeWindow ("line_image", nx//2, ny//2)
cv2.imshow ("line_image", line_image)
cv2.waitKey (0)

