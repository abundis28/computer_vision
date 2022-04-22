import cv2, sys, numpy as np
from matplotlib.pyplot import get

if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[1], file=sys.stderr)
    exit (1)
print ("The filename to work on is %s." % sys.argv[1])

image = cv2.imread (sys.argv[1])
b_image = image[:,:,0]

cv2.namedWindow (sys.argv[0], cv2.WINDOW_NORMAL)
ny, nx, nc = image.shape
cv2.resizeWindow (sys.argv[0], nx//2, ny//2)
cv2.imshow (sys.argv[0], b_image)
cv2.waitKey (0)