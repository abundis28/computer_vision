import cv2

im = cv2.imread("Labs/lab-02/banana-01.png")

nx, ny, nz = im.shape

# # Display the result.
# cv2.namedWindow ("banana", cv2.WINDOW_NORMAL)
# ny, nx, nc = im.shape
# cv2.resizeWindow ("banana", nx//2, ny//2)
# cv2.imshow ("banana", im)
# cv2.waitKey (0)

print(im.var())