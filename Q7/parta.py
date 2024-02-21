import cv2 
from matplotlib import pyplot as plt
import numpy as np

im=cv2.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/a1q5images/im02small.png",cv2.IMREAD_GRAYSCALE)
im2=cv2.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/a1q5images/im02.png",cv2.IMREAD_GRAYSCALE)
s=4
rows=int(s*im.shape[0])
cols=int(s*im.shape[1])
zoomed=np.zeros((rows,cols),dtype=im.dtype)
for i in range(0,cols):
    for j in range(0,rows):
        orig_i = int(i / s)
        orig_j = int(j / s)
        zoomed[j, i] = im[orig_j, orig_i]
cv2.imshow("Original Image", im2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imshow("zoomed image", zoomed)
cv2.waitKey(0)
cv2.destroyAllWindows()
