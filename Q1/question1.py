import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

im1 = cv.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/margot_golden_gray.jpg")
assert im1 is not None

t = np.zeros(256, dtype=np.uint8)
#t[0:50] = np.array([int(x*200/255) for x in range(50)])
#t[50:151] = np.array([int(x*200/255 + 50) for x in range(50,151)])
#t[151:256] = np.array([int(x*200/255) for x in range(151,256)])
t[0:221] = np.array([int(x*200/255) for x in range(221)])
t[221:256] = np.array([int(x*200/255 + 50) for x in range(221,256)])
im2 = t[im1]

fig, ax = plt.subplots(1,2, figsize=(10,10))
ax[0].imshow(im1, vmin=0, vmax=255, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(im2, vmin=0, vmax=255, cmap='gray')
ax[1].set_title('Intensity Transformed')
plt.savefig("Q1/Intensity transform.png")
plt.show()
