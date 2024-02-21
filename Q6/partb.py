from scipy import datasets,ndimage
import matplotlib.pyplot as plt
import cv2
import numpy as np
img1 = cv2.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/einstein.png", cv2.IMREAD_GRAYSCALE)
ascent = datasets.ascent().astype('int32')
sobel_h = ndimage.sobel(img1, axis=0) 
sobel_v = ndimage.sobel(img1, axis=1) 
magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
#magnitude *= 255.0 / np.max(magnitude)  

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plt.gray()  
axs[0, 0].imshow(img1)
axs[0, 1].imshow(sobel_h)
axs[1, 0].imshow(sobel_v)
axs[1, 1].imshow(magnitude)
titles = ["original", "horizontal", "vertical", "magnitude"]
for i, ax in enumerate(axs.ravel()):
    ax.set_title(titles[i])
    ax.axis("off")
plt.savefig("Q6/partb.png")
plt.show()
