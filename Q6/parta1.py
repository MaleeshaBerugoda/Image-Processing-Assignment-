import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img=cv.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/einstein.png",cv.IMREAD_REDUCED_GRAYSCALE_2)

kernel=np.array([(1,0,-1),(2,0,-2),(1,0,-1)],dtype='float')
image=cv.filter2D(img,-1,kernel)
fig,axes=plt.subplots(1,2,sharex='all',sharey='all',figsize=(5,5))
axes[0].imshow(img,cmap='gray')
axes[0].set_title('original')
axes[0].set_xticks([]),axes[0].set_yticks([])
axes[1].imshow(image,cmap='gray')
axes[1].set_title('Filter2d image')
axes[1].set_xticks([]),axes[1].set_yticks([])
plt.savefig("Q6/parta.png")
plt.show()