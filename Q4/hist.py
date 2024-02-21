import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/shells.tif",0)

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

equ = cv2.equalizeHist(img)
#res = np.hstack((img,equ)) #stacking images side-by-side
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(equ, cv2.COLOR_BGR2RGB))
axes[1].set_title('Vibrance-enhanced Image')
axes[1].axis('off')

#plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('histogram'), loc = 'upper left')
plt.savefig("Q4/Histogram.png")
plt.show()