import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img1 = cv.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/rice_gaussian_noise.png", cv.IMREAD_GRAYSCALE)

img2 = cv.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/rice_salt_pepper_noise.png", cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: Unable to read one or both images")
else:
    print("Images were read successfully")
kernel1= np.ones((5,5),np.uint8)
opening1 = cv.morphologyEx(img1, cv.MORPH_OPEN, kernel1)
opening2 = cv.morphologyEx(img2, cv.MORPH_OPEN, kernel1)
kernel2 = np.ones((5,5),np.uint8)
closing1 = cv.morphologyEx(opening1, cv.MORPH_CLOSE, kernel2)
closing2 = cv.morphologyEx(opening2, cv.MORPH_CLOSE, kernel2)

_, thresh_otsu1 = cv.threshold(opening1, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
_, thresh_otsu2 = cv.threshold(opening2, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU) 

retval1, labels2= cv.connectedComponents(thresh_otsu1,None, 8, cv.CV_32S)
retval2, labels2 = cv.connectedComponents(thresh_otsu2,None, 8, cv.CV_32S)

print("Number of connected components1:", retval1)
print("Number of connected components2:", retval2)


#cv.imshow("zoomed image", closing1)
#cv.waitKey(0)
#cv.destroyAllWindows()

#images = [img1, closing1,
 #         img2, closing1]

#titles = ['Original Noisy Image 1', "Removed objects",
#          'Original Noisy Image 2', "Removed objects"]

#plt.figure(figsize=(10, 8))
#for i in range(4):
 #   plt.subplot(2, 2, i+1)
  #  plt.imshow(images[i], 'gray')
   # plt.title(titles[i])
    #plt.xticks([]), plt.yticks([])
#plt.savefig("Q5/partf.png")
#plt.show()