import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img1 = cv.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/rice_gaussian_noise.png", cv.IMREAD_GRAYSCALE)

img2 = cv.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/rice_salt_pepper_noise.png", cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: Unable to read one or both images")
else:
    print("Images were read successfully")
kernel1= np.ones((2,2),np.uint8)
opening1 = cv.morphologyEx(img1, cv.MORPH_OPEN, kernel1)
opening2 = cv.morphologyEx(img2, cv.MORPH_OPEN, kernel1)
kernel2 = np.ones((3,3),np.uint8)
closing1 = cv.morphologyEx(opening1, cv.MORPH_CLOSE, kernel2)
closing2 = cv.morphologyEx(opening2, cv.MORPH_CLOSE, kernel2)

images = [img1, closing1,
          img2, closing1]

titles = ['Original Noisy Image 1', "Removed objects",
          'Original Noisy Image 2', "Removed objects"]

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.savefig("Q5/parte.png")
plt.show()