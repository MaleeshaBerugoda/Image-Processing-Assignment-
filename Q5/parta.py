import numpy as np 
import cv2 
from matplotlib import pyplot as plt 

image = cv2.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/rice_gaussian_noise.png") 
noise_removed_img=	cv2.fastNlMeansDenoising(image, None,40,7,21) 

fig, axarr = plt.subplots(2,2)
axarr[0, 0].imshow(image)
axarr[0, 0].set_title('Original Image')
axarr[0, 1].imshow(noise_removed_img)
axarr[0, 1].set_title('image after noise remove')

images = [image, noise_removed_img]

titles = ['Original Image', "image after noise remove"]

plt.figure(figsize=(10, 8))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.savefig("Q5/parta.png")
plt.show()