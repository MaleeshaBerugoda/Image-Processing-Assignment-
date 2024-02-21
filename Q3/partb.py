import cv2 
import numpy as np
import matplotlib.pyplot as plt

im1 = cv2.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/spider.png")
im2=cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)

hsv_image = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)

h,s,v=cv2.split(hsv_image)
a = 0.5
sigma = 70
transformed_s = np.minimum(s+a*128*np.exp(-(s-128)**2/(2*sigma**2)),255).astype(np.uint8)
img_hsv_transformed = cv2.merge([h, transformed_s, v])
img_transformed = cv2.cvtColor(img_hsv_transformed, cv2.COLOR_HSV2RGB)

images = [im2, transformed_s,
          img_hsv_transformed, img_transformed]

titles = ['Original Image', "sacturation transformed",
          'HSV merged image', "Transformed image"]

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.savefig("Q3/sacturationplane.png")
plt.show()
#cv2.imshow("Original Image", im2)
#cv2.imshow("Transformed Image", img_transformed)
#cv2.imwrite("output_image.jpg", img_transformed)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
