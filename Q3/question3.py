import cv2 
import numpy as np
import matplotlib.pyplot as plt

im1 = cv2.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/spider.png")
im2=cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)

hsv_image = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)

h,s,v=cv2.split(hsv_image)
fig, axarr = plt.subplots(2,2)  # Create a 3x2 grid of subplots

axarr[0, 0].imshow(im2)
axarr[0, 0].set_title('Original Image')
axarr[0, 1].imshow(h)
axarr[0, 1].set_title('Hue')
axarr[1, 0].imshow(s)
axarr[1, 0].set_title('Saturation')
axarr[1, 1].imshow(v)
axarr[1, 1].set_title('value')
plt.savefig("Q3/Spiderman.png")
plt.show()

#cv2.imshow("Hue",h)
#cv2.imshow("saturation",s)
#cv2.imshow("value",v)
#cv2.waitKey(0)
#cv2.destroyAllWindows()