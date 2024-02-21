import cv2
import numpy as np
import matplotlib.pyplot as plt
img1 = cv2.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/einstein.png", cv2.IMREAD_GRAYSCALE)

Px = np.array([[1], [2], [1]]) @ np.array([[1, 0, -1]])
Py = np.array([[1, 2, 1]]) @ np.array([[1], [0], [-1]])

Gx = cv2.filter2D(img1, -1, Px)
Gy = cv2.filter2D(img1, -1, Py)

gradient_magnitude = np.sqrt(Gx**2 + Gy**2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(img1, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(Gx, cmap='gray')
plt.title('Horizontal Edges')
plt.axis('off')

plt.savefig("Q6/partc.png")
plt.show()


