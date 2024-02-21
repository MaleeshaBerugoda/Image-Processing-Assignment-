import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/spider.png")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img_hsv)

sigma = 70

arange = np.linspace(0, 1, num=5)  

fig, axes = plt.subplots(1, len(arange) + 1, figsize=(5 * (len(arange) + 1), 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

for i, a in enumerate(arange, start=1):
   
    a = np.clip(a, 0, 1)
    
    transformed_s = np.minimum(s+a*128*np.exp(-(s-128)**2/(2*sigma**2)), 255).astype(np.uint8)

    img_hsv_transformed = cv2.merge([h, transformed_s, v])
    img_transformed = cv2.cvtColor(img_hsv_transformed, cv2.COLOR_HSV2BGR)

    axes[i].imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f'a = {a:.2f}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("Q3/SpidermanIntensity2.png")
plt.show()

