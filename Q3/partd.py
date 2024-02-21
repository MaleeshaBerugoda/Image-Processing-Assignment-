import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread("/Users/admin/Downloads/_DSC0726.JPG")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img_hsv)

a = 0.67  
sigma = 70  

transformed_s = np.minimum(s+a*128*np.exp(-(s-128)**2/(2*sigma**2)),255).astype(np.uint8)

img_hsv_transformed = cv2.merge([h, transformed_s, v])

img_transformed = cv2.cvtColor(img_hsv_transformed, cv2.COLOR_HSV2BGR)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
axes[1].set_title('Vibrance-enhanced Image')
axes[1].axis('off')

x = np.arange(0, 256)
transformation = np.minimum(x+a*128*np.exp(-(x-128)**2/(2*sigma**2)), 255)
axes[2].plot(x, transformation, color='red')
axes[2].set_title('Intensity Transformation')
axes[2].set_xlabel('Input Intensity (x)')
axes[2].set_ylabel('Output Intensity')
axes[2].set_xlim([0, 255])
axes[2].set_ylim([0, 255])

plt.tight_layout()
#plt.savefig("Q3/SpidermanIntensity.png")
plt.show()
cv2.imwrite("adithya.png",cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
