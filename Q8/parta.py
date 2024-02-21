import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/daisy.jpg")
assert img is not None, "file could not be read, check with os.path.exists()"
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (30,150,539,400)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
forground = np.where((mask==0)|(mask==2),0,1).astype('uint8')
background = np.where((mask==0)|(mask==2),1,0).astype('uint8')
#flower=np.where((fgdModel==1,0,1))
#blurred_background=cv.blur(src=img, ksize=(40, 40))

img1 = img*forground[:,:,np.newaxis]
img2 = img*background[:,:,np.newaxis]
blurred_background=cv.blur(src=img2, ksize=(40, 40))
#img3 = blurred_background*background[:,:,np.newaxis]
img4 = cv.add(blurred_background,img1)

images = [forground, cv.cvtColor(img1,cv.COLOR_BGR2RGB) ,cv.cvtColor(img2,cv.COLOR_BGR2RGB),
          cv.cvtColor(img4,cv.COLOR_BGR2RGB), cv.cvtColor(img,cv.COLOR_BGR2RGB),]

titles = ['Final Segment Mask', "Forground Image",
          'Background image', "Background Blurred image",'Original image']

plt.figure(figsize=(10, 8))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.savefig("Q8/flower.png")
plt.show()

#plt.show()
#plt.imshow(cv.cvtColor(blurred_background,cv.COLOR_BGR2RGB),cmap='gray')
plt.imshow(cv.cvtColor(blurred_background,cv.COLOR_BGR2RGB))
#plt.show()