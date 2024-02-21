import cv2 
from matplotlib import pyplot as plt
import numpy as np

image=["/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/a1q5images/im01small.png",
       "/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/a1q5images/im02small.png"
       
      ]
image2=["/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/a1q5images/im01.png",
         "/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/a1q5images/im02.png"
         
         ]
no=['1','2']
s=4
for im_path,im2_path,number in zip(image,image2,no):
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    rows=int(s*im.shape[0])
    cols=int(s*im.shape[1])
    zoomed=np.zeros((rows,cols),dtype=im.dtype)
    for i in range(0,cols):
        for j in range(0,rows):
            orig_i = int(i / s)
            orig_j = int(j / s)
            zoomed[j, i] = im[orig_j, orig_i]

    height, width = im.shape

    new_width = width * s
    new_height = height * s

    interpolated_image = np.zeros((new_height, new_width), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            src_x = x * width // new_width
            src_y = y * height // new_height

            x0 = int(src_x)
            x1 = min(x0 + 1, width - 1)
            y0 = int(src_y)
            y1 = min(y0 + 1, height - 1)

            dx = src_x - x0
            dy = src_y - y0

            interpolated_values = (1 - dx) * (1 - dy) * im[y0, x0] + \
                             dx * (1 - dy) * im[y0, x1] + \
                             (1 - dx) * dy * im[y1, x0] + \
                             dx * dy * im[y1, x1]

            interpolated_image[y, x] = interpolated_values

    
    im2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)
    near_n=np.sum((zoomed.astype(np.float32) - im2.astype(np.float32)) ** 2) / np.prod(zoomed.shape)
    by_int= np.sum((interpolated_image.astype(np.float32) -im2.astype(np.float32)) ** 2) / np.prod(interpolated_image.shape)
    
    print("nearest neighbourhood SSD for image "+number, near_n)
    print("bylinear interpolation SSD for image "+number,by_int)
