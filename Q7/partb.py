import numpy as np
import cv2

image = cv2.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/a1q5images/im02small.png", cv2.IMREAD_GRAYSCALE)
im2=cv2.imread("/Users/admin/Desktop/Maleesha’s Air/Maleesha's Air/KDU/Semester 5/Image processing and machine vision/Assignment/assignment_01_images/a1q5images/im02.png",cv2.IMREAD_GRAYSCALE)
s = 4
height, width = image.shape

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

        interpolated_values = (1 - dx) * (1 - dy) * image[y0, x0] + \
                             dx * (1 - dy) * image[y0, x1] + \
                             (1 - dx) * dy * image[y1, x0] + \
                             dx * dy * image[y1, x1]

        interpolated_image[y, x] = interpolated_values

cv2.imshow('Original Image', im2)
cv2.imshow('Interpolated Image', interpolated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
