import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('image1.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg',cv2.IMREAD_GRAYSCALE)

ret1, binarized_img1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
ret2, binarized_img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
# AND
and_img = cv2.bitwise_and(binarized_img1,binarized_img2)
# OR
or_img = cv2.bitwise_or(binarized_img1,binarized_img2)
# NOT
not_img = cv2.bitwise_not(binarized_img1)

#출력
images = [img1, img2,and_img, or_img, not_img]
titles = ['Image 1', 'Image 2', 
          'AND', 'OR', 'image1 NOT']

plt.figure(figsize=(12, 10))
for i in range(5):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()