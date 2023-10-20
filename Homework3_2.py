"""
    MSSV     : N20DCCN083
    Họ và tên: Nguyễn Thái Trưởng
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

binary_image = np.fromfile('lady.bin', dtype=np.uint8, sep="")
# print(binary_image)

img_lady = np.array(binary_image.reshape(256, 256))
cv2.imwrite("Lady.png", img_lady)

plt.hist(binary_image.ravel(), bins=256, range=(0, 256), density=True, color='b', alpha=0.6)
plt.xlabel('Pixel Value')
plt.ylabel('Normalized Frequency')
plt.title('Histogram of Original Image')
plt.xlim(0, 256)
plt.grid()

min_pixel = np.min(img_lady)
max_pixel = np.max(img_lady)
stretched_image = ((img_lady - min_pixel) / (max_pixel - min_pixel) * 255).astype(np.uint8)

plt.figure()
plt.hist(stretched_image.ravel(), bins=256, range=(0, 300), density=True, color='b', alpha=0.6)
plt.xlabel('Pixel Value')
plt.ylabel('Normalized Frequency')
plt.title('Histogram of Stretched Image')
plt.xlim(0, 256)
plt.grid()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img_lady, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(stretched_image, cmap='gray')
plt.title('Stretched Image')

plt.show()
