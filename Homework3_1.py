"""
    MSSV     : N20DCCN083
    Họ và tên: Nguyễn Thái Trưởng
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

binary_image = np.fromfile('Mammogram256x256.bin', dtype='uint8', sep="")
# print(binary_image)

img_mammogram = np.array(binary_image.reshape(256, 256))
# print(img_mammogram)

threshold_value = 127

_, binary_image = cv2.threshold(img_mammogram, threshold_value, 255, cv2.THRESH_BINARY)

cv2.imwrite("BinaryMammogram.png", binary_image)
def generate_contour_image(binaryImage):
    contour_image = np.zeros_like(binaryImage, dtype=np.uint8)

    for y in range(1, binaryImage.shape[0] - 1):
        for x in range(1, binaryImage.shape[1] - 1):
            if binaryImage[y, x] == 255:
                neighbors = [
                    binaryImage[y - 1, x - 1],
                    binaryImage[y - 1, x],
                    binaryImage[y - 1, x + 1],
                    binaryImage[y, x - 1],
                    binaryImage[y, x + 1],
                    binaryImage[y + 1, x - 1],
                    binaryImage[y + 1, x],
                    binaryImage[y + 1, x + 1],
                ]
                if 0 in neighbors:
                    contour_image[y, x] = 255

    return contour_image

contour_image = generate_contour_image(binary_image)
cv2.imwrite("ContourMammogram.png", contour_image)
fig, axes = plt.subplots(1, 3, figsize=(16, 8))
figManager = plt.get_current_fig_manager()

axes[0].imshow(img_mammogram, cmap='gray')
axes[0].set_title('Origin Image')

axes[1].imshow(binary_image, cmap='gray')
axes[1].set_title('Binary Mammogram')

axes[2].imshow(contour_image, cmap='gray')
axes[2].set_title('Contour Image');

plt.tight_layout()
plt.bar(np.arange(256), 0)
plt.show()
