"""
    MSSV     : N20DCCN083
    Họ và tên: Nguyễn Thái Trưởng
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

binary_image = np.fromfile('johnny.bin', dtype=np.uint8, sep="")

img_johnny = np.array(binary_image.reshape(256, 256))

equalized_image = cv2.equalizeHist(img_johnny)

plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(img_johnny, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Equalized Image")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Histogram of Original Johnny Image")
plt.hist(img_johnny.ravel(), 256, [0, 256])
plt.xlim(0, 255)
# plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Histogram of Equalized Johnny Image")
plt.hist(equalized_image.ravel(), 256, [0, 256])
plt.xlim(0, 255)
# plt.axis('off')

plt.tight_layout()
plt.show()
# fig, axes = plt.subplots(1, 4, figsize=(16, 8))
# figManager = plt.get_current_fig_manager()
# axes[0].imshow(img_johnny, cmap='gray')
# axes[0].set_title('Original Image')
#
# axes[1].imshow(equalized_image, cmap='gray')
# axes[1].set_title('Equalized Image')
#
# axes[2].hist(img_johnny.ravel(), 256, [0, 256])
# axes[2].set_title('Histogram (Original)')
# axes[2].set_xlim(0, 255)
#
# axes[3].hist(equalized_image.ravel(), 256, [0, 256])
# axes[3].set_title('Histogram (Equalized)')
# axes[3].set_xlim(0, 255)
#
# plt.tight_layout()
# plt.bar(np.arange(256), 0)
# plt.show()
