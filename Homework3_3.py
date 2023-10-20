"""
    MSSV     : N20DCCN083
    Họ và tên: Nguyễn Thái Trưởng
"""

import cv2
import numpy as np

binary_image = np.fromfile('actontBin.bin', dtype=np.uint8, sep="")
img_actont = np.array(binary_image.reshape(256, 256))
cv2.imwrite("Actont.png", img_actont)

template = np.array([[1, 1, 1],
                    [0, 1, 0],
                    [0, 1, 0]], dtype=np.uint8)

template_height, template_width = template.shape


threshold = 100  # Adjust as needed

output_image = np.zeros_like(img_actont, dtype=np.uint8)

for y in range(img_actont.shape[0] - template_height + 1):
    for x in range(img_actont.shape[1] - template_width + 1):
        region_of_interest = img_actont[y:y+template_height, x:x+template_width]

        match_measure = np.sum(np.logical_and(region_of_interest, template))

        output_image[y, x] = match_measure


binary_result = (output_image > threshold).astype(np.uint8) * 255

cv2.imwrite('J2.png', binary_result)
