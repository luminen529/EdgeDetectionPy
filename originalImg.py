import numpy as np
import cv2

img = np.zeros((256, 256), dtype=np.uint8)
cv2.rectangle(img, (60, 60), (190, 190), 255, -1)