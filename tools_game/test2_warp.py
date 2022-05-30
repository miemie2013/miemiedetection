'''
用
cv2.getPerspectiveTransform()
cv2.warpPerspective()
实现斜视图片
'''
import numpy as np

import cv2

img_bgr = cv2.imread('aaa.jpg')
h, w, c = img_bgr.shape

pts1 = np.float32([[0, 0], [1298, 0], [1298, 599], [0, 599]])
pts2 = np.float32([[0, 0], [1298, 100], [1298, 499], [0, 599]])

M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img_bgr, M, (w, h))

cv2.imwrite('warp.jpg', dst)




