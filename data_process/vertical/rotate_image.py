# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     rotate_image
   Description :
   Author :       'li'
   date：          2018/8/1
-------------------------------------------------
   Change Activity:
                   2018/8/1:
-------------------------------------------------
"""
__author__ = 'li'

import numpy as np
import cv2

img = cv2.imread("1.jpg")
cv2.imshow("temp", img)
cv2.waitKey(0)

img90 = np.rot90(img)

cv2.imshow("rotate", img90)
cv2.waitKey(0)


