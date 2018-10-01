# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     seperate_image
   Description :
   Author :       'li'
   date：          2018/8/1
-------------------------------------------------
   Change Activity:
                   2018/8/1:
-------------------------------------------------
"""
import shutil

import cv2

from utility.file_path_utility import get_all_file_from_dir

__author__ = 'li'
"""
把不同形状的训练集做区分：水平的和垂直的。
"""
ORIGINAL_PATH = 'E:\dataset/text_area/text_area/'
HORIZONTAL_PATH = 'E:\dataset/text_area\horizontal/'
VERTICAL_PATH = 'E:\dataset/text_area/vertical/'

all_image_path = get_all_file_from_dir(ORIGINAL_PATH)
for path in all_image_path:
    try:
        print(path)
        if path.find('.jpg') <= 0:
            continue
        img = cv2.imread(path)
        if img is not None:
            shape = img.shape
            print(shape)
            image_name = path.split('\\')[-1]
            if shape[0] >= shape[1]:  # 高大于宽
                shutil.copy(path, VERTICAL_PATH + image_name)
                continue
            shutil.copy(path, HORIZONTAL_PATH + image_name)
    except Exception as e:
        print(e)
