# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     resize_image
   Description :
   Author :       'li'
   date：          2018/8/1
-------------------------------------------------
   Change Activity:
                   2018/8/1:
-------------------------------------------------
"""
import cv2
import numpy as np

from utility import show_img
from utility.file_path_utility import get_all_file_from_dir
from utility.image_utility import save_img

__author__ = 'li'
"""
讲图片格式化为32高，100宽的大小
"""
HORIZONTAL_PATH = 'E:\dataset/text_area\horizontal/'


def get_resize_image(path):
    """
    获得重新定义过大小的img[32,100]
    输入图片需要时水平方向
    :param path:
    :return:
    """
    img = cv2.imread(path)
    shape = img.shape
    ratio = 32.0 / shape[0]
    new_w = int(ratio * shape[1])
    if new_w > 100:
        image = cv2.resize(img, (100, 32))
        return image
    bg = np.zeros((32, 100, 3), dtype=np.int)
    image = cv2.resize(img, (new_w, 32))
    bg[:, 0:new_w, :] = image
    return bg


def main():
    all_image_path = get_all_file_from_dir(HORIZONTAL_PATH)
    length = len(all_image_path)
    index = 0
    for path in all_image_path:
        index += 1
        if index % 100 == 0:
            print(str(index * 1.0 / length))
        new_img = get_resize_image(path)
        # print(path)
        save_img(new_img, path)


if __name__ == '__main__':
    main()
