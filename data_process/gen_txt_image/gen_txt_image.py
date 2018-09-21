# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gen_txt_image
   Description :
   Author :       'li'
   date：          2018/8/3
-------------------------------------------------
   Change Activity:
                   2018/8/3:
-------------------------------------------------
"""
import json
import os
import uuid

import cv2

from utility import show_img
from utility.file_io_utility import read_all_content
from utility.image_utility import save_img

__author__ = 'li'
"""
通过读取定位出来的图片，进行切分
"""
DETECTION_PATH = 'C:\\Users\\mar\\Desktop\\results\\'
TEXT_AREA = 'C:\\Users\\mar\Desktop\\text_area\\'


def get_result_list(DETECTION_PATH):
    """
    获取结果图片，以及文字区域路径
    :param DETECTION_PATH:
    :return:
    """
    arr_list = []
    for root, dirs, files in os.walk(DETECTION_PATH, topdown=False):
        obj = {}
        for name in files:
            if name.find('input') >= 0:
                file_name = os.path.join(root, name)
                obj['image_path'] = file_name
            elif name.find('json') > 0:
                file_name = os.path.join(root, name)
                json_str = read_all_content(file_name)
                json_obj = json.loads(json_str)
                obj['info'] = json_obj
        arr_list.append(obj)
    return arr_list


def get_text_area(img, area):
    """
    获取文字区域
    :param img:
    :param area:
    :return:
    """
    x0 = int(area['x0'])
    y0 = int(area['y0'])
    x2 = int(area['x2'])
    y2 = int(area['y2'])
    # if x2 - x0 < y2 - y0:  # 如果是垂直方向的,高度增加
    #     y0 = y0 - 5
    #     y2 = y2 + 6
    text_image = img[y0:y2, x0:x2, :]
    return text_image


def save_text_image(arr_list):
    """
    保存文字区域图片
    :param arr_list:
    :return:
    """
    i = 0
    for content in arr_list:
        try:
            image_path = content['image_path']
            info = content['info']
            text_lines = info['text_lines']
            if len(text_lines) > 0:
                for area in text_lines:
                    img = cv2.imread(image_path)
                    text_area = get_text_area(img, area)
                    save_img(text_area, TEXT_AREA + str(i) + '.jpg')
                    i += 1
        except Exception as e:
            print(e)


def main():
    # 读取解析result
    arr_list = get_result_list(DETECTION_PATH)
    save_text_image(arr_list)


if __name__ == '__main__':
    main()
