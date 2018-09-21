# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     delete_file
   Description :
   Author :       'li'
   date：          2018/8/20
-------------------------------------------------
   Change Activity:
                   2018/8/20:
-------------------------------------------------
"""
import os

from utility.file_path_utility import get_all_file_from_dir

__author__ = 'li'

if __name__ == '__main__':
    # path = 'F:\dataset\container_dataset/text_area_vertical'
    path = 'F:\dataset\container_dataset/text_area_horizontal'
    paths = get_all_file_from_dir(path)
    for p in paths:
        os.remove(p)
