# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_validate
   Description :
   Author :       'li'
   date：          2018/9/9
-------------------------------------------------
   Change Activity:
                   2018/9/9:
-------------------------------------------------
"""
import os

from utility.file_path_utility import get_all_file_from_dir

__author__ = 'li'
images_path = 'F:\dataset\horizontal'
paths = get_all_file_from_dir(images_path)
for p in paths:
    _, image_name = os.path.split(p)
    label = image_name.split('-')[0]
    for c in label:
        if not ('1234567890'.find(c) >= 0 or 'QWERTYUIOPLKJHGFDSAZXCVBNM'.find(c) >= 0 or '#'.find(c) >= 0):
            print(p)
