# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     fornat_label
   Description :
   Author :       'li'
   date：          2018/9/28
-------------------------------------------------
   Change Activity:
                   2018/9/28:
-------------------------------------------------
"""
import shutil
import uuid

from utility.file_path_utility import get_all_files_under_directory

ORIGINAL_DIR_PATH = 'E:\dataset/text_area/111111111111111111111111\six/'
DES_PATH = 'E:\dataset/text_area/111111111111111111111111/res/'


def get_label(p):
    """
    :param p:
    :return:
    """
    p = p.replace('[', '')
    p = p.replace(']', '')
    for i in range(5000):
        blank = '(' + str(i) + ')'
        p = p.replace(blank, '')
    label = p.split('-')[-1].split('.')[0]
    return label


def main():
    paths = get_all_files_under_directory(ORIGINAL_DIR_PATH)
    for p in paths:
        label = get_label(p)
        print(label)
        new_path = DES_PATH + label + '-' + str(uuid.uuid4()) + '.jpg'
        shutil.copy(p, new_path)


if __name__ == '__main__':
    main()
