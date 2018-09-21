# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     text_generate
   Description :
   Author :       'li'
   date：          2018/8/1
-------------------------------------------------
   Change Activity:
                   2018/8/1:
-------------------------------------------------
"""
from utility.file_path_utility import get_all_file_from_dir

__author__ = 'li'
IMAGE_PATH = 'F:/dataset/container_dataset/text_area_vertical/'
image_paths = get_all_file_from_dir(IMAGE_PATH)


def change_label(label):
    """
    修改label
    :param label:
    :return:
    """
    first = label[0]
    if first != '#':
        return label
    num = label[1]
    return num
    # if num == '1':
    #     return '1'
    # elif num == '2':
    #     return '2'
    # elif num == '3':
    #     return '3'
    # elif num == '4':
    #     return '四'
    # elif num == '5':
    #     return '五'
    # elif num == '6':
    #     return '六'
    # elif num == '7':
    #     return '七'
    # elif num == '8':
    #     return '八'
    # elif num == '9':
    #     return '九'
    # elif num == '0':
    #     return '十'


with open('sample.txt', mode='w', encoding='gbk') as file:
    for path in image_paths:
        if path.find('.jpg') <= 0:
            continue
        l = path.split('\\')[-1].split('-')[0]
        label = change_label(l)
        line = path + ' ' + label + '\n'
        file.write(line)

print('写入完成')
with open('sample.txt', mode='r', encoding='gbk') as file:
    lines = file.readlines()
    labels = set()
    for line in lines:
        content = line.split(' ')
        label = content[1].replace('\n', '')
        if label.find('.') >= 0 or label.find('e') >= 0 or label.find('j') >= 0 or label.find('g') >= 0 or label.find(
                'l') >= 0 or label.find('b') >= 0:
            print(content[0])
        for c in label:
            labels.add(c)
    print(labels)
