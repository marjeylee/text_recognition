# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     training_data_provider
   Description :
   Author :       'li'
   date：          2018/9/9
-------------------------------------------------
   Change Activity:
                   2018/9/9:
-------------------------------------------------
"""
import os

import cv2

from utility.file_path_utility import get_all_file_from_dir
import numpy as np
import tensorflow as tf

__author__ = 'li'
"""
provide training images and labels
"""
char_mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'A': 10,
    'B': 11,
    'C': 12,
    'D': 13,
    'E': 14,
    'F': 15,
    'G': 16,
    'H': 17,
    'I': 18,
    'J': 19,
    'K': 20,
    'L': 21,
    'M': 22,
    'N': 23,
    'O': 24,
    'P': 25,
    'Q': 26,
    'R': 27,
    'S': 28,
    'T': 29,
    'U': 30,
    'V': 31,
    'W': 32,
    'X': 33,
    'Y': 34,
    'Z': 35,
}

# IMAGE_PATH = 'E:\dataset/text_area/vertical'
IMAGE_PATH = '/gpu_data/back/images/test_area/vertical_classifys/'

def load_all_data():
    files_paths = get_all_file_from_dir(IMAGE_PATH)
    print(('training images size:' + str(len(files_paths))))
    print('start load training data')
    data = {'one': {}, 'four': {}, 'six': {}, 'zero': []}
    for p in files_paths:
        img = cv2.imread(p)
        shape = img.shape
        img = img.reshape((1, shape[0], shape[1], shape[2]))
        _, image_name = os.path.split(p)
        label = image_name.replace('#', '').split('-')[0]
        if len(label) == 1:
            if label not in data['one'].keys():
                data['one'][label] = []
            data['one'][label].append({'label': label, 'img': img})
        elif len(label) == 4:
            if label not in data['four'].keys():
                data['four'][label] = []
            data['four'][label].append({'label': label, 'img': img})
        elif len(label) == 6:
            if label not in data['six'].keys():
                data['six'][label] = []
            data['six'][label].append({'label': label, 'img': img})
        else:
            data['zero'].append({'label': label, 'img': img})
    print('loading data finish')
    return data


training_data = load_all_data()
ones = training_data['one']
ones_keys = list(ones.keys())
fours = training_data['four']
fours_keys = list(fours.keys())
sixs = training_data['six']
sixs_keys = list(sixs.keys())


def load_training_images(random_images_path):
    """
    load images
    :param random_images_path:
    :return:
    """
    assert len(random_images_path) > 1
    training_images = []
    for p in random_images_path:
        img = cv2.imread(p)
        shape = img.shape
        img = img.reshape((1, shape[0], shape[1], shape[2]))
        training_images.append(img)
    return np.concatenate(training_images, axis=0)


def load_training_labels(random_images_path):
    """
    load labels
    :param random_images_path:
    :return:
    """
    labels = []
    max_size = 0
    for p in random_images_path:
        code_label = []
        _, image_name = os.path.split(p)
        l = str(image_name).split('-')[0].replace('#', '')
        label_length = len(l)
        if max_size < label_length:
            max_size = label_length
        for c in l:
            code = char_mapping[c]
            code_label.append(code)
        labels.append(code_label)
    indices = []
    values = []
    size = [len(labels), max_size]
    for x in range(len(labels)):
        for y in range(len(labels[x])):
            indices.append([x, y])
            values.append(labels[x][y])
    ten = tf.SparseTensorValue(indices, values, size)
    return ten


def get_training_image(total):
    imgs = []
    for item in total:
        img = item['img']
        imgs.append(img)
    return np.concatenate(imgs, axis=0)


def get_training_label(total):
    labels = []
    max_size = 0
    for item in total:
        label = item['label']
        label = str(label).upper()
        code_label = []
        label_length = len(label)
        if max_size < label_length:
            max_size = label_length
        for c in label:
            code = char_mapping[c]
            code_label.append(code)
        labels.append(code_label)
    indices = []
    values = []
    size = [len(labels), max_size]
    for x in range(len(labels)):
        for y in range(len(labels[x])):
            indices.append([x, y])
            values.append(labels[x][y])
    ten = tf.SparseTensorValue(indices, values, size)
    return ten


def get_training_data():
    """
    get_trainingdata  32
    :return:
    """
    '''one'''
    random_ones_keys = np.random.choice(ones_keys, size=8, replace=False)
    ones_objs = []
    for key in random_ones_keys:
        if not str(key).isdigit():
            key = np.random.choice(ones_keys, size=1)[0]  # if key is alph ,choice again
        obj = np.random.choice(ones[key], size=1, replace=False).tolist()[0]
        ones_objs.append(obj)
    '''four'''
    random_fours_keys = np.random.choice(fours_keys, size=13, replace=False)
    fours_objs = []
    for key in random_fours_keys:
        obj = np.random.choice(fours[key], size=1, replace=False).tolist()[0]
        fours_objs.append(obj)
    '''six'''
    random_sixs_keys = np.random.choice(sixs_keys, size=11, replace=False)
    sixs_objs = []
    for key in random_sixs_keys:
        obj = np.random.choice(sixs[key], size=1, replace=False).tolist()[0]
        sixs_objs.append(obj)
    '''else'''
    # zero = training_data['zero']
    # zero_training_data = np.random.choice(zero, size=1).tolist()
    total = ones_objs + fours_objs + sixs_objs  # + zero_training_data
    training_img = get_training_image(total)
    training_labels = get_training_label(total)
    return training_img, training_labels


def main():
    get_training_data()


if __name__ == '__main__':
    main()
