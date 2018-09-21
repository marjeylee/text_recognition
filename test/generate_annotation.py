"""
生成注解文件
"""
import os

data_path = 'F:/code/orc/dataset/4_alpha/'


def get_file_name(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        file_list.extend(files)
    return file_list


file_names = get_file_name(data_path)
with open('sameple.txt', mode='w', encoding='utf8') as file:
    for name in file_names:
        label = name[0:4]
        line = data_path + name + ' ' + label + '\n'
        file.write(line)
