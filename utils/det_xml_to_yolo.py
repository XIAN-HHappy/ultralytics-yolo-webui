#-*-coding:utf-8-*-
# date:2024-10-01
# Author: DataBall - Xian
# function: xml to yolo labels

import xml.etree.cElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from tqdm import tqdm

def convert(size, box):
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]
    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]
    return (x, y, w, h)

def convert_annotation(xml_files_path, save_txt_files_path, classes):
    xml_files = os.listdir(xml_files_path)

    idx = 0

    for xml_name in tqdm(xml_files,desc="xml to yolo"):

        if "xml" not in xml_name.split(".")[-1]:
            continue
        xml_file = os.path.join(xml_files_path, xml_name)
        out_txt_path = os.path.join(save_txt_files_path, xml_name.split('.')[0] + '.txt')
        out_txt_f = open(out_txt_path, 'w')
        idx += 1
        # print(" -> [{}] {}".format(idx,xml_file),end = "\r")
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            # b=(xmin, xmax, ymin, ymax)
            # print(w, h, b)
            bb = convert((w, h), b)
            out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == "__main__":
    # 需要转换的类别，需要一一对应
    classes_ = ['chameleon']
    # 2、voc格式的xml标签文件路径
    xml_files_ = r'D:\wjx\dataset\img_list\chameleon'
    # 3、转化为yolo格式的txt标签文件存储路径
    save_txt_files_ = r'D:\wjx\dataset\img_list\chameleon_label'
    if not os.path.exists(save_txt_files_):
        os.mkdir(save_txt_files_)
    convert_annotation(xml_files_, save_txt_files_, classes_)
