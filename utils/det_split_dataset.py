#-*-coding:utf-8-*-
# date:2024-10-01
# Author: DataBall - Xian
# function: split make yolo dataset

import os
import shutil
import random
from tqdm import tqdm
random.seed(123)

def split_data(file_path,txt_path, new_file_path, train_rate=0.9, val_rate=0.05, test_rate=0.05):
    each_class_image = []
    each_class_label = []
    for image in os.listdir(file_path):
        if (".jpg" in image) or (".png" in image):
            each_class_image.append(image)
    for label in os.listdir(txt_path):
        if "txt" in label:
            each_class_label.append(label)
    data=list(zip(each_class_image,each_class_label))
    total = len(each_class_image)
    random.shuffle(data)
    each_class_image,each_class_label=zip(*data)
    train_images = each_class_image[0:int(train_rate * total)]
    val_images = each_class_image[int(train_rate * total):int((train_rate + val_rate) * total)]
    test_images = each_class_image[int((train_rate + val_rate) * total):]
    train_labels = each_class_label[0:int(train_rate * total)]
    val_labels = each_class_label[int(train_rate * total):int((train_rate + val_rate) * total)]
    test_labels = each_class_label[int((train_rate + val_rate) * total):]


    for image in tqdm(train_images,desc="make yolo train imgs"):
        # print(image)
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'train' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)

    for label in tqdm(train_labels,desc="make yolo train label"):
        # print(label)
        old_path = txt_path + '/' + label
        new_path1 = new_file_path + '/' + 'train' + '/' + 'labels'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + label
        shutil.copy(old_path, new_path)

    for image in tqdm(val_images,desc="make yolo val imgs"):
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'val' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)

    for label in tqdm(val_labels,desc="make yolo val label"):
        old_path = txt_path + '/' + label
        new_path1 = new_file_path + '/' + 'val' + '/' + 'labels'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + label
        shutil.copy(old_path, new_path)

    for image in tqdm(test_images,desc="make yolo test imgs"):
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'test' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)

    for label in tqdm(test_labels,desc="make yolo test label"):
        old_path = txt_path + '/' + label
        new_path1 = new_file_path + '/' + 'test' + '/' + 'labels'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + label
        shutil.copy(old_path, new_path)

if __name__ == '__main__':

    file_path = r"D:\wjx\dataset\img_list\chameleon"
    txt_path = r'D:\wjx\dataset\img_list\chameleon_label'
    new_file_path = r"./datas/"
    if not os.path.exists(new_file_path):
        os.mkdir(new_file_path)
    split_data(file_path,txt_path, new_file_path, train_rate=0.9, val_rate=0.05, test_rate=0.05)
