#-*-coding:utf-8-*-
# date:2024-10-01
# Author: DataBall - Xian
# function: yolo data preprocess

import os
import gradio as gr
import time
import numpy as np
import time
from utils.det_xml_to_yolo import convert_annotation
from utils.det_split_dataset import split_data

def fun_retry():
    return gr.update(value="Xml_To_Txt")
def fun_retry2():
    return gr.update(value="Make_Yolo_DataSet")

def fun_xml2txt(path_xml,path_yolo_txt,text_yolo_label):
    print("path_xml     :{}".format(path_xml))
    print("path_yolo_txt:{}".format(path_yolo_txt))
    #-------------------------------------------------
    xml_files_ = r"{}".format(path_xml)
    # xml 转化为 yolo 格式的txt标签文件存储路径
    save_txt_files_ = r"{}".format(path_yolo_txt)
    if not os.path.exists(save_txt_files_):
        os.mkdir(save_txt_files_)
    classes_ = text_yolo_label.strip().split(" ")
    cls_num = len(classes_)
    print("classes:{}".format(classes_))
    convert_annotation(xml_files_, save_txt_files_, classes_)

    print("-> xml2txt done !")
    return path_yolo_txt,text_yolo_label,cls_num,gr.update(value="Done")

def fun_split_data(file_path,txt_path,new_file_path,text_yaml,text_yolo_label,cls_num):
    if not os.path.exists(new_file_path):
        os.mkdir(new_file_path)
    text_yaml += "/"
    if not os.path.exists(text_yaml):
        os.mkdir(text_yaml)
    split_data(file_path,txt_path, new_file_path, train_rate=0.9, val_rate=0.05, test_rate=0.05)
    #
    classes_ = text_yolo_label.strip().split(" ")
    #
    path_root = os.getcwd().replace("\\","/") + "/"
    str_input = 'train: {}/train/images'.format(path_root + new_file_path) + "\n"\
        'val: {}/train/images'.format(path_root + new_file_path)  + "\n"\
        'test: {}/train/images'.format(path_root + new_file_path) + "\n"\
        'nc: {}'.format(cls_num)+ "\n"\
        "# Classes"+ "\n"\
        "names: {}".format(str(classes_))

    with open(text_yaml + '/data.yaml', 'w',encoding='utf-8') as file:
        file.write(str_input)

    # print( text_yolo_label.strip().split(" "))
    print("text_yolo_label:",text_yolo_label)

    return gr.update(value="Done")

def AppFunUI(chapter_name = "***"):

    print("loading  {} ~ ".format(chapter_name))
    with gr.Tab(chapter_name):
        with gr.Accordion("Step 1-1: Xml to Yolo_txt"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            button_1 = gr.Button("Xml_To_Txt",size="lg",variant="primary")
                            button_1b = gr.Button("ReStart",size="lg",variant="primary")
                        with gr.Column(scale=6, min_width=100):
                            text_xml= gr.Textbox(label = "Xml_Path",show_label = True,container =True,lines=1,placeholder="xml path")
                            text_yolo_label = gr.Textbox(label = "Class_Name",show_label = True,container =True,lines=1,
                                placeholder="class_name : Spaces distinguish categories")
                            text_yolo_txt = gr.Textbox(label = "YoLo_Txt_Path",show_label = True,container =True,lines=1,value = "yolo_label_text")

        with gr.Accordion("Step 1-2: Make YoLo Train|Val|Test Dataset"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            button_2 = gr.Button("Make_Yolo_DataSet",size="lg",variant="primary")
                            button_2b = gr.Button("ReStart",size="lg",variant="primary")
                        with gr.Column(scale=6, min_width=100):
                            text_img = gr.Textbox(label = "Images_Path",show_label = True,container =True,lines=1,placeholder="image path")
                            text_yolo_data = gr.Textbox(label = "YoLo_Train|Val|Test_Dataset_Path ",show_label = True,container =True,lines=1,value = "yolo_datasets")
                            text_yaml = gr.Textbox(label = "Yaml_Path",show_label = True,container =True,lines=1,value = "./config")
                            text_txt= gr.Textbox(label = "YoLo_Txt",show_label = True,container =True,lines=1,placeholder="yolo txt according to step 1-1",interactive = False)
                            text_cls = gr.Textbox(label = "Class_Name",show_label = True,container =True,lines=1,placeholder="class_name according to step 1-1",interactive = False)
                            text_cls_num = gr.Textbox(label = "Class_Number",show_label = True,container =True,lines=1,placeholder="class_number according to step 1-1",interactive = False)

        button_1.click(fun_xml2txt, inputs=[text_xml,text_yolo_txt,text_yolo_label],outputs=[text_txt,text_cls,text_cls_num,button_1],show_progress = True)
        button_1b.click(fun_retry, inputs=[],outputs=[button_1],show_progress = True)

        button_2.click(fun_split_data, inputs=[text_img,text_txt,text_yolo_data,text_yaml,text_cls,text_cls_num],outputs=[button_2],show_progress = True)
        button_2b.click(fun_retry2, inputs=[],outputs=[button_2],show_progress = True)
