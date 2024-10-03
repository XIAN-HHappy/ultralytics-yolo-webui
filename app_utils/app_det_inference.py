#-*-coding:utf-8-*-
# date:2024-10-01
# Author: DataBall - Xian
# function: train yolo

import os
os.environ["WANDB_DISABLED"]="true"
import gradio as gr
import time
import numpy as np
import time
import cv2
from ultralytics import YOLO
from ultralytics import settings
import supervision as sv


def fun_retry():
    return gr.update(value="Train")


def fun_load_ckpt(ckpt_path):
    global m_model
    ckpt_path = ckpt_path.replace("\\","/")
    print(" -> load ckpt_path:",ckpt_path)
    m_model = YOLO(ckpt_path)

    return "load successful : " + ckpt_path
def fun_inference(img,imgsz,conf):
    global m_model
    box_annotator = sv.BoxAnnotator()
    imgsz = int(imgsz)
    conf = float(conf)
    # imm = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(" -> m_model.names:",m_model.names)
    print(" -> imgsz:",imgsz,type(imgsz))
    print(" -> confidence:",conf,type(conf))
    print("img:",img.shape)
    results = m_model.predict(img,
        project='./',       # 保存预测结果的根目录
        name='exp',         # 保存预测结果目录名称
        exist_ok=True,
        save=False,
        imgsz=imgsz,          # 推理模型输入图像尺寸
        conf=conf            # 置信度阈值
        )
    # print("results:",results)
    for result in results:
        boxes = result.boxes.xyxy.cpu().detach().numpy()  # 获取检测目标边界框
        conf = result.boxes.conf.cpu().detach().numpy()  # 获取检测目标置信度
        cls = result.boxes.cls.cpu().detach().numpy() # 获取检测目标标签
        print(" -> boxes ,conf,cls:",boxes.shape ,conf.shape,cls.shape,conf)

        for i in range(boxes.shape[0]):
            box_ = boxes[i].reshape(-1,4)
            det_ = sv.Detections(xyxy=box_)
            conf_ =conf[i]
            img = box_annotator.annotate(scene=img, detections=det_, labels=[m_model.names[cls[i].reshape(-1)[0]] + " {:.2f}".format(conf_)])
    return img
def AppFunUI(chapter_name = "***"):

    global m_model

    print("loading  {} ~ ".format(chapter_name))
    target_size = [480,480]
    with gr.Tab(chapter_name):
        with gr.Accordion("Inference YoLo"):
            with gr.Row():
                with gr.Column():
                    text_ckpt_path = gr.Textbox(label = "Ckpt_File",show_label = True,container =True,lines=1,value = "ckpt\yolov8s.pt",interactive = True)
            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    with gr.Row():
                        text_size = gr.Textbox(label = "Imgsz",show_label = True,container =True,lines=1,value = 640,interactive = True)
                    with gr.Row():
                        text_confidence = gr.Textbox(label = "Confidence",show_label = True,container =True,lines=1,value = 0.3,interactive = True)
                with gr.Column(scale=3, min_width=100):
                        img_input = gr.Image(show_label = True,label = "input",type='numpy',
                                            image_mode = "RGB",height= target_size[0],width = target_size[1])
                with gr.Column(scale=3, min_width=100):
                        img_output = gr.Image(show_label = True,label = "output",type='numpy',
                                            image_mode = "RGB",height= target_size[0],width = target_size[1])
            with gr.Row():
                with gr.Column():
                    text_ckpt_now = gr.Textbox(label = "Inference_Model",show_label = True,container =True,lines=1,value = "None",interactive = False)
            with gr.Row():
                with gr.Column():
                    button_inf = gr.Button("Inference",size="lg",variant="primary")
                with gr.Column():
                    button_load_model = gr.Button("Load Model",size="lg",variant="primary")

        button_load_model.click(fun_load_ckpt, inputs=[text_ckpt_path],outputs=[text_ckpt_now],show_progress = True)
        button_inf.click(fun_inference, inputs=[img_input,text_size,text_confidence],outputs=[img_output],show_progress = True)
