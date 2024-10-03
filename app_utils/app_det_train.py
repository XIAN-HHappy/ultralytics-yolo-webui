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
from ultralytics import YOLO
from ultralytics import settings

def fun_retry():
    return gr.update(value="Train")

def fun_train(data_yaml,optimizer,finetune_model,model_yaml,epoch,imgsz,batch_size,workers,device_id):
    # View all settings
    # print(settings)
    runs_dir = os.getcwd().replace("\\","/")+"/runs"
    data_yaml = os.getcwd().replace("\\","/")+"/" + data_yaml
    print(" -> runs_dir:",runs_dir)
    print(" -> data_yaml:",data_yaml)
    print(" -> optimizer:",optimizer)
    print(" -> finetune_model:",finetune_model)
    print(" -> model_yaml:",model_yaml)
    print(" -> epoch:",epoch)
    print(" -> imgsz:",imgsz)
    print(" -> batch_size:",batch_size)
    print(" -> workers:",workers)
    print(" -> device_id:",device_id)
    # Update a setting
    settings.update({'runs_dir': runs_dir})
    # Update multiple settings
    settings.update({'runs_dir': runs_dir, 'tensorboard': False})
    # Reset settings to default values
    # settings.reset()
    print(settings)
    # 加载一个模型
    model = YOLO(model_yaml)  # 从YAML建立一个新模型
    model.load('./ckpt/{}'.format(finetune_model))
    # train : 训练模型
    # optimizer : auto, SGD, Adam, AdamW, NAdam, RAdam, RMSProp
    results = model.train(data=data_yaml,optimizer = optimizer,
                      epochs=int(epoch), imgsz=int(imgsz), device=int(device_id), workers=int(workers), batch=int(batch_size), cache=True,dropout=0.0)

    print(" -> train yolo done !")

    return gr.update(value="Done")

def AppFunUI(chapter_name = "***"):

    print("loading  {} ~ ".format(chapter_name))
    with gr.Tab(chapter_name):
        with gr.Accordion("Train YoLo"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            button_t = gr.Button("Train",size="lg",variant="primary")
                            button_tb = gr.Button("ReStart",size="lg",variant="primary")
                        with gr.Column(scale=6, min_width=100):
                            text_train_yaml = gr.Textbox(label = "Yaml_File",show_label = True,container =True,lines=1,value = "config/data.yaml",interactive = True)
                            text_optimizer = gr.Textbox(label = "Optimizer",show_label = True,container =True,lines=1,value = "auto",interactive = True)
                            text_finetune_model = gr.Textbox(label = "FineTune_Model",show_label = True,container =True,lines=1,value = "yolov8s.pt",interactive = True)
                            text_model_yaml = gr.Textbox(label = "FineTune_Model",show_label = True,container =True,lines=1,value = "yolov8s.yaml",interactive = True)
                            text_train_epoch = gr.Textbox(label = "Train_Epoch",show_label = True,container =True,lines=1,value = "100",interactive = True)
                            text_imgz = gr.Textbox(label = "Image_Size",show_label = True,container =True,lines=1,value = 640,interactive = True)
                            text_batch_size = gr.Textbox(label = "Batch_Size",show_label = True,container =True,lines=1,value = "8",interactive = True)
                            text_workers = gr.Textbox(label = "Workers",show_label = True,container =True,lines=1,value = "2",interactive = True)
                            text_device_id = gr.Textbox(label = "Device",show_label = True,container =True,lines=1,value = 0,interactive = True)

        button_t.click(fun_train, inputs=[text_train_yaml,text_optimizer,text_finetune_model,text_model_yaml,
            text_train_epoch,text_imgz,text_batch_size,text_workers,text_device_id],outputs=[button_t],show_progress = True)
        button_tb.click(fun_retry, inputs=[],outputs=[button_t],show_progress = True)
