#-*-coding:utf-8-*-
# date:2024-10-02
# Author: DataBall - Xian
# function: webui detect
import sys
import os
import gradio as gr
import time
import cv2
import webbrowser
from app_utils import app_det_preprocess
from app_utils import app_det_train
from app_utils import app_det_inference

print("\n/************************** Welcome  ultralytics-yolo-webui : detect >> Data Ball  ****************************/")

def fun_login(text_):
    return text_ +" start learning "
with gr.Blocks(css="footer{display:none !important}").queue() as demo:

    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            gr.Image("./doc/TechLinkX.jpg",image_mode = "RGBA",container = False,show_fullscreen_button = False,
                show_label=False,show_download_button = False,width = 100,height=100)
        with gr.Column(scale=1, min_width=100):
            gr.Image("./doc/DataBall-log.png",image_mode = "RGBA",container = False,show_fullscreen_button = False,
                show_label=False,show_download_button = False,width = 100,height=100)
        with gr.Column(scale=5, min_width=100):
            gr.Markdown("# Ultralytics-Yolo-WebUI:Detect")

    app_det_preprocess.AppFunUI("Step1 数据预处理-Data Preprocess")
    app_det_train.AppFunUI("Step2 模型训练 YoLo Train")
    app_det_inference.AppFunUI("Step3 模型推理 YoLo Inference")

def webui():
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":

    loc_time = time.localtime()
    g_str_time = time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)
    print("------------- ultralytics-yolo-webui : detect >> Data Ball  ： {}".format(g_str_time))

    if not os.path.exists("./yolo_datasets"):
        os.mkdir("./yolo_datasets")
    if not os.path.exists("./yolo_label_text"):
        os.mkdir("./yolo_label_text")

    url = 'http://0.0.0.0:7860'
    webbrowser.open(url)
    webui()
