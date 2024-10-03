import os
os.environ["WANDB_DISABLED"]="true"
from ultralytics import YOLO
from ultralytics import settings


#多线程添加代码
if __name__ == '__main__':
    # View all settings
    print(settings)
    # Update a setting
    settings.update({'runs_dir': './runs'})
    # Update multiple settings
    settings.update({'runs_dir': './runs', 'tensorboard': False})
    # Reset settings to default values
    # settings.reset()
    print(settings)

    # 加载一个模型 ,workers=8
    model = YOLO(r'ultralytics\cfg\models\v8\yolov8s.yaml')  # 从YAML建立一个新模型
    model.load('../ckpt/yolov8s.pt')

    # 训练模型
    # SGD, Adam, AdamW, NAdam, RAdam, RMSProp
    results = model.train(data='config/data.yaml',optimizer = "auto",
                      epochs=300, imgsz=640, device=0, workers=2, batch=8, cache=True,dropout=0.0)
