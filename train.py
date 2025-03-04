import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-GhostDynamicConv.yaml.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='D:/yolo11/ultralytics-yolo11-20241015/ultralytics-yolo11-main/dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=0,
                workers=0,
                # device='0',
                optimizer='SGD',
                project='runs/train',
                name='test',
                )
