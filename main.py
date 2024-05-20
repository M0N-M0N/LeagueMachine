from ultralytics import YOLO
import torch
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

if __name__ == '__main__':
    #load yolo8 model
    model = YOLO('yolov8n.yaml')

    torch.cuda.is_available()

    #model
    #results = model.train(data="./leagueofegends.v1i.yolov8/data.yaml", epochs=50, imgsz=640, device=1)  #train model device 1 is gpu
    results = model.train(data="./trainingsets/data.yaml", epochs=100, device=1)