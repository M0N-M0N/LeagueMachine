import os
import torch
from ultralytics import YOLO


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
#set device to gpu
torch.cuda.is_available()
torch.cuda.set_device(0)


if __name__ == '__main__':
    # Load the YOLOv8 model
    model = YOLO("runs/detect/train3/weights/last.pt")
    version = project.version(2)
    version.deploy("yolov8", "runs/detect/train4/weights/", "best.pt")