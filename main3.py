from ultralytics import YOLO
import torch
import os

dirpath = os.path.dirname(__file__)
# open(filepath, 'r')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def train_model( num_iteration = "" ):
    #load yolo8 model
    # model = YOLO(os.path.join(dirpath, 'runs/detect/train2/weights/best.pt'))
    # load_weights = f"runs/detect/train{num_iteration}/weights/best.pt"
    load_weights = 'yolov9t.yaml'
    model = YOLO(load_weights)
    model.resume = True

    filepath = os.path.join(dirpath, 'trainingsets/data.yaml')
    # print(load_weights)
    torch.cuda.is_available()

    #model
    #results = model.train(data="./leagueofegends.v1i.yolov8/data.yaml", epochs=50, imgsz=640, device=1)  #train model device 1 is gpu
    results = model.train(data=filepath, epochs=1500, batch=2, device=1)
    # model.save()
    # results = model.val()
    # results = model.eval()


if __name__ == '__main__':

    train_model()
    #
    # i = 1
    # while i < 6:
    #     train_model(i)
    #     i += 1

