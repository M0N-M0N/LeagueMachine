from ultralytics import YOLO
# from ultralytics import YOLOv10
import torch
import os

dirpath = os.path.dirname(__file__)
# open(filepath, 'r')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def train_model( train_type, num_iteration):
    #load yolo8 model
    # model = YOLO(os.path.join(dirpath, 'runs/detect/train2/weights/best.pt'))
    # model = YOLO(os.path.join(dirpath, load_weights))

    #prep num_iteration
    if num_iteration == 0:
        new_epoch = 600
        # load_weights = 'yolov8s.yaml'
        load_weights = 'yolov10s.yaml'
    else:
        train_iteration = "" if num_iteration == 1 else num_iteration
        new_epoch = 10 * num_iteration
        # new_epoch = 600 #* num_iteration
        load_weights = f"runs/detect/train{train_iteration}/weights/best.pt"

    #for YOLOv10
    # model = YOLOv10("yolov10s.yaml")
    model = YOLO(load_weights)
    model.resume = True

    filepath = os.path.join(dirpath, 'trainingsets/data.yaml')
    # print(load_weights)
    torch.cuda.is_available()

    #model
    #results = model.train(data="./leagueofegends.v1i.yolov8/data.yaml", epochs=50, imgsz=640, device=1)  #train model device 1 is gpu
    # model.save()
    # results = model.val()
    # results = model.eval()
    if train_type == "train":
        # results = model.train(data=filepath, epochs=new_epoch, device=1, workers=4, cache=False)
        #train till iteration 21 or consecutive early stoppage
        # results = model.train(data=filepath, epochs=new_epoch, device=1,batch=6, imgsz=[1280, 720], cache=False)
        #train till iteration 22
        # results = model.train(data=filepath, epochs=new_epoch, device=1,batch=5, imgsz=1280, optimizer='Adam', warmup_epochs=50, cache=False)
        results = model.train(data=filepath, epochs=new_epoch, device=1, batch=5, imgsz=1280, optimizer='AdamW', warmup_epochs=50, cache=False)
    elif train_type == "val":
        results = model.val()

    return results

if __name__ == '__main__':
    train_type = 'train'
    # train_model()
    
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.randn(1).cuda())
    # print(torch.cuda.get_device_properties(0))

    # i = 3
    # while i <= 3:
    #     torch.cuda.empty_cache()
    #     train_model(train_type, i)
    #     i += 1

    # train_model(train_type)




