import os
import torch
from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

import supervision as sv
from datetime import datetime
from PIL import Image

import pytesseract
import easyocr



import sys

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
#set device to gpu
torch.cuda.is_available()
torch.cuda.set_device(0)

reader = easyocr.Reader(['en'], detect_network='craft', gpu=True)

image_detection_res = "screenshots/detections"


if __name__ == '__main__':
    # Load the YOLO model
    model = YOLO("runsyolov8shires/detect/train31/weights/best.pt")

    # Open the  file
    video_path = "clips/smart33.mp4"
    image_path = "screenshots/77_jpg.rf.823ae2f740963b46dafded5a879e6557.jpg"

    # npimage_path = "screenshots/ally-champion_level.jpg"
    #cap = cv2.VideoCapture(video_path)

    byte_tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()
    is_imgshow = False

    cv_image = cv2.imread(image_path)
    # np_image = np.array(Image.open(npimage_path))
    # cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    ocr_img_names = ["player-champion", "kda", "game-time", "creep-kill", "player-gold", "ally-champion"]

    results = model(cv_image, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)
    labels = [
        f"#{detection[-2]} {model.model.names[detection[-3]]} {detection[-4]:0.2f}"
        if len(detection) >= 5 else "Invalid detection"
        for detection in detections
    ]

    # resframe =  trace_annotator.annotate(scene=resframe, detections=detections)

    #getting and cropping images within the bounding box for checking the content later with ocr

    # result = reader.readtext(cv_image)
    # print(result)
    resources = {}
    for i, det in enumerate(detections):
        cls_name = det[-1]["class_name"]
        #check if detected classes with necessary resources/metrics
        if det[-1]["class_name"] in ocr_img_names:
            # print(det)
            file_name = f"{image_detection_res}/{cls_name}_{det[-2]}.jpg"
            cropped_image = sv.crop_image(image=cv_image, xyxy=det[0])
            ocr_result = reader.readtext(
                cropped_image,
                allowlist ='0123456789.:/',
                mag_ratio=2.1,
                min_size=15,
            )  # easyocr

            for i2, ocr in enumerate(ocr_result):
                # print(ocr)
                if(ocr == []):
                    text = 'none'
                else:
                    text = ocr
                    if(text[-1] > 0.5): #check conf
                        resources.update({cls_name : ocr[-2]})
                        print(cls_name, ocr[-2], type(ocr))
                        # print(f"{cls_name}#{det[-2]}_level {text}")

            # np_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            # result = pytesseract.image_to_string(np_image)
            # cv2_img = np.array(np_image)
            # cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)

            # print(result)
            cv2.imwrite(file_name, cropped_image)

        resframe = box_annotator.annotate(scene=cv_image.copy(), detections=detections)
        resframe = label_annotator.annotate(scene=resframe, detections=detections, labels=labels)

    print(resources)




    cv2.imshow("ocr", resframe)
    cv2.waitKey(0)
    cv2.destroyAllWindows()