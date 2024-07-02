import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

from ultralytics.utils.files import increment_path



# Set CUDA devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

model_path="runs/detect/train31/weights/last.pt"
video_path = "clips/temp1.mp4"

def run(weights='runs/detect/train31/weights/last.pt', source='clips/temp1.mp4', view_img=False, save_img=False, exist_ok=False):
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source path {source} does not exists.")

    download_yolov8s_model(model_path)
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=0.3,
        device="cuda")

    cap = cv2.VideoCapture(source)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4)),
    fps, fourcc = int(cap.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    save_dir = increment_path("sahi/", exist_ok)
    save_dir.mkdir(parents = True, exist_ok=True)

    video_writer = cv2.VideoWriter(f"sahi_result.mp4", fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if not success:
            break

        results = get_sliced_prediction(
            frame, sahi_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        object_prediction_list = results.object_prediction_list

        boxes_list = []
        clss_list = []

        for ind, _ in enumerate(object_prediction_list):
            boxes =(
                object_prediction_list[ind].bbox.minx,
                object_prediction_list[ind].bbox.miny,
                object_prediction_list[ind].bbox.maxx,
                object_prediction_list[ind].bbox.maxy,
            )
            clss = object_prediction_list[ind].category.name
            boxes_list.append(boxes)
            clss_list.append(clss)

        for box, cls in zip(boxes_list, clss_list):
            x, y, w, h = box
            cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (56, 56, 255), 2)
            label = str(cls)
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]

            cv2.rectangle(
                frame, (int(x), int(y)-t_size[1]-3), (int(x)+t_size[0], int(y)+3), (56, 56, 255), -1
            )

            cv2.putText(
                frame, label, (int(x), int(y)-2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA
            )

        if view_img:
            cv2.imshow("YOLOv8 sahi", frame)
        if save_img:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run(model_path, video_path, save_img=True)