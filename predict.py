import os
import torch
from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

import supervision as sv


import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
#set device to gpu
torch.cuda.is_available()
torch.cuda.set_device(0)

if __name__ == '__main__':
    # Load the YOLOv8 model
    model = YOLO("runs/detect/train21/weights/last.pt")

    # Open the video file
    video_path = "clips/timingmoments.mp4"
    #cap = cv2.VideoCapture(video_path)

    byte_tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()


    def process_frame(frame: np.ndarray, index: int) -> np.ndarray:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        # print(len(detections))
        detections = byte_tracker.update_with_detections(detections)
        # print(len(detections))
        # print(detections)
        # sys.exit()

        labels = [
            f"#{detection[-2]} {model.model.names[detection[-3]]} {detection[-4]:0.2f}"
            if len(detection) >= 5 else "Invalid detection"
            for detection in detections
        ]

        resframe = box_annotator.annotate(scene=frame.copy(), detections=detections)
        resframe = label_annotator.annotate(scene=resframe, detections=detections, labels=labels)

        return trace_annotator.annotate(scene=resframe, detections=detections)

    sv.process_video(source_path=video_path, target_path=f"result.mp4", callback=process_frame)
