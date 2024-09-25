import os
import torch
from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
import supervision as sv
import time



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
#set device to gpu
torch.cuda.is_available()
torch.cuda.set_device(0)

if __name__ == '__main__':
    # Load the YOLOv8 model
    model = YOLO("runsyolov8shires/detect/train31/weights/best.pt")

    # Open the video file
    video_path = "clips/temp1.mp4"
    cap = cv2.VideoCapture(video_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Store the track history
    track_history = defaultdict(lambda: [])


    new_frame_time = 0
    prev_frame_time = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        #compute fps
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=.5)

            if results[0].boxes.id != None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy().astype(int)

                class_ids = results[0].boxes.cls.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)

                # get finished tracks and do some logic with them
                finished_tracks = track_history.keys() - track_ids
                for ft_id in finished_tracks:
                    ft = track_history.pop(ft_id)


                cv2.putText( annotated_frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA )
                # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()