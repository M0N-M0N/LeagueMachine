import os
import torch
from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
#set device to gpu
torch.cuda.is_available()
torch.cuda.set_device(0)

if __name__ == '__main__':
    # VIDEOS_DIR = os.path.join('.', 'clips')
    #
    # video_path = os.path.join(VIDEOS_DIR, 'timingmoments.mp4')
    # video_path_out = '{}_out.mp4'.format(video_path)
    #
    # cap = cv2.VideoCapture(video_path)
    # ret, frame = cap.read()
    # H, W, _ = frame.shape
    # out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
    #
    # model_path = os.path.join('.', 'runs', 'detect', 'train6', 'weights', 'last.pt')
    #
    #
    # # Load a model
    # model = YOLO(model_path)  # load a custom model
    #
    # threshold = 0.7
    #
    # while ret:
    #
    #     #results = model(frame)
    #     results = model.track(frame[0], persist=True)
    #
    #
    #     for result in results.boxes.data.tolist():
    #         x1, y1, x2, y2, score, class_id = result
    #
    #         if score > threshold:
    #             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
    #             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    #
    #     out.write(frame)
    #     ret, frame = cap.read()
    #
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

    # Load the YOLOv8 model
    model = YOLO("runs/detect/train6/weights/last.pt")

    # Open the video file
    video_path = "clips/timingmoments.mp4"
    cap = cv2.VideoCapture(video_path)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

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
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                # get finished tracks and do some logic with them
                finished_tracks = track_history.keys() - track_ids
                for ft_id in finished_tracks:
                    ft = track_history.pop(ft_id)
                    # do some logic with ft here.........

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