import cv2
from ultralytics import YOLO

#load yolo8 model
model = YOLO('yolov8n.pt')

#video
video_path = './clips/fakerclip.mp4';
cap = cv2.VideoCapture(video_path)

ret = True

#read it
while ret:
    ret, frame = cap.read()

    #find objects #track them
    results = model.track(frame, persist = True)
    H, W, _ = frame.shape

    #plot results
    frame_ = results[0].plot()

    #v
    cv2.imshow('frame', frame_)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break