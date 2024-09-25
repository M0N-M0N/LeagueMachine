import os
import sys

import cv2
import getvid4dataset as getvid


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def screenshot_maker(vid_path, capt_start = 0 ):
    screenshot_path = "screenshots/"
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    screenshot_interval = int( fps * 5)

    image_captured = capt_start

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if not success:
            break

        #get screenshots every 15 seconds
        if image_captured % screenshot_interval == 0:
            #save frame as screenshot
            screenshot_name = f"{int(image_captured/screenshot_interval)}.jpg"
            is_saved = cv2.imwrite(os.path.join(screenshot_path, screenshot_name), frame)
            if is_saved:
                print(f"Screenshot saved at {os.path.join(screenshot_path, screenshot_name)}")
        image_captured += 1

    cap.release()
    return image_captured

def get_vids(vid_path):
    video_extensions = [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"]
    video_files = []

    for root, dirs, files in os.walk(vid_path):
        for file in files:
            if file.lower().endswith(tuple(video_extensions)):
                video_files.append(os.path.join(root, file))

    return video_files

if __name__ == '__main__':
    # Open the video file
    video_path = "datasetvids/"
    cap_start = 0

    playlist_url = "https://www.youtube.com/playlist?list=PLwg1qJ9kcSUKlkGYLDbw_IQPvub8rV4U9"
    output_folder = "datasetvids/"

    #youtube vids
    vids_got = getvid.download_youtube_playlist(playlist_url, output_folder)

    if not vids_got:
        sys.exit()


    #screenshots
    # vids = get_vids(video_path)
    # if vids:
    #     for vid in vids:
    #         print(vid)
    #         cap_start = screenshot_maker(vid, cap_start)
    # else:
    #     print("No videos found")
    #




