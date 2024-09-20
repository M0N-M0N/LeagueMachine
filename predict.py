import os
import pprint
import re
from textwrap import wrap
import timeit

from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

import supervision as sv
from datetime import datetime

import llama38b as llm
import easyocr


import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
#set device to gpu
torch.cuda.is_available()
torch.cuda.set_device(0)




#from stasck overflow https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
def toImgOpenCV(imgPIL): # Conver imgPIL to imgOpenCV
    i = np.array(imgPIL) # After mapping from PIL to numpy : [R,G,B,A]
    # numpy Image Channel system: [B,G,R,A]
    red = i[:,:,0].copy(); i[:,:,0] = i[:,:,2].copy(); i[:,:,2] = red;
    return i;

def toImgPIL(imgOpenCV): return Image.fromarray(cv2.cvtColor(imgOpenCV, cv2.COLOR_BGR2RGB));


def add_text_bubble(texts, font_size=24, max_width=400):
    # print(text)
    # Open the image
    # img = convert_from_cv2_to_image(img_path)
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size=font_size)

    # find text size
    text_bbox = draw.textbbox((0, 0), texts, font=font)

    # find bubble size
    bubble_width = min(max_width, text_bbox[2] - text_bbox[0] + 20)
    # Add padding
    bubble_height = text_bbox[3] - text_bbox[1] + 20

    # text position by anchor
    x, y, y2 = 40,20, 0

    # Draw the text bubble
    # draw.rectangle([x, y, x + bubble_width, y + bubble_height], fill=(0, 0, 0))  # Dark background
    draw.rectangle([x, y, x + bubble_width, y + bubble_height], width=0)
    # draw.text((x + 10, y + 10), text, fill=(255, 255, 255), font=font)
    draw.text((x + 10, y + 10), texts, font=font)

    # print(len(texts), texts)
    # try:
    #     analysis, advice = map(str, texts.split("Tips to improve:"))
    #     texts_lines = wrap(analysis, 170, replace_whitespace=True, fix_sentence_endings=True, drop_whitespace=True)
    #     for i, line in enumerate(texts_lines):
    #         line=line.replace("               ", "\n")
    #         print(i, line)
    #         draw.text((x + 10, y + y2), line, font=font)
    #         y2 += 30
    #     draw.text((x + 10, y + y2 + 40), "Tips to improve:", font=font)
    #     draw.text((x + 10, y + y2 + 20), advice, font=font)
    #
    # except ValueError:
    #     print(ValueError)
    #     texts_lines = wrap(texts, 170, replace_whitespace=True, fix_sentence_endings=True, drop_whitespace=True)
    #     for i, line in enumerate(texts_lines):
    #         line=line.replace("               ", "\n")
    #         print(i, line)
    #         y2 += 20
    #         draw.text((x + 10, y + y2), line.replace("               ", "\n"), font=font)
    #         draw.text((x + 10, y + y2 + 20), line, font=font)
    #     draw.text((x + 10, y + y2 + 20), texts, font=font)
    # print(texts_lines)


    # img_np = np.array(img)

    return toImgOpenCV(img)

def process_frame(frame: np.ndarray, index: int) -> np.ndarray:
    global is_frame, text_bubble, advice_timer, show_advice, resource_dict, frame_interval, position
    is_analysed = False
    # index = 900
    if index%frame_interval == 0 and index != 0:
        # print(index)
        is_frame = True

    if is_frame:
        # print("is_frame", index)
        resframe = frame.copy()
        results = model(frame, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(results)
        # print(len(detections))
        detections = byte_tracker.update_with_detections(detections)
        # print(len(detections))
        # print(detections)
        # sys.exit()

        # np_image = np.array(Image.open(npimage_path))
        # cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        labels = [
            f"#{detection[-2]} {model.model.names[detection[-3]]} {detection[-4]:0.2f}"
            if len(detection) >= 5 else "Invalid detection"
            for detection in detections
        ]

        resframe = box_annotator.annotate(scene=resframe, detections=detections)
        resframe = label_annotator.annotate(scene=resframe, detections=detections, labels=labels)
        resframe = trace_annotator.annotate(scene=resframe, detections=detections)
        # print("process_frame", resources)
        resources = read_resources(resframe, detections)

        for resource in resources:
            if resource in resource_dict.keys():
                if re.match(kda_pattern, resources[resource]) and resource == "kda":
                    resource_dict.update({"kda": resources[resource]})
                if re.match(time_pattern, resources[resource]) and resource == "game-time":
                    resource_dict.update({"game-time": resources[resource]})
                if resource == "creep-kill":
                    resource_dict.update({"creep-kill": resources[resource]})


        # resource_dict = {"kda": "7/7/7", "game-time": "10:20", "creep-kill": "30"}
        if all(resource_dict.values()):
            print(resource_dict)
            is_analysed, text_bubble = add_analysis(resource_dict, position)
            resource_dict = dict.fromkeys(resource_dict, None)

        # print(is_analysed)
        if is_analysed:
            is_frame = False
            analyzed_frame = sv.draw_image(resframe, text_bubble, .90, rect)
            show_advice = True
            return analyzed_frame
        else:
            is_frame = True
            return resframe
    else:
        if show_advice == True:
            if advice_timer<300:
                analyzed_frame = sv.draw_image(frame, text_bubble, .90, rect)
                advice_timer += 1
                return analyzed_frame
            else:
                advice_timer = 0
                show_advice = False

        return frame

def read_resources(frame, detections):
    resources = {}
    for i, det in enumerate(detections):
        cls_name = det[-1]["class_name"]
        #check if detected classes with necessary resources/metrics
        if det[-1]["class_name"] in ocr_img_names:
            # print("detections", det[-1]["class_name"])
            cropped_image = sv.crop_image(image=frame, xyxy=det[0])
            ocr_result = reader.readtext(
                cropped_image,
                allowlist ='0123456789.:/',
                mag_ratio=2.1,
                min_size=13,
            )  # easyocr

            # print("ocr_result", ocr_result)
            for i2, ocr in enumerate(ocr_result):
                if not ocr:
                    text = 'none'
                else:
                    text = ocr
                    if(text[-1] > 0.5): #check conf
                        resources.update({cls_name : ocr[-2]})
                        # print(cls_name, ocr[-2], type(ocr))
                        # print(f"{cls_name}#{det[-2]}_level {text}")
    return resources

def add_analysis(resources, position="adc" ):
    kda = resources['kda']
    game_time = resources['game-time']

    if "." in game_time:
        mins, seconds = map(int, game_time.split("."))
    else:
        mins, seconds = map(int, game_time.split(":"))

    # print(resources)
    # kda = format_kda(kda)

    # question = (f"Im a position {position} with kills/deaths/assists of {kda} and {resources['creep-kill']} cs within "
    #             f"a game-time of {mins} minutes, "
    #             f"compare it to the average kills/deaths/assists and cs of pro given the same Position")

    question = (f"game-time: {mins} minutes,"
                f"Position: {position},"
                f"kills/deaths/assists: {kda},"
                f"cs: {resources['creep-kill']},"
                f"Compare them to averages of their equivalent with the same position within similar game-time")
    retriever_question = f"find from data game-time: {mins} minutes and position_{position}"

    answer = myLLm.generate(question, retriever_question)
    # answer = """To improve, you should focus on killing minions, taking objectives, and getting kills. With a kda of 0/0/0, it's clear you're not doing well in the early game, so let's look at the data.
    #
    # The best mid laners in the world have an average kda of around 2/1/3 in the first 3 minutes. This is much higher than your current 0/0/0. They also tend to have a lot of cs, around 20-30, in the same time frame.
    #
    # To get closer to their level, you should focus on the following:
    #
    # 1. Last-hit minions to get gold and cs. This will help you get a better economy and a stronger early game.
    # 2. Take objectives, such as the first tower, to get an advantage and put pressure on the enemy team.
    # 3. Try to get a kill or two in the early game to get a lead and demoralize the enemy team.
    # 4. Focus on roaming and helping your team in the early game to set up a strong foundation for the rest of the game.
    #
    # # Remember, the early game is crucial in setting up the rest of the game, so make sure you're doing well in this period to have a good chance of winning"""
    # lines = answer.split(".")
    # line_breaks = 0
    # pprint.pprint(text)
    text_bubble = add_text_bubble(answer, 20, max_width=400)

    return True, text_bubble

def format_kda(input_str):
    # format kda for prompt
    parts = input_str.split('/')

    k, d, a = map(int, parts)

    kills = str(k % 100).zfill(2)
    deaths = str(d % 100).zfill(2)
    assists = str(a % 100).zfill(2)

    formatted_date = f"{kills}/{deaths}/{assists}"

    return formatted_date

import tkinter as tk
from tkinter import filedialog, ttk

class LeagueMachineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("League Machine")
        self.root.geometry("900x350")
        self.root.resizable(0, 0)  # not resizable
        self.root.configure(bg='black')
        self.assets = "assets/img"
        self.file_path = ""
        # root.wm_attributes("-transparentcolor", 'grey')

        self.bg_image = tk.PhotoImage(file=f"{self.assets}/league_machine_bg.png")
        self.bg_label = tk.Label(root, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1)

        self.title_label = tk.Label(root, text="League Machine", font=("Helvetica", 24), bg='black', fg='white')
        self.title_label.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Video", command=self.upload_video, bg='gray', fg='white')
        self.upload_button.pack(pady=10)

        self.position_label = tk.Label(root, text="Position", bg='black', fg='white')
        self.position_label.pack(pady=5)
        self.position_var = tk.StringVar()
        self.position_dropdown = ttk.Combobox(root, textvariable=self.position_var)
        self.position_dropdown['values'] = ("adc", "top", "midlane", "support")
        self.position_dropdown.pack(pady=5)

        self.analyze_button = tk.Button(root, text="Analyze", command=self.analyze, bg='gray', fg='white')
        self.analyze_button.pack(pady=20)

    def upload_video(self):
        self.file_path = filedialog.askopenfilename()
        print(f"Video uploaded: {self.file_path}")

    # video prediction
    def analyze(self):
        position = self.position_var.get()
        print(f"Analyzing for position: {position} with filepath {self.file_path}")
        current_dateTime = datetime.now()
        print("start Process Video", is_frame, current_dateTime)
        sv.process_video(source_path=video_path, target_path=f"vtemp.mp4", callback=process_frame)
        end_dateTime = datetime.now()
        proc_time = end_dateTime - current_dateTime
        print("End Process Video", is_frame, "end time:", end_dateTime, " processing Time: ", proc_time)
        print(f"Analysis complete")

if __name__ == '__main__':
    # Load the YOLO model
    # model = YOLO("runsyolov8shires/detect/train31/weights/last.pt")
    model = YOLO("runsyolov10shires/detect/train4/weights/last.pt")

    # assets for the application
    video_path = "clips/dl_gameplay_edited-fullgame.mp4"
    image_path = "screenshots/"
    image_name = "84_jpg.rf.3cbedbd35007e2d4fb1e75a3fa7ce16d"
    img_path = "assets/img/tbox_bg.png"
    font_path = "assets/fonts/Spiegel-TTF/Spiegel_TT_Regular.ttf"
    #cap = cv2.VideoCapture(video_path)

    byte_tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    #load llm
    myLLm = llm.LeagueLLM()
    #specify a rectangular bounding box for analysis
    rect = sv.Rect(450,650,1000,330)
    #load ocr
    reader = easyocr.Reader(['en'], detect_network='craft', gpu=True)
    #names of resources to read
    ocr_img_names = ["player-champion", "kda", "game-time", "creep-kill", "player-gold", "ally-champion"]
    advice_timer = 0
    show_advice = False
    text_bubble = None
    resource_dict = {"kda": None, "game-time": None, "creep-kill": None}
    required_resources = ['kda', 'creep-kill', 'game-time']
    kda_pattern = r'^\d{1,2}/\d{1,2}/\d{1,2}$'
    time_pattern = r'^\d{1,2}[:.]\d{1,2}$'
    frame_interval = 9000
    position = "adc"
    is_frame = False








    root = tk.Tk()
    app = LeagueMachineApp(root)
    root.mainloop()

    #image prediction
    # cv_image = cv2.imread(f"{image_path}/{image_name}.jpg")
    # current_dateTime = datetime.now()
    # this_image = process_frame(cv_image,0)

    # sv.Point(x=, y=)
    color_gray = sv.Color(128,128,128)



    # texts = text.replace('\n', "")
    # print(texts)
    # for line in lines:
    #     if line != '':
    #         this_image = add_text_to_image(
    #             this_image,
    #             line,
    #             font_scale=0.5,
    #             font_color_rgb=(0,0,0),
    #             bg_color_rgb=(128,128,128),
    #             outline_color_rgb=(255,255,255),
    #             top_left_xy=(450, 650+line_breaks*15),)
    #         line_breaks += 1

    # print(datetime.now() - current_dateTime)
    # cv2.imwrite(f"{image_path}/detections/text_{image_name}_yolov10s.jpg", this_image)

    # print(text_bubble)
    # success = cv2.imwrite(f"{image_path}/detections/sample.png", this_image)
    # print(success)


    # print(this_image)
    # this_image = add_analysis(toImgPIL(cv_image), resource_dict)
    # cv2.imshow("Text Bubble Image", this_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



# if __name__ == "__main__":
#     root = tk.Tk()
#     app = LeagueMachineApp(root)
#     root.mainloop()