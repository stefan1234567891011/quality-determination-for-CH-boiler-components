from ultralytics import YOLO

import cv2
import os
from datetime import datetime

model = YOLO("yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

img = cv2.imread('D:/school/stage HTES lectoraat/git/quality-determination-for-CH-boiler-components/scripts/test_camera_with_yolo/bus.jpg')

results = model(img)

current_date = datetime.now()

current_date = current_date.strftime("%H-%M-%S_%d-%m-%Y")

dir = 'D:/school/stage HTES lectoraat/git/quality-determination-for-CH-boiler-components/scripts/POCs/runs/' + current_date

os.mkdir(dir)

# for r in results:
# boxes = r.boxes
boxes = results[0].boxes

for i in range(len(boxes)):
    box = boxes[i]
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    subimg = img[y1:y2,x1:x2]

    cls = int(box.cls[0]) 

    img_name = str(i) + "_" + classNames[cls] + ".jpg"

    temp_dir = (dir + "/" + classNames[cls])
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    os.chdir(temp_dir)
    cv2.imwrite(img_name, subimg)
    