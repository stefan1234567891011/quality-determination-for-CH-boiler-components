from ultralytics import YOLO

import cv2
import os
from datetime import datetime

model = YOLO("/home/a/Documents/best.pt")
can_model = YOLO("/home/a/Documents/runs/classify/train3/weights/best.pt")

classNames = ["can", "remote"]
qualityNames = ["heavy damage", "light damage", "no damage"]


# img = cv2.imread('/home/a/Documents/datasets/Detect Cans - Remotes.v3i.yolov8/test/images/7710-108-_png.rf.ec470c8efde49b9658d2879d2318ead2.jpg')
# img = cv2.imread('/home/a/Documents/datasets/Detect Cans - Remotes.v3i.yolov8/test/images/7711-11-_png.rf.429a2bf8145b2eb7111f7278a2a66f70.jpg' )
img = cv2.imread('/home/a/Documents/datasets/Detect Cans - Remotes.v3i.yolov8/test/images/7729-45-_png.rf.5f7c04669b0ec2f5da1818ff306729bb.jpg')


results = model(img)

current_date = datetime.now()

current_date = current_date.strftime("%d-%m-%Y_%H-%M-%S")

dir = '/home/a/Documents/git/quality-determination-for-CH-boiler-components/scripts/POCs/runs/' + current_date

os.mkdir(dir)

boxes = results[0].boxes

for i in range(len(boxes)):
    box = boxes[i]
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    subimg = img[y1:y2,x1:x2]

    cls = int(box.cls[0])
    className = classNames[cls]


    if cls==0:
        print("can found")
        quality_pred = can_model(subimg)
        quality = qualityNames[quality_pred[0].probs.top1]
        print("\n quality: " + quality + "\n")
    elif cls==1:
        quality = ""
        print("remote found \n")

    img_name = str(i) + "_" + className + "_" + quality + ".jpg"

    temp_dir = (dir + "/" + className)
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    os.chdir(temp_dir)
    cv2.imwrite(img_name, subimg)
