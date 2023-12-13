from ultralytics import YOLO
import cv2

model = YOLO("yolo-Weights/yolov8n.pt")
model2 = YOLO("yolo-Weights/yolov8n.pt")

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

print("detect image:")
results = model(img)

detected_objects_with_quality = []

print()

for r in results:
    boxes = r.boxes

    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        subimg = img[y1:y2,x1:x2]

        cls = int(box.cls[0])

        print("detect subimage " + classNames[cls] + ":")
        result_sub_img = model(subimg)

        try:
            box_sub = result_sub_img[0].boxes[0]
            cls_sub = int(box.cls[0])

            print("quality: " + classNames[cls_sub] + "\n")
            detected_objects_with_quality.append(classNames[cls] + ": " + classNames[cls_sub])
        
        except Exception as e:
            print("An exception occurred:", str(e))
            detected_objects_with_quality.append(classNames[cls] + ": no object found")

print(detected_objects_with_quality)
