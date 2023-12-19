from ultralytics import YOLO
import cv2
import math 
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("/home/a/Documents/best.pt")
# model = YOLO("/home/a/Documents/runs/detect/train11/weights/last.pt")
can_model = YOLO("/home/a/Documents/runs/classify/train3/weights/best.pt")

classNames = ["can", "remote"]
qualityNames = ["heavy damage", "light damage", "no damage"]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            subimg = img[y1:y2,x1:x2]

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            if cls==0:
                print("can found \n")
                quality_pred = can_model(subimg)
                quality = qualityNames[quality_pred[0].probs.top1]
                confidence = quality_pred[0].probs.top1conf.tolist()
                print(" quality: " + quality + "\n")
                print(" confidence: " + str(confidence) + "\n")
            elif cls==1:
                quality = ""
                print("remote found \n")

            # object details
            text = classNames[cls] + ": " + quality + " " + str(confidence)
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, text, org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
