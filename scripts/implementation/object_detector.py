from ultralytics import YOLO

class object_detector:
    def __init__(self, path, class_names):
        self.path = path
        self.model = YOLO(path)
        self.class_names = class_names

    def detect_objects(self, img):
        # gebruik object recognition
        results = self.model(img)

        # print(results)
        
        boxes = results[0].boxes

        detected_objects = []

        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            subimg = img[y1:y2,x1:x2]

            cls = int(box.cls[0])

            detected_object = {
                "class_name": self.class_names[cls],
                "sub_image": subimg,
                "box": [x1, x2, y1, y2]
            }

            detected_objects.append(detected_object)        

        # return object naam met bounding box locations
        return detected_objects