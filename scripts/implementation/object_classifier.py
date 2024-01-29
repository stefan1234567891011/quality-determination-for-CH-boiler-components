from ultralytics import YOLO

class object_classifier:

    def __init__(self, path, qualities):
        self.path = path
        self.model = YOLO(path)
        self.qualities = qualities

    def classify_objects(self, sub_img):
        # classify object
        pred_quality = self.model(sub_img)
        quality = self.qualities[pred_quality[0].probs.top1]
        # return quality
        return quality