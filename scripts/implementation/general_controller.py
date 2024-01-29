import camera_controller
import object_detector
import object_classifier
import cv2

class general_controller:
    def __init__(self, camera, detector, classifier):
        self.camera = camera
        self.detector = detector
        self.classifier = classifier

    def detect_obj_quality(self):
        img = self.camera.take_picture()
        image = cv2.cvtColor(img.get_data(), cv2.COLOR_RGBA2RGB)

        detected_obj = self.detector.detect_objects(image)

        for obj in detected_obj:
            quality = self.classifier.classify_objects(obj["sub_image"])
            print(obj["class_name"] + ": " + quality)
        
            x1, x2, y1, y2 = obj["box"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            text = obj["class_name"]+ ": " + quality
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(image, text, org, font, fontScale, color, thickness)

        return image