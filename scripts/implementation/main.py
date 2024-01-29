import general_controller
import camera_controller
import object_detector
import object_classifier

from ultralytics import YOLO
import cv2
import numpy as np

def main():
    model_path = "/home/a/Documents/best.pt"
    can_model_path = "/home/a/Documents/runs/classify/train3/weights/best.pt"
    
    classNames = ["can", "remote"]
    qualities = ["heavy damage", "light damage", "no damage"]

    camera = camera_controller.ZED2()
    detector = object_detector.object_detector(model_path, classNames)
    classifier = object_classifier.object_classifier(can_model_path, qualities)
    controller = general_controller.general_controller(camera, detector, classifier)

    image = controller.detect_obj_quality()

    while True:
        cv2.imshow('Webcam', image)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()