from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

results = model.train(data='/home/a/Documents/datasets/cans.v5i.folder', epochs=100, imgsz=224)
