from ultralytics import YOLO
import torch
import torchvision

print(torch.__version__)
print(torchvision.__version__)

print(torch.cuda.device_count())

cudaEnabled = torch.cuda.is_available()
print(cudaEnabled)
if cudaEnabled:
    print(torch.cuda.get_device_properties(0).name)

# model = YOLO("yolov8n.pt")
model = YOLO("yolov8n.pt")
# model.train(data='/home/a/Documents/datasets/Detect Cans - Remotes.v1i.yolov8/data.yaml', epochs=100, imgsz=256)
model.train(data='/home/a/Documents/datasets/Detect Cans - Remotes.v3i.yolov8/data.yaml', epochs=100, imgsz=640)
# model.train(data='/home/a/Documents/datasets/Detect Cans - Remotes.v4i.yolov8/data.yaml', epochs=200, imgsz=258)
