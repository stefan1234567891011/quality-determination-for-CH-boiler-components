import cv2
import os

path = "D:\school\stage HTES lectoraat\git\quality-determination-for-CH-boiler-components\scripts"

os.chdir(path)

cam = cv2.VideoCapture(0)
s, img = cam.read()
if s:
    cv2.imwrite("filename.jpg",img)