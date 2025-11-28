# Model definition
import numpy as np
import cv2
from ultralytics import YOLO

model=YOLO("yolo11n.pt")
results=model.train(data="data/data.yaml", epochs=10, imgsz=300)
model.save("model_11n_15e.h5")
