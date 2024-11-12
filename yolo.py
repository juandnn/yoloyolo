# Importa las librerías necesarias
from roboflow import Roboflow
from ultralytics import YOLO

# descargar datos
rf = Roboflow(api_key="k")
project = rf.workspace("juan-ekotv").project("detector-juan-niebles")
version = project.version(1)

#entrenar modelo
dataset = version.download("yolov8")
model = YOLO("yolov8n.pt")
results = model.train(data=dataset.location + "/data.yaml", epochs=25, imgsz=640)
