import cv2
from PIL import Image

from ultralytics import YOLO

model = YOLO("juanito.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo recibir el frame (stream end?). Saliendo...")
        break

    results = model.predict(source=frame, show=False)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Webcam Predictions", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # salir
        break

cap.release()
cv2.destroyAllWindows()



### Información útil #####

# results = model.predict(
#     source=frame,      # Procesa el frame capturado
#     conf=0.5,          # Confianza mínima de 50%
#     iou=0.4,           # IoU para NMS de 40%
#     show=True,         # Muestra la imagen con anotaciones
#     save=True,         # Guarda las predicciones
#     device='cuda',     # Ejecuta en la GPU si está disponible
#     classes=[0, 2],    # Solo detecta personas (clase 0) y coches (clase 2)
#     agnostic_nms=False # Supresión de no-máximos sensible a la clase
# )