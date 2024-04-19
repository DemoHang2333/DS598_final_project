import cv2
import torch

from ultralytics.utils.plotting import Annotator, colors

model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt', force_reload=True).to('cuda' if torch.cuda.is_available() else 'cpu')

url = 'http://admin:admin@192.0.0.2:8081/video'
cap = cv2.VideoCapture(url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model([img], size=640)

    annotator = Annotator(frame, line_width=2, example=str(model.names))
    for *xyxy, conf, cls in results.xyxy[0]:
        c = int(cls)
        label = f'{model.names[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))

    cv2.imshow('YOLOv5 Detection', annotator.result())

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
