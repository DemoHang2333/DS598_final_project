import cv2
import torch

# 加载YOLOv5模型，确保使用GPU（如果可用）
from ultralytics.utils.plotting import Annotator, colors

model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/40_epochs/weights/best.pt', force_reload=True).to('cuda' if torch.cuda.is_available() else 'cpu')

# 打开电脑摄像头
url = 'http://admin:admin@192.0.0.2:8081/video'
cap = cv2.VideoCapture(url)

while cap.isOpened():
    ret, frame = cap.read()  # 读取一帧图像
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 将图像转换为PyTorch张量并进行推理
    results = model([img], size=640)  # size参数是可选的

    # 绘制检测结果
    annotator = Annotator(frame, line_width=2, example=str(model.names))
    for *xyxy, conf, cls in results.xyxy[0]:
        c = int(cls)  # 整数类别
        label = f'{model.names[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))

    # 将带有标注的图像显示出来
    cv2.imshow('YOLOv5 Detection', annotator.result())

    if cv2.waitKey(1) == ord('q'):  # 按'q'退出
        break

cap.release()
cv2.destroyAllWindows()
