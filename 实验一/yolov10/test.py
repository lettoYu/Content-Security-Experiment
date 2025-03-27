from ultralytics import YOLOv10
import cv2

# 加载训练好的YOLOv10模型
model = YOLOv10('runs/detect/train8/weights/best.pt')

# 类别中文名称 (根据你自己的类别定义调整)
class_names = ['orange', 'apple', 'banana']

# 进行推理
results = model.predict('example.jpg', imgsz=640, conf=0.25)

# 加载图片
img = cv2.imread('example.jpg')

# 绘制边框、类别和置信度
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f'{class_names[int(cls)]}: {score:.2f}'

        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签背景
        cv2.rectangle(img, (x1, y1-25), (x1+len(label)*15, y1), (0, 255, 0), -1)
        
        # 显示标签文字
        cv2.putText(img, label, (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)

# 显示图片结果（按任意键关闭）
cv2.imshow('YOLOv10 Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存可视化结果
cv2.imwrite('result.jpg', img)
