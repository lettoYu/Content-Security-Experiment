from ultralytics import YOLOv10

def main():
    model = YOLOv10('yolov10m_test.yaml').load('yolov10m.pt')

    model.train(
        data='my.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=4  # 建议小于等于4，适配 Windows
    )

if __name__ == '__main__':
    main()
